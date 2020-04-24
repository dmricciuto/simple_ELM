import numpy
import time
#from sklearn.neural_network import MLPRegressor
#import pickle

class MyModel(object):
    
    def __init__(self):
        self.name = 'DALEC'
        self.npfts  = 4
        self.parms = {'gdd_min':   numpy.zeros([self.npfts], numpy.float)+20.,   \
                      'ndays_on':    numpy.zeros([self.npfts], numpy.float)+30.,     \
                      'tsmin':     numpy.zeros([self.npfts], numpy.float)-2.0,   \
                      'leaffall':  numpy.zeros([self.npfts], numpy.float)+0.3,   \
                      'flnr':       numpy.zeros([self.npfts], numpy.float)+0.08,   \
                      'nue':       numpy.zeros([self.npfts], numpy.float)+15.0,   \
                      'lma':       numpy.zeros([self.npfts], numpy.float)+80.0,   \
                      'leafcn':    numpy.zeros([self.npfts], numpy.float)+25.0,   \
                      'rg_frac':   numpy.zeros([self.npfts], numpy.float)+0.3,   \
                      'aleaf':     numpy.zeros([self.npfts], numpy.float)+0.30,  \
                      'astem':     numpy.zeros([self.npfts], numpy.float)+0.50,   \
                      'troot':     numpy.zeros([self.npfts], numpy.float)+5.0,   \
                      'tstem':     numpy.zeros([self.npfts], numpy.float)+50.0,   \
                      'evergreen': numpy.zeros([self.npfts], numpy.int)+0, \
                      'br_mr':     numpy.zeros([1], numpy.float)+2.52e-6,   \
                      'q10_mr':    numpy.zeros([1], numpy.float)+2.0,   \
                      'q10_hr':    numpy.zeros([1], numpy.float)+2.0,   \
                      'br_lit':    numpy.zeros([1], numpy.float)+3.0,   \
                      'br_som':    numpy.zeros([1], numpy.float)+500.0,   \
                      'dr':        numpy.zeros([1], numpy.float)+0.001}
        self.pdef = self.parms.copy()        
        self.statevars = ['leafc_stor_pft', 'xsmr_pft', 'gdd_pft', 'stemc_pft','rootc_pft','litrc','somc','leafc_pft','lai_pft']  
        self.fluxvars  = ['mr_pft', 'gr_pft', 'gpp_pft', 'npp_pft', 'hr', 'nee', 'nee_lc_light']

        #SPRUCE-specific parameter values
        self.parms['evergreen'][0] = 1.0  #Black Spruce
        self.parms['evergreen'][3] = 1.0  #Moss
        self.parms['lma'][0] = 142.
        self.parms['lma'][1] = 35.
        self.parms['lma'][2] = 50.
        self.parms['lma'][3] = 140.
        #self.parms['tstem'][2] = 5.0
        self.parms['tstem'][3] = 1.0
        self.parms['troot'][3] = 1.0
        self.parms['astem'][3] = 0.05
        #self.parms['aleaf'][2] = 0.7
        self.parms['aleaf'][3] = 0.7
        #self.parms['flnr'][0] = 0.08
        #self.parms['flnr'][1] = 0.15
        #self.parms['flnr'][2] = 0.15
        #self.parms['flnr'][3] = 0.15

        #get neural network
        #pkl_filename = './GPP_model_NN/bestmodel_daily.pkl'
        #with open(pkl_filename, 'rb') as file:
        #  self.nnmodel = pickle.load(file)
        #nsamples=20000
        #self.nparms_nn = 14  #15
        #ptrain_orig   = (numpy.loadtxt('./GPP_model_NN/ptrain_daily.dat'))[0:nsamples,:]
        #self.pmin_nn = numpy.zeros([self.nparms_nn], numpy.float)
        #self.pmax_nn = numpy.zeros([self.nparms_nn], numpy.float)
        #for i in range(0,self.nparms_nn):
        #  self.pmin_nn[i] = min(ptrain_orig[:,i])
        #  self.pmax_nn[i] = max(ptrain_orig[:,i])

    def run_dalec(self, parms, ad_cycles=0, final_cycles=0, trans_startyear=-1, pftwt=-1):
        self.states={}
        self.fluxes={}
        
        if (pftwt == -1):
          pftwt = numpy.ones([self.npfts], numpy.float)/self.npfts
 
        #determine number of transient cycles
        if trans_startyear < 0:
          trans_startyear = self.start_year
        nyr_cycle = (self.end_year - self.start_year)+1
	trans_cycles = int(numpy.ceil((self.end_year - trans_startyear+1)/float(nyr_cycle)))
        self.model_startyear = self.end_year-nyr_cycle*trans_cycles+1        

        for v in self.statevars:
          if ('_pft' in v):
            self.states[v] = numpy.zeros([self.npfts, int(self.nobs*trans_cycles)+1], numpy.float)
          else:
            self.states[v] = numpy.zeros([int(self.nobs*trans_cycles)+1], numpy.float)
        for v in self.fluxvars:
          if ('_pft' in v):
            self.fluxes[v] = numpy.zeros([self.npfts, int(self.nobs*trans_cycles)+1], numpy.float)
          else:
            self.fluxes[v] = numpy.zeros([int(self.nobs*trans_cycles)+1], numpy.float)      

        #initial values
        for p in range(0,self.npfts):
          if (self.parms['evergreen'][p] == 0):
            self.states['leafc_stor_pft'][p,0] = 300.0
          else:
            self.states['leafc_pft'][p,0] = 100.0

        #Model parameters
        gdd_min = self.parms['gdd_min']
        ndays_on = self.parms['ndays_on']
        tsmin   = self.parms['tsmin']
        leaffall= self.parms['leaffall']
        nue     = self.parms['nue']
        rg_frac = self.parms['rg_frac']
        br_mr   = self.parms['br_mr']
        q10_mr  = self.parms['q10_mr']
        aleaf   = self.parms['aleaf']
        astem   = self.parms['astem']
        troot   = 1/(self.parms['troot']*365.0)
        tstem_base   = 1/(self.parms['tstem']*365.0)
        tstem = tstem_base.copy()
        q10_hr  = self.parms['q10_hr']
        br_lit  = 1/(self.parms['br_lit']*365.0)
        br_som_base  = 1/(self.parms['br_som']*365.0)
        dr      = self.parms['dr']
        lma     = self.parms['lma']
        leafcn  = self.parms['leafcn']

        #met_thistimestep_norm=numpy.zeros([1,self.nparms_nn], numpy.float)
        #Run the model
        ad_factor_stem = numpy.zeros([self.npfts], numpy.float)
        ad_factor_som =  max(self.parms['br_som']/5, 1.0)
        for p in range(0,self.npfts):
          ad_factor_stem[p] = max(self.parms['tstem'][p]/5, 1.0)

        for s in range(0,ad_cycles+final_cycles+trans_cycles):
          if (s > 0 and s <= ad_cycles+final_cycles):
            for var in self.states:
              if ('_pft' in var):
                for p in range(0,self.npfts):
                  self.states[var][p,0] = self.states[var][p,self.nobs]
              else:
                self.states[var][0] = self.states[var][self.nobs]
          if (s < ad_cycles):
            br_som = br_som_base*ad_factor_som
            for p in range(0,self.npfts):
              tstem[p]  = tstem_base[p]*ad_factor_stem[p]
          elif (s == ad_cycles):
            br_som = br_som_base 
            for p in range(0,self.npfts):
              tstem[p]  = tstem_base[p] 
            if (ad_cycles > 0):
              for p in range(0,self.npfts):
                self.states['stemc_pft'][p,0] = sum(self.states['stemc_pft'][p,0:self.nobs])/self.nobs*ad_factor_stem[p]
              self.states['somc'][0] = sum(self.states['somc'][0:self.nobs])/self.nobs*ad_factor_som

          this_trans_cycle = s - ad_cycles-final_cycles

          #set PFT-level temporary variables           
          leafout_fromstor = numpy.zeros([self.npfts], numpy.float)
          ndays_leafout = numpy.zeros([self.npfts], numpy.float)

          for tf in range(0,self.nobs):
            v = max(this_trans_cycle,0)*self.nobs + tf
            model_year = max(self.model_startyear, self.model_startyear+(this_trans_cycle*self.nobs+tf)/365.)

            if (int(model_year) == 1974 and self.forcings['doy'][tf] == 1):
               #Strip cut harvest the trees
               self.states['stemc_pft'][0:2,v] = 0.01 * self.states['stemc_pft'][0:2,v]
               self.states['leafc_pft'][0:2,v] = 0.01 * self.states['leafc_pft'][0:2,v]
               self.states['leafc_stor_pft'][0:2,v] = 0.01 * self.states['leafc_stor_pft'][0:2,v]
               self.states['rootc_pft'][0:2,v] = 0.01 * self.states['rootc_pft'][0:2,v]
               self.states['xsmr_pft'][0:2,v] = 0.01 * self.states['xsmr_pft'][0:2,v]
               #leaf, root to litter
               self.states['litrc'][v] = self.states['litrc'][v] + pftwt[0] * (90*self.states['leafc_pft'][0,v] + \
                                      99*self.states['rootc_pft'][0,v])
               self.states['litrc'][v] = self.states['litrc'][v] + pftwt[1] * (90*self.states['leafc_stor_pft'][1,v] + \
                                      99*self.states['rootc_pft'][1,v])

            #set PFT-level temporary variables
            leafc_on = numpy.zeros([self.npfts], numpy.float)
            leafc_off = numpy.zeros([self.npfts], numpy.float)
            leaf_litter = 0
            stem_litter = 0
            root_litter = 0
            vegc = 0
            vegc_last = 0
            for p in range(0,self.npfts):
              a = [nue[p], 0.0156935, 4.22273, 208.868, 0.0453194, 0.37836, 7.19298, 0.011136, \
                   2.1001, 0.789798]
              #Phenology
              self.states['gdd_pft'][p,v+1] = (self.forcings['doy'][tf] > 1) * (self.states['gdd_pft'][p,v] + \
                       max(0.5*(self.forcings['tmax'][tf]+self.forcings['tmin'][tf])-10.0, 0.0))
              if (self.parms['evergreen'][p] == 0):
                if (self.forcings['doy'][tf] < 200):
                  if (self.states['gdd_pft'][p,v+1] > gdd_min[p] and self.states['gdd_pft'][p,v] < gdd_min[p]):
                    leafout_fromstor[p] = self.states['leafc_stor_pft'][p,v] / 2.0
                    ndays_leafout[p] = ndays_on[p]
                  if (ndays_leafout[p] > 0):
                    leafc_on[p] = leafout_fromstor[p] / ndays_on[p]
                    ndays_leafout[p] = ndays_leafout[p] - 1
                elif (self.forcings['tmin'][tf] < tsmin[p] and self.states['lai_pft'][p,v] > 0):
                  leafc_off[p] = min(leaffall[p]*leafout_fromstor[p], self.states['leafc_pft'][p,v])
              else:
                leafc_off[p] = self.states['leafc_pft'][p,v] / (3.0 * 365) 
              #if (p == 1):
              #  print self.forcings['doy'][tf], self.states['lai_pft'][p,v], ndays_leafout[p], leafc_off[p]
              #  time.sleep(0.03)
              #Calculate GPP flux
              if (self.states['lai_pft'][p,v] > 1e-3):
                rtot = 1.0
                psid = -2.0
                leafn = lma[p]/leafcn[p]
                if (self.forcings['tmax'][tf]+self.forcings['tmin'][tf])/2 > 0:
                  gs = abs(psid)**a[9]/((a[5]*rtot+(self.forcings['tmax'][tf]-self.forcings['tmin'][tf])))
                  pp = max(self.states['lai_pft'][p,v], 0.5)*leafn/gs*a[0]*numpy.exp(a[7]*self.forcings['tmax'][tf])
                  qq = a[2]-a[3]
                  #internal co2 concentration
                  ci = 0.5*(self.forcings['cair'][tf]+qq-pp+((self.forcings['cair'][tf]+qq-pp)**2-4.* \
                           (self.forcings['cair'][tf]*qq-pp*a[2]))**0.5)
                  e0 = a[6]*max(self.states['lai_pft'][p,v],0.5)**2/(max(self.states['lai_pft'][p,v],0.5)**2+a[8])
                  cps   = e0*self.forcings['rad'][tf]*gs*(self.forcings['cair'][tf]-ci)/ \
                          (e0*self.forcings['rad'][tf]+gs*(self.forcings['cair'][tf]-ci))
                  self.fluxes['gpp_pft'][p,v+1] = cps*(a[1]*self.forcings['dayl'][tf]+a[4])
                  #ACM not valid at LAI < 0.5.  Scale linearly
                  if (self.states['lai_pft'][p,v] < 0.5):
                    self.fluxes['gpp_pft'][p,v+1] = self.fluxes['gpp_pft'][p,v+1]*self.states['lai_pft'][p,v]/0.5
                else:
                  self.fluxes['gpp_pft'][p,v+1] = 0.0
              else:
                self.fluxes['gpp_pft'][p,v+1] = 0.0
              
              #Use the Neural network trained with ELM data
              #dayl_factor = (self.forcings['dayl'][tf]/max(self.forcings['dayl'][0:365]))**2.0
              #flnr = self.parms['flnr'][p]
              #if (v < 10):
              #  t10 = (self.forcings['tmax'][tf]+self.forcings['tmin'][tf])/2.0+273.15
              #else:
              #  t10 = sum(self.forcings['tmax'][tf-10:tf]+self.forcings['tmin'][tf-10:tf])/20.0+273.15
              ##Use the NN trained on daily data
              #slatop = 1.0/lma[p]
              #met_thistimestep=[1.0, self.states['lai_pft'][p,v], self.states['lai_pft'][p,v]/4.0, self.forcings['tmax'][tf]+273.15, \
              #                  self.forcings['tmin'][tf]+273.15, t10, self.forcings['rad'][tf]*1e6, 50.0, self.forcings['cair'][tf]/10.0, \
              #                  dayl_factor, flnr, slatop, leafcn[p], 9.0]
              #for i in range(0,self.nparms_nn):   #normalize
              #  met_thistimestep_norm[0,i] = ( met_thistimestep[i] - self.pmin_nn[i] ) / \
              #           (self.pmax_nn[i] - self.pmin_nn[i])
              #self.fluxes['gpp_pft'][p,v+1] = max(self.nnmodel.predict(met_thistimestep_norm), 0.0)

              #Autotrophic repiration fluxes
              trate = q10_mr**((0.5*(self.forcings['tmax'][tf]+self.forcings['tmin'][tf])-20)/10.0)
              if (0.5*(self.forcings['tmax'][tf]+self.forcings['tmin'][tf]) < 0):
                trate = 0.0    #no maintenance respiration if air temperature below freezing
              self.fluxes['mr_pft'][p,v] = (self.states['leafc_pft'][p,v] / parms['leafcn'][p] + \
                                              0.1*self.states['rootc_pft'][p,v] / 42.0)*br_mr*86400*trate
              self.fluxes['gr_pft'][p,v] = max(rg_frac[p]*(self.fluxes['gpp_pft'][p,v] - \
                                             self.fluxes['mr_pft'][p,v]), 0.0)
              self.fluxes['npp_pft'][p,v] = self.fluxes['gpp_pft'][p,v] - self.fluxes['mr_pft'][p,v] - self.fluxes['gr_pft'][p,v]
              if (p >= 2):
                self.fluxes['nee_lc_light'][v] = self.fluxes['nee_lc_light'][v] - pftwt[p] * (self.fluxes['gpp_pft'][p,v]*24. \
                                               /self.forcings['dayl'][tf] -self.fluxes['mr_pft'][p,v] - self.fluxes['gr_pft'][p,v])
                self.fluxes['nee'][v] = self.fluxes['nee'][v] - pftwt[p] * self.fluxes['npp_pft'][p,v]
              elif (p == 0):
                self.fluxes['nee_lc_light'][v] = pftwt[p] *  (0.1*self.states['rootc_pft'][p,v] / 42.0)*br_mr*86400*trate                
                self.fluxes['nee'][v] = -1.0 * pftwt[p] * self.fluxes['npp_pft'][p,v]
              elif (p == 1):
                self.fluxes['nee_lc_light'][v] = self.fluxes['nee_lc_light'][v] + \
                                                    pftwt[p] * (0.1*self.states['rootc_pft'][p,v] / 42.0)*br_mr*86400*trate
                self.fluxes['nee'][v] = self.fluxes['nee'][v] - pftwt[p] * self.fluxes['npp_pft'][p,v]

              #Update PFT-level variabiles
              npp_alloc = max(self.fluxes['npp_pft'][p,v], 0.0)
              npp_alloc_to_xsmr = min(-1.0*self.states['xsmr_pft'][p,v]/30.0, npp_alloc)
              self.states['xsmr_pft'][p,v+1] = self.states['xsmr_pft'][p,v] + min(self.fluxes['npp_pft'][p,v], 0.0) + npp_alloc_to_xsmr
              npp_alloc = npp_alloc - npp_alloc_to_xsmr

              leaf_litter = leaf_litter + pftwt[p] * leafc_off[p]
              stem_litter = stem_litter + pftwt[p] * (tstem[p] * self.states['stemc_pft'][p,v])
              root_litter = root_litter + pftwt[p] * (troot[p] * self.states['rootc_pft'][p,v])
              if (self.parms['evergreen'][p] == 0):
                self.states['leafc_stor_pft'][p,v+1] = self.states['leafc_stor_pft'][p,v] + npp_alloc*aleaf[p] - leafc_on[p]
                self.states['leafc_pft'][p,v+1] = self.states['leafc_pft'][p,v] + leafc_on[p] - leafc_off[p]
              else:
                self.states['leafc_pft'][p,v+1] = self.states['leafc_pft'][p,v] + npp_alloc*aleaf[p] - leafc_off[p]

              self.states['stemc_pft'][p,v+1] = self.states['stemc_pft'][p,v] - (tstem[p] * self.states['stemc_pft'][p,v]) + \
                                                npp_alloc*astem[p]
              self.states['rootc_pft'][p,v+1] = self.states['rootc_pft'][p,v] - (troot[p] * self.states['rootc_pft'][p,v]) + \
                                                npp_alloc*(1.0-aleaf[p]-astem[p]) 
              vegc = vegc + pftwt[p] * (self.states['leafc_pft'][p,v+1] + self.states['stemc_pft'][p,v+1] + self.states['rootc_pft'][p,v+1])
              vegc_last = vegc_last + pftwt[p] * (self.states['leafc_pft'][p,v] + self.states['stemc_pft'][p,v] + self.states['rootc_pft'][p,v])
              self.states['lai_pft'][p,v+1] = self.states['leafc_pft'][p,v] / lma[p] 


            #Update vegetation and litter pools (allocation, litterfall and decomp)
            trate = q10_hr**((0.5*(self.forcings['tmax'][tf]+self.forcings['tmin'][tf])-10)/10.0)
            self.states['litrc'][v+1] = self.states['litrc'][v] + leaf_litter + stem_litter + root_litter - \
                                dr*self.states['litrc'][v] - br_lit*self.states['litrc'][v]*trate
            self.states['somc'][v+1] = self.states['somc'][v] + dr*self.states['litrc'][v] - \
                                             br_som*self.states['somc'][v]*trate
            self.fluxes['hr'][v] = br_lit*self.states['litrc'][v]*trate + br_som*self.states['somc'][v]*trate
            self.fluxes['nee_lc_light'][v] = (self.fluxes['nee_lc_light'][v] + self.fluxes['hr'][v]) / self.forcings['dayl'][tf]  #gC/m2/hr (daytime)
            self.fluxes['nee'][v] = self.fluxes['nee'][v] + self.fluxes['hr'][v]

    def generate_synthetic_obs(self, parms, err):
        #generate synthetic observations from model with Gaussian error
        self.obs = numpy.zeros([self.nobs], numpy.float)
        self.obs_err = numpy.zeros([self.nobs], numpy.float)+err
        self.run(parms)
        for v in range(0,self.nobs):
            self.obs[v] = self.fluxes[v]+numpy.random.normal(0,err,1)
        self.issynthetic = True
