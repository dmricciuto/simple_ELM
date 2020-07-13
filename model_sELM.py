from sklearn.neural_network import MLPRegressor
import pickle
import numpy
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians
import math, time, os
import utils
import forcings

class MyModel(object):

    def __init__(self):
        self.name = 'sELM'
        self.npfts = 4
        self.parms = {'gdd_crit':   numpy.zeros([self.npfts], numpy.float)+500.,   \
                      'crit_dayl':  numpy.zeros([self.npfts], numpy.float)+39300., \
                      'ndays_on':   numpy.zeros([self.npfts], numpy.float)+30.,    \
                      'ndays_off':  numpy.zeros([self.npfts], numpy.float)+15.,    \
                      'nue':        numpy.zeros([self.npfts], numpy.float)+15.,    \
                      'flnr':       numpy.zeros([self.npfts], numpy.float)+0.10,   \
                      'slatop':     numpy.zeros([self.npfts], numpy.float)+0.03,   \
                      'leafcn':     numpy.zeros([self.npfts], numpy.float)+25.0,   \
                      'leaflitcn':  numpy.zeros([self.npfts], numpy.float)+50.9,   \
                      'livewdcn':   numpy.zeros([self.npfts], numpy.float)+50,     \
                      'frootcn':    numpy.zeros([self.npfts], numpy.float)+42.0,   \
		      'deadwdcn':   numpy.zeros([self.npfts], numpy.float)+500,    \
                      'mbbopt':     numpy.zeros([self.npfts], numpy.float)+9.0,    \
                      'roota_par':  numpy.zeros([self.npfts], numpy.float)+7.0,    \
                      'rootb_par':  numpy.zeros([self.npfts], numpy.float)+2.0,    \
                      'fstor2tran': numpy.zeros([self.npfts], numpy.float)+0.5,    \
                      'stem_leaf':  numpy.zeros([self.npfts], numpy.float)-2.7,    \
                      'croot_stem': numpy.zeros([self.npfts], numpy.float)+0.3,    \
                      'f_livewd':   numpy.zeros([self.npfts], numpy.float)+0.1,    \
                      'froot_leaf': numpy.zeros([self.npfts], numpy.float)+1.0,    \
                      'rg_frac':    numpy.zeros([self.npfts], numpy.float)+0.3,    \
                      'br_mr':      numpy.zeros([self.npfts], numpy.float)+2.52e-6,\
                      'q10_mr':     numpy.zeros([self.npfts], numpy.float)+1.5,    \
                      'cstor_tau':  numpy.zeros([self.npfts], numpy.float)+3.0,    \
                      'leaf_long':  numpy.zeros([self.npfts], numpy.float)+3.0,    \
                      'froot_long': numpy.zeros([self.npfts], numpy.float)+3.0,    \
                      'season_decid': numpy.zeros([self.npfts], numpy.int)+1,      \
		      'r_mort':     numpy.zeros([1], numpy.float)+0.02,            \
                      'lwtop_ann':  numpy.zeros([1], numpy.float)+0.7,             \
                      'q10_hr':     numpy.zeros([1], numpy.float)+1.5,             \
                      'k_l1':       numpy.zeros([1], numpy.float)+1.2039728,       \
                      'k_l2':       numpy.zeros([1], numpy.float)+0.0725707,       \
                      'k_l3':       numpy.zeros([1], numpy.float)+0.0140989,       \
                      'k_s1':       numpy.zeros([1], numpy.float)+0.0725707,       \
                      'k_s2':       numpy.zeros([1], numpy.float)+0.0140989244,    \
                      'k_s3':       numpy.zeros([1], numpy.float)+0.00140098,      \
                      'k_s4':       numpy.zeros([1], numpy.float)+0.0001,          \
                      'k_frag':     numpy.zeros([1], numpy.float)+0.0010005,       \
                      'rf_l1s1':    numpy.zeros([1], numpy.float)+0.39,            \
                      'rf_l2s2':    numpy.zeros([1], numpy.float)+0.55,            \
                      'rf_l3s3':    numpy.zeros([1], numpy.float)+0.29,            \
                      'rf_s1s2':    numpy.zeros([1], numpy.float)+0.28,            \
                      'rf_s2s3':    numpy.zeros([1], numpy.float)+0.46,            \
                      'rf_s3s4':    numpy.zeros([1], numpy.float)+0.55,            \
                      'soil4ci':    numpy.zeros([1], numpy.float)+100.,              \
                      'cwd_flig':   numpy.zeros([1], numpy.float)+0.24,            \
                      'fr_flig':    numpy.zeros([1], numpy.float)+0.25,            \
                      'lf_flig':    numpy.zeros([1], numpy.float)+0.25,            \
                      'fr_flab':    numpy.zeros([1], numpy.float)+0.25,            \
                      'lf_flab':    numpy.zeros([1], numpy.float)+0.25,            \
                      'fpi':        numpy.zeros([1], numpy.float)+0.1,             \
                      'fpg':        numpy.zeros([1], numpy.float)+0.9}
        self.site='none'
        self.nsoil_layers=10
        self.pdefault={}
        self.pmin = {}
        self.pmax = {}
        self.nparms = 0
         
        #PFT-specific mods
        #PFT 1 = Evergreen tree
        #PFT 2 = Deciduous tree
        #PFT 3 = Shrub
        #PFT 4 = Sphagnum 
        self.parms['season_decid'][0] = 0
        self.parms['season_decid'][3] = 0
        #SPRUCE optimization
        self.parms['cstor_tau'][2] = 1.0
        self.parms['cstor_tau'][3] = 1.0
        self.parms['leafcn'][0] = 57.265
        self.parms['leafcn'][1] = 64.233
        self.parms['leafcn'][2] = 36.701
        self.parms['leafcn'][3] = 42.555
        self.parms['leaflitcn'][0] = 57.265*2
        self.parms['leaflitcn'][1] = 64.233*2
        self.parms['leaflitcn'][2] = 36.701*2
        self.parms['leaflitcn'][3] = 42.555*2
        self.parms['flnr'][0] = 0.1305
        self.parms['flnr'][1] = 0.2410
        self.parms['flnr'][2] = 0.1026
        self.parms['flnr'][3] = 0.1642
        self.parms['stem_leaf'][0] = -0.5174
        self.parms['stem_leaf'][1] = -0.7836
        self.parms['stem_leaf'][2] = 0.0515
        self.parms['stem_leaf'][3] = 0.0 
        self.parms['croot_stem'][0] = 0.2396
        self.parms['croot_stem'][1] = 0.2452
        self.parms['croot_stem'][2] = 0.4320
        self.parms['croot_stem'][3] = 0
        self.parms['slatop'][1] = 0.023845
        self.parms['slatop'][0] = 0.00700
        self.parms['slatop'][2] = 0.01696
        self.parms['slatop'][3] = 0.007595
        self.parms['froot_leaf'][0] = 1.807
        self.parms['froot_leaf'][1] = 1.0345
        self.parms['froot_leaf'][2] = 1.0547
        self.parms['froot_leaf'][3] = 0.7774
        self.parms['r_mort'][0] = 0.03516
        self.parms['br_mr'][:] = 3.245e-6
        self.parms['rg_frac'][:] = 0.401
        self.parms['q10_mr'][:] = 2.08325
        self.parms['lwtop_ann'][0] = 0.884018

        for p in self.parms:
            self.pdefault[p] = self.parms[p]
            if (p == 'crit_dayl'):
                self.pmin[p] = self.parms[p]*0.0+36000.
                self.pmax[p] = self.parms[p]*0.0+43000.
            elif (p == 'gdd_crit'):
                self.pmin[p] = self.parms[p][:]*0.0+100.
                self.pmax[p] = self.parms[p][:]*0.0+700.
            elif (p == 'fpg'):
                self.pmin[p] = self.parms[p][:]*0.0+0.70
                self.pmax[p] = self.parms[p][:]*0.0+0.95
            elif (p == 'nue'):
                self.pmin[p] = 0.50*self.parms[p][:]
                self.pmax[p] = 2.50*self.parms[p][:]
            elif (not 'season_decid' in p):
                self.pmin[p] = self.parms[p]*0.50
                self.pmax[p] = self.parms[p]*1.50
            self.nparms = self.nparms+len(self.parms[p])
        self.issynthetic = False
        self.ne = 1

        #Model outputs
        self.outvars = ['gpp_pft','npp_pft','gr_pft', 'mr_pft','hr','nee','lai_pft','leafc_pft','leafc_stor_pft','frootc_pft', \
                        'frootc_stor_pft','livestemc_pft','deadstemc_pft','livecrootc_pft','deadcrootc_pft','totecosysc', \
                        'totsomc','totlitc','cstor_pft','sminn_vr','nstor_pft','ndep','nfix','fpg_pft','fpi_vr','cwdc','totlitn','ctcpools_vr']

        #get neural network
        pkl_filename = './GPP_model_NN/bestmodel_daily.pkl'
        with open(pkl_filename, 'rb') as file:
          self.nnmodel = pickle.load(file)
        nsamples=20000
        self.nparms_nn = 14  #15
        ptrain_orig   = (numpy.loadtxt('./GPP_model_NN/ptrain_daily.dat'))[0:nsamples,:]
        self.pmin_nn = numpy.zeros([self.nparms_nn], numpy.float)
        self.pmax_nn = numpy.zeros([self.nparms_nn], numpy.float)
        for i in range(0,self.nparms_nn):
          self.pmin_nn[i] = min(ptrain_orig[:,i])
          self.pmax_nn[i] = max(ptrain_orig[:,i])

    def selm_instance(self, parms, use_nn=False, spinup_cycles=0, pftwt=[1.0,0,0]):

        calc_nlimitation = True
        npfts = self.npfts
        #--------------- Initialize ------------------------
        self.output={}
        for v in self.outvars:
          if ('_pft' in v):
            self.output[v] = numpy.zeros([self.npfts,int(self.nobs)+1])
          elif ('_vr' in v):
            if ('ctcpools' in v):
              self.output[v] = numpy.zeros([16,self.nsoil_layers,int(self.nobs)+1])
            else:
              self.output[v] = numpy.zeros([self.nsoil_layers,int(self.nobs)+1])
          else:
            self.output[v] = numpy.zeros([int(self.nobs)+1])
        #Flux variables
        gpp = self.output['gpp_pft']
        npp = self.output['npp_pft']
        gr  = self.output['gr_pft']
        mr  = self.output['mr_pft']
        hr  = self.output['hr']
        nee = self.output['nee']
        #State variables
        lai         = self.output['lai_pft']
        leafc       = self.output['leafc_pft']
        leafc_stor  = self.output['leafc_stor_pft']
        frootc      = self.output['frootc_pft']
        frootc_stor = self.output['frootc_stor_pft']
        livestemc   = self.output['livestemc_pft']
        deadstemc   = self.output['deadstemc_pft']
        livecrootc  = self.output['livecrootc_pft']
        deadcrootc  = self.output['deadcrootc_pft']
        totecosysc  = self.output['totecosysc']
        totsomc     = self.output['totsomc']
        totlitc     = self.output['totlitc']
        cstor       = self.output['cstor_pft']
        sminn_vr    = self.output['sminn_vr']
        nstor       = self.output['nstor_pft']
        ndep        = self.output['ndep']
        nfix        = self.output['nfix']
        fpg         = self.output['fpg_pft']
        fpi_vr      = self.output['fpi_vr']
        cwdc        = self.output['cwdc']
        totlitn     = self.output['totlitn']
        ctcpools_vr = self.output['ctcpools_vr']

        #vertically resolved variables (local)
        root_frac = numpy.zeros([npfts,self.nsoil_layers], numpy.float)
        surf_prof = numpy.zeros([self.nsoil_layers], numpy.float)
        depth_scalar = numpy.zeros([self.nsoil_layers], numpy.float)+1.0

        #set soil layers
        decomp_depth_efolding = 0.3743
        if (self.nsoil_layers > 1):
          soil_nodes = numpy.zeros([self.nsoil_layers], numpy.float)
          soil_dz    = numpy.zeros([self.nsoil_layers], numpy.float)
          soil_hi    = numpy.zeros([self.nsoil_layers], numpy.float)
          soil_depth = numpy.zeros([self.nsoil_layers], numpy.float)
          for i in range(0,self.nsoil_layers):
            soil_nodes[i] = 0.025*(numpy.exp(0.5*(i+0.5))-1)
          for i in range(0,self.nsoil_layers):
            if (i == 0):
              soil_dz[i] = 0.5*(soil_nodes[0]+soil_nodes[1])
              soil_hi[i] = 0.5*(soil_nodes[i]+soil_nodes[i+1])
              soil_depth[i] = soil_dz[i]/2.0
            elif (i < self.nsoil_layers-1):
              soil_dz[i] = 0.5*(soil_nodes[i+1]-soil_nodes[i-1])
              soil_hi[i] = 0.5*(soil_nodes[i]+soil_nodes[i+1])
              soil_depth[i] = soil_hi[i-1]+soil_dz[i]/2.0
            else: 
              soil_dz[i] = 1.5058
              soil_hi[i] = 3.8018   #layer 10 from CLM
              soil_depth[i] = soil_hi[i-1]+soil_dz[i]/2.0 
            depth_scalar[i] = math.exp(-soil_depth[i] / decomp_depth_efolding)
          for i in range(0,self.nsoil_layers):
           for p in range(0,npfts):
            #Figure out root fraction
            if (i == 0):
              root_frac[p,i] = 0.5*(numpy.exp(-1.0*parms['roota_par'][p]*0.0)+ \
                                numpy.exp(-1.0*parms['rootb_par'][p]*0.0) - \
                                numpy.exp(-1.0*parms['roota_par'][p]*soil_hi[i]) - \
                                numpy.exp(-1.0*parms['rootb_par'][p]*soil_hi[i]) )
            else:
              root_frac[p,i] = 0.5*(numpy.exp(-1.0*parms['roota_par'][p]*soil_hi[i-1])+ \
                                numpy.exp(-1.0*parms['rootb_par'][p]*soil_hi[i-1]) - \
                                numpy.exp(-1.0*parms['roota_par'][p]*soil_hi[i]) - \
                                numpy.exp(-1.0*parms['rootb_par'][p]*soil_hi[i]) )
           surf_prof[i] = (numpy.exp(-10.0*soil_nodes[i])) / soil_dz[i]
        else:
          surf_prof[0] = 1.0
          root_frac[:,0] = 1.0

        for p in range(0,npfts):
          root_frac[p,:] = root_frac[p,:]/sum(root_frac[p,:])
        surf_prof = surf_prof/sum(surf_prof)

        #Set nonzero initial States 
        for p in range(0,npfts):
          if parms['season_decid'][p] == 1:
            leafc_stor[p,0] = 10.0
          else:
            leafc[p,0]      = 10.0
         
        nstor[:,0]       = 1.0
        ctcpools_vr[6,:,0] = parms['soil4ci']
        ctcpools_vr[14,:,0] = parms['soil4ci']/10.0   #ASSUME CN of 10

        #Forcings
        tmax = self.forcings['tmax']
        tmin = self.forcings['tmin']
        rad  = self.forcings['rad']
        doy  = self.forcings['doy']
        cair = self.forcings['cair']
        dayl = self.forcings['dayl']
        btran = self.forcings['btran']
        #Coefficents for ACM (GPP submodel)
        a=numpy.zeros([npfts,10], numpy.float)
        for p in range(0,npfts):
          a[p,:] = [parms['nue'][p], 0.0156935, 4.22273, 208.868, 0.0453194, 0.37836, 7.19298, 0.011136, \
               2.1001, 0.789798]

        #Turnover times for CTC model
        k_ctc = [parms['k_l1'],parms['k_l2'],parms['k_l3'],parms['k_s1'], \
                 parms['k_s2'],parms['k_s3'],parms['k_s4'],parms['k_frag']]
        #Respiration fractions for CTC model pools
        rf_ctc = [parms['rf_l1s1'],parms['rf_l2s2'],parms['rf_l3s3'] , \
                  parms['rf_s1s2'],parms['rf_s2s3'],parms['rf_s3s4'], 1.0, 0.0]
        #transfer matrix for CTC model
        tr_ctc = numpy.zeros([8,8],numpy.float)
        tr_ctc[0,3] = 1.0 - parms['rf_l1s1']
        tr_ctc[1,4] = 1.0 - parms['rf_l2s2']
        tr_ctc[2,5] = 1.0 - parms['rf_l3s3']
        tr_ctc[3,4] = 1.0 - parms['rf_s1s2']
        tr_ctc[4,5] = 1.0 - parms['rf_s2s3']
        tr_ctc[5,6] = 1.0 - parms['rf_s3s4']
        tr_ctc[7,1] = parms['cwd_flig']
        tr_ctc[7,2] = 1.0 - parms['cwd_flig']

        #Initialize local variables
        gdd     = numpy.zeros([npfts], numpy.float)+0.0
        leafon  = numpy.zeros([npfts], numpy.float)+0.0
        leafoff = numpy.zeros([npfts], numpy.float)+0.0
        leafc_trans  = numpy.zeros([npfts], numpy.float)+0.0
        frootc_trans = numpy.zeros([npfts], numpy.float)+0.0
        leafc_trans_tot  = numpy.zeros([npfts], numpy.float)+0.0
        frootc_trans_tot = numpy.zeros([npfts], numpy.float)+0.0
        leafc_litter = numpy.zeros([npfts], numpy.float)+0.0
        frootc_litter = numpy.zeros([npfts], numpy.float)+0.0
        leafc_litter_tot = numpy.zeros([npfts], numpy.float)+0.0
        frootc_litter_tot = numpy.zeros([npfts], numpy.float)+0.0
        leafn_litter = numpy.zeros([npfts], numpy.float)+0.0
        livestemc_turnover = numpy.zeros([npfts], numpy.float)+0.0
        livecrootc_turnover = numpy.zeros([npfts], numpy.float)+0.0
        annsum_npp = numpy.zeros([npfts], numpy.float)+0.0
        annsum_npp_temp = numpy.zeros([npfts], numpy.float)+0.0
        retransn = numpy.zeros([npfts], numpy.float)+0.0
        annsum_retransn = numpy.zeros([npfts], numpy.float)+0.0
        annsum_retransn_temp = numpy.zeros([npfts], numpy.float)+0.0
        annsum_gpp = numpy.zeros([npfts], numpy.float)+1000.0
        annsum_gpp_temp = numpy.zeros([npfts], numpy.float)+1000.0
        availc = numpy.zeros([npfts], numpy.float)+0.0
        cstor_alloc = numpy.zeros([npfts], numpy.float)+0.0
        xsmr = numpy.zeros([npfts], numpy.float)+0.0
        callom = numpy.zeros([npfts], numpy.float)+0.0
        nallom = numpy.zeros([npfts], numpy.float)+0.0
        leafc_alloc = numpy.zeros([npfts], numpy.float)+0.0
        leafcstor_alloc = numpy.zeros([npfts], numpy.float)+0.0
        frootc_alloc = numpy.zeros([npfts], numpy.float)+0.0
        frootcstor_alloc = numpy.zeros([npfts], numpy.float)+0.0
        livestemc_alloc = numpy.zeros([npfts], numpy.float)+0.0
        deadstemc_alloc = numpy.zeros([npfts], numpy.float)+0.0
        livecrootc_alloc = numpy.zeros([npfts], numpy.float)+0.0
        deadcrootc_alloc = numpy.zeros([npfts], numpy.float)+0.0
        plant_ndemand = numpy.zeros([npfts], numpy.float)+0.0
        plant_nalloc = numpy.zeros([npfts], numpy.float)+0.0
        cstor_turnover = numpy.zeros([npfts], numpy.float)+0.0
        
        met_thistimestep_norm=numpy.zeros([1,self.nparms_nn], numpy.float)
        #Run the model
        for s in range(0,spinup_cycles+1):
          totecosysc_last = totecosysc[0]
          if (s > 0):
            for p in range(0,npfts):
              leafc_stor[p,0]  = leafc_stor[p,self.nobs-1]
              leafc[p,0]       = leafc[p,self.nobs-1]
              frootc_stor[p,0] = frootc_stor[p,self.nobs-1]
              frootc[p,0]      = frootc[p,self.nobs-1]
              livestemc[p,0]   = livestemc[p,self.nobs-1]
              deadstemc[p,0]   = deadstemc[p,self.nobs-1]
              livecrootc[p,0]  = livecrootc[p,self.nobs-1]
              deadcrootc[p,0]  = deadcrootc[p,self.nobs-1]
              cstor[p,0]       = cstor[p,self.nobs-1]
              nstor[p,0]       = nstor[p,self.nobs-1]
            for nl in range(0,self.nsoil_layers):
              ctcpools_vr[:,nl,0]  = ctcpools_vr[:,nl,self.nobs-1]
              sminn_vr[nl,0]       = sminn_vr[nl,self.nobs-1]
            if (s == spinup_cycles and spinup_cycles > 0):
              #accelerated mortality and spinup
              for p in range(0,npfts):
                deadstemc[p,0] = deadstemc[p,0]*10
                deadcrootc[p,0] = deadcrootc[p,0]*10
              for nl in range(0,self.nsoil_layers):
                ctcpools_vr[5,nl,0] = ctcpools_vr[5,nl,0]*5.0
                ctcpools_vr[6,nl,0] = ctcpools_vr[6,nl,0]*30.0
                ctcpools_vr[7,nl,0] = ctcpools_vr[7,nl,0]*3.0
                ctcpools_vr[13,nl,0] = ctcpools_vr[13,nl,0]*5.0
                ctcpools_vr[14,nl,0] = ctcpools_vr[14,nl,0]*30.0
                ctcpools_vr[15,nl,0] = ctcpools_vr[15,nl,0]*3.0


          for v in range(0,self.nobs):
            sum_plant_ndemand = 0.0
            for p in range(0,npfts):
              # --------------------1.  Phenology -------------------------
              #Calculate leaf on
              if (parms['season_decid'][p] == 1):     #Decidous phenology
                gdd_last = gdd[p]
                dayl_last = dayl[v-1]
                gdd_base = 0.0
                gdd[p] = (doy[v] > 1) * (gdd[p] + max(0.5*(tmax[v]+tmin[v])-gdd_base, 0.0))
                if (gdd[p] >= parms['gdd_crit'][p] and gdd_last < parms['gdd_crit'][p]):
                  leafon[p] = parms['ndays_on'][p]
                  leafc_trans_tot[p]  = leafc_stor[p,v]*parms['fstor2tran'][p]
                  frootc_trans_tot[p] = frootc_stor[p,v]*parms['fstor2tran'][p]
                if (leafon[p] > 0):
                  leafc_trans[p]  = leafc_trans_tot[p]  / parms['ndays_on'][p]
                  frootc_trans[p] = frootc_trans_tot[p] / parms['ndays_on'][p]
                  leafon[p] = leafon[p] - 1
                else:
                  leafc_trans[p] = 0.0
                  frootc_trans[p] = 0.0
                #Calculate leaf off
                if (dayl_last >= parms['crit_dayl'][p]/3600. and dayl[v] < parms['crit_dayl'][p]/3600.):
                   leafoff[p] = parms['ndays_off'][p]
                   leafc_litter_tot[p]  = leafc[p,v]
                   frootc_litter_tot[p] = frootc[p,v]
                if (leafoff[p] > 0):
                   leafc_litter[p]  = min(leafc_litter_tot[p]  / parms['ndays_off'][p], leafc[p,v])
                   frootc_litter[p] = min(frootc_litter_tot[p] / parms['ndays_off'][p], frootc[p,v])
                   leafoff[p] = leafoff[p] - 1
                else:
                   leafc_litter[p]  = 0.0
                   frootc_litter[p] = 0.0
                leafn_litter[p] = leafc_litter[p] /parms['leaflitcn'][p]
                retransn[p] = leafc_litter[p] / parms['leafcn'][p] - leafn_litter[p]
              else:               #Evergreen phenology / leaf mortality`
                retransn[p] = leafc[p,v]  * 1.0 / (parms['leaf_long'][p]*365. ) * \
                                    (1.0 / parms['leafcn'][p] - 1.0 / parms['leaflitcn'][p])
                leafc_litter[p]  = parms['r_mort'] * leafc[p,v]/365.0  + leafc[p,v]  * 1.0 / (parms['leaf_long'][p]*365. )
                leafn_litter[p]  = parms['r_mort'] * leafc[p,v]/365.0  / parms['leafcn'][p] +  \
                               leafc[p,v]  * 1.0 / (parms['leaf_long'][p]*365. ) / parms['leaflitcn'][p]
                frootc_litter[p] = parms['r_mort'] * frootc[p,v]/365.0 + frootc[p,v] * 1.0 / (parms['froot_long'][p]*365.)

              #Calculate live wood turnover
              livestemc_turnover[p]  = parms['lwtop_ann'] / 365. * livestemc[p,v]
              livecrootc_turnover[p] = parms['lwtop_ann'] / 365. * livecrootc[p,v]
              retransn[p] = retransn[p] + (livestemc_turnover[p]+livecrootc_turnover[p]) * \
                                  (1.0/parms['livewdcn'][p]-1.0/parms['deadwdcn'][p])
              slatop = parms['slatop'][p]
              lai[p,v+1] = leafc[p,v] * slatop

              #---------------------2. GPP -------------------------------------
              #Calculate GPP flux using the ACM model (Williams et al., 1997)

              if (lai[p,v] > 1e-3):
                if (use_nn == False):
                  #Use the ACM model from DALEC
                  rtot = 1.0
                  psid = -2.0
                  myleafn = 1.0/(parms['leafcn'][p] * slatop)
                  gs = abs(psid)**a[p,9]/((a[p,5]*rtot+(tmax[v]-tmin[v])))
                  pp = max(lai[p,v],0.5)*myleafn/gs*a[p,0]*numpy.exp(a[p,7]*tmax[v])
                  qq = a[p,2]-a[p,3]
                  #internal co2 concentration
                  ci = 0.5*(cair[v]+qq-pp+((cair[v]+qq-pp)**2-4.*(cair[v]*qq-pp*a[p,2]))**0.5)
                  e0 = a[p,6]*max(lai[p,v],0.5)**2/(max(lai[p,v],0.5)**2+a[p,8])
                  cps   = e0*rad[v]*gs*(cair[v]-ci)/(e0*rad[v]+gs*(cair[v]-ci))
                  gpp[p,v+1] = cps*(a[p,1]*dayl[v]+a[p,4])
                  #ACM is not valid for LAI < 0.5, so reduce GPP linearly for low LAI
                  if (lai[p,v] < 0.5):
                    gpp[p,v+1] = gpp[p,v+1]*lai[p,v]/0.5
                  gpp[p,v+1] = gpp[p,v+1]*btran[v]
                else:
                   #Use the Neural network trained with ELM data
                   dayl_factor = (dayl[v]/max(dayl[0:365]))**2.0
                   flnr = parms['flnr'][p]
                   if (v < 10):
                     t10 = (tmax[v]+tmin[v])/2.0+273.15
                   else:
                     t10 = sum(tmax[v-10:v]+tmin[v-10:v])/20.0+273.15
                   #Use the NN trained on daily data
                   met_thistimestep=[btran[v], lai[p,v], lai[p,v]/4.0, tmax[v]+273.15, tmin[v]+273.15, t10, \
                                   rad[v]*1e6, 50.0, cair[v]/10.0, dayl_factor, flnr, slatop, parms['leafcn'][p], parms['mbbopt'][p]]
                   for i in range(0,self.nparms_nn):   #normalize
                     met_thistimestep_norm[0,i] = ( met_thistimestep[i] - self.pmin_nn[i] ) / \
                         (self.pmax_nn[i] - self.pmin_nn[i])
                   gpp[p,v+1] = max(self.nnmodel.predict(met_thistimestep_norm), 0.0)
              else:
                  gpp[p,v+1] = 0.0

              #--------------------3.  Maintenace respiration ------------------------
              #Maintenance respiration
              trate = parms['q10_mr'][p]**((0.5*(tmax[v]+tmin[v])-25.0)/25.0)
              mr[p,v+1] = (leafc[p,v]/parms['leafcn'][p] + frootc[p,v]/parms['frootcn'][p] + \
                       (livecrootc[p,v]+livestemc[p,v])/parms['livewdcn'][p])* \
                       (parms['br_mr'][p]*24*3600)*trate
              #Nutrient limitation
              availc[p]      = max(gpp[p,v+1]-mr[p,v+1],0.0)
              xsmr[p] = max(mr[p,v+1]-gpp[p,v+1],0.0)

              #---------------4.  Allocation and growth respiration -------------------
              frg  = parms['rg_frac'][p]
              flw  = parms['f_livewd'][p]
              f1   = parms['froot_leaf'][p]
         
              if (parms['stem_leaf'][p] < 0):
                f2   = max(-1.0*parms['stem_leaf'][p]/(1.0+numpy.exp(-0.004*(annsum_npp[p] - \
                               300.0))) - 0.4, 0.1)
                f3   = parms['croot_stem'][p]
              else:
                f2 = parms['stem_leaf'][p]
                f3 = parms['croot_stem'][p]
              callom[p] = (1.0+frg)*(1.0 + f1 + f2*(1+f3))
              nallom[p] = 1.0 / parms['leafcn'][p] + f1 / parms['frootcn'][p] + \
                    f2 * flw * (1.0 + f3) / parms['livewdcn'][p] + \
                    f2 * (1.0 - flw) * (1.0 + f3) / parms['deadwdcn'][p]
              if (parms['season_decid'][p] == 1):
                leafc_alloc[p]      = 0.
                frootc_alloc[p]     = 0.
                leafcstor_alloc[p]  = availc[p] * 1.0/callom[p]
                frootcstor_alloc[p] = availc[p] * f1/callom[p]
              else:
                leafcstor_alloc[p]  = 0.
                frootcstor_alloc[p] = 0.
                leafc_alloc[p]      = availc[p] * 1.0/callom[p]
                frootc_alloc[p]     = availc[p] * f1/callom[p]
              livestemc_alloc[p]  = availc[p] * flw*f2/callom[p]
              deadstemc_alloc[p]  = availc[p] * (1.0-flw) * f2/callom[p]
              livecrootc_alloc[p] = availc[p] * flw*(f2*f3)/callom[p]
              deadcrootc_alloc[p] = availc[p] * (1.0-flw) * f2*f3/callom[p]
              #Calculate nitrogen demand from smminn, subtracting off retranslocated proportion
              plant_ndemand[p] = availc[p] * nallom[p]/callom[p] - annsum_retransn[p]*gpp[p,v+1]/annsum_gpp[p]
              sum_plant_ndemand = sum_plant_ndemand + pftwt[p] * plant_ndemand[p]

              if (calc_nlimitation):
                rc = 3.0 * max(annsum_npp[p] * nallom[p]/callom[p], 0.01)
                r = max(1.0, rc/max(nstor[p,v],1e-15))
                plant_nalloc[p] = (plant_ndemand[p] + annsum_retransn[p]*gpp[p,v+1]/annsum_gpp[p]) / r
                fpg[p,v] = 1/r    #Growth limiation due to npool resistance
                cstor_alloc[p] = availc[p] * (1.0 - fpg[p,v])
              else:
                fpg[p,v] = parms['fpg'][p]
                cstor_alloc[p] = availc[p] * (1.0 - parms['fpg'][p])
              gr[p,v+1] = availc[p] * fpg[p,v] * frg * (1.0 + f1+f2*(1+f3))/callom[p]

            #Calculate resistance term and actual uptake f_om npool
            ctc_cn = numpy.zeros([8,self.nsoil_layers], numpy.float)+10.0   #default SOM pools to 10
            if (calc_nlimitation):
              #Calculate potential immobilization (assume from litter-> SOM transitions only)
              potential_immob_vr = numpy.zeros([self.nsoil_layers], numpy.float)
              trate = parms['q10_hr']**((0.5*(tmax[v]+tmin[v])-10)/10.0)
              for p in range(0,3):
                for nl in range(0,self.nsoil_layers):
                  if (ctcpools_vr[p,nl,v] > 0 and ctcpools_vr[p+8,nl,v] > 0):
                    #Calculate CN ratios for litter pools (SOM pools are constant)
                    ctc_cn[p,nl] = ctcpools_vr[p,nl,v] / ctcpools_vr[p+8,nl,v]   
                  #Immobilization depends on transitions from litter (p) to SOM (p+3)
                  potential_immob_vr[nl] = potential_immob_vr[nl] + max((1.0-rf_ctc[p])*k_ctc[p]*trate * \
                     depth_scalar[nl]*ctcpools_vr[p,nl,v]*(1.0/ctc_cn[p+3,nl] - 1.0/ctc_cn[p,nl]), 0.0)
              #Calculate CWD CN ratio (pool 7) to be used later
              for nl in range(0,self.nsoil_layers):
                if (ctcpools_vr[7,nl,v] > 0 and ctcpools_vr[15,nl,v] > 0):
                  ctc_cn[7,nl] = ctcpools_vr[7,nl,v]/ctcpools_vr[15,nl,v]

              #calculate fpi for each layer (microbial and plant competition)
              plant_ndemand_vr = numpy.zeros([self.nsoil_layers])
              fpi = 0.0
              for nl in range(0,self.nsoil_layers):
                #plant_ndemand_vr[nl] = plant_ndemand * root_frac[nl]
                #Vertical distribution of plant N demand scales with available mineral N
                if (sum(sminn_vr[:,v]) > 0):
                  plant_ndemand_vr[nl] = sum_plant_ndemand * sminn_vr[nl,v] / sum(sminn_vr[:,v])
                else:
                  if (s == 0):
                    plant_ndemand_vr[nl] = sum_plant_ndemand
                if (plant_ndemand_vr[nl] + potential_immob_vr[nl] >= sminn_vr[nl,v] and \
                   (plant_ndemand_vr[nl] + potential_immob_vr[nl]) > 0):
                  fpi_vr[nl,v] = sminn_vr[nl,v] / (plant_ndemand_vr[nl] + potential_immob_vr[nl])               
                  if (sum_plant_ndemand > 0):
                    fpi = fpi + fpi_vr[nl,v]*plant_ndemand_vr[nl]/sum_plant_ndemand
                  else:
                    fpi = 1.0
                else:
                  fpi_vr[nl,v] = 1.0
                  if (sum_plant_ndemand > 0):
                    fpi = fpi + 1.0*plant_ndemand_vr[nl]/sum_plant_ndemand
                  else:
                    fpi = 1.0
            #print v, fpi, plant_ndemand[0], nstor[0,v], retransn[p], plant_nalloc[p]
            #time.sleep(0.05)
            if (s < spinup_cycles):
              mort_factor = 10.0
            else:
              mort_factor = 1.0

            #Mortality fluxes
            leafc_litter_vr      = numpy.zeros([self.nsoil_layers],numpy.float)
            frootc_litter_vr     = numpy.zeros([self.nsoil_layers],numpy.float)
            leafn_litter_vr      = numpy.zeros([self.nsoil_layers],numpy.float)
            livestemc_litter_vr  = numpy.zeros([self.nsoil_layers],numpy.float)
            livecrootc_litter_vr = numpy.zeros([self.nsoil_layers],numpy.float)
            deadstemc_litter_vr  = numpy.zeros([self.nsoil_layers],numpy.float)
            deadcrootc_litter_vr = numpy.zeros([self.nsoil_layers],numpy.float)
            cstor_litter_vr      = numpy.zeros([self.nsoil_layers],numpy.float)
            nstor_litter_vr      = numpy.zeros([self.nsoil_layers],numpy.float)
            for nl in range(0,self.nsoil_layers):
              for p in range(0,npfts):
                leafc_litter_vr[nl]  = leafc_litter_vr[nl] + pftwt[p] * leafc_litter[p] * surf_prof[nl]
                frootc_litter_vr[nl] = frootc_litter_vr[nl] + pftwt[p] * frootc_litter[p] * surf_prof[nl]
                leafn_litter_vr[nl]  = leafn_litter_vr[nl] + pftwt[p] * leafn_litter[p] * surf_prof[nl]
                livestemc_litter_vr[nl] = livestemc_litter_vr[nl] + pftwt[p] * parms['r_mort'] / 365.0 * livestemc[p,v] * surf_prof[nl]
                livecrootc_litter_vr[nl] = livecrootc_litter_vr[nl] + pftwt[p] * parms['r_mort'] / 365.0 * livecrootc[p,v] * root_frac[p,nl]
                deadstemc_litter_vr[nl] = deadstemc_litter_vr[nl] + pftwt[p] * parms['r_mort'] * mort_factor / 365.0 * deadstemc[p,v] * surf_prof[nl]
                deadcrootc_litter_vr[nl] = deadcrootc_litter_vr[nl] + pftwt[p] * parms['r_mort'] * mort_factor / 365.0 * deadcrootc[p,v] * root_frac[p,nl]
                cstor_litter_vr[nl] = cstor_litter_vr[nl] + pftwt[p] * parms['r_mort'] / 365.0 * cstor[p,v] * surf_prof[nl]
                nstor_litter_vr[nl] = nstor_litter_vr[nl] + pftwt[p] * parms['r_mort'] / 365.0 * nstor[p,v] * surf_prof[nl]
            # Take XSMR From cpool instead (below)
            for p in range(0,npfts):
              cstor_turnover[p] = 1.0 / (parms['cstor_tau'][p] * 365) * cstor[p,v] * trate

              #increment plant C pools
              leafc[p,v+1]       = leafc[p,v]       + fpg[p,v]*leafc_alloc[p] + leafc_trans[p] - leafc_litter[p]
              leafc_stor[p,v+1]  = leafc_stor[p,v]  + fpg[p,v]*leafcstor_alloc[p] - leafc_trans[p]
              frootc[p,v+1]      = frootc[p,v]      + fpg[p,v]*frootc_alloc[p] + frootc_trans[p] - frootc_litter[p]
              frootc_stor[p,v+1] = frootc_stor[p,v] + fpg[p,v]*frootcstor_alloc[p] - frootc_trans[p]
              livestemc[p,v+1]   = livestemc[p,v]   + fpg[p,v]*livestemc_alloc[p] - parms['r_mort'] / 365.0 * livestemc[p,v] \
                                                - livestemc_turnover[p]
              deadstemc[p,v+1]   = deadstemc[p,v]   + fpg[p,v]*deadstemc_alloc[p] - parms['r_mort'] * mort_factor / 365.0 * deadstemc[p,v] \
                                                + livestemc_turnover[p]
              livecrootc[p,v+1]  = livecrootc[p,v]  + fpg[p,v]*livecrootc_alloc[p] - parms['r_mort'] / 365.0 * livecrootc[p,v] \
                                                - livecrootc_turnover[p]
              deadcrootc[p,v+1]  = deadcrootc[p,v]  + fpg[p,v]*deadcrootc_alloc[p] - parms['r_mort'] * mort_factor / 365.0 * deadcrootc[p,v] \
                                                + livecrootc_turnover[p]
              cstor[p,v+1]       = cstor[p,v] + cstor_alloc[p] - parms['r_mort'] / 365.0 * cstor[p,v] - cstor_turnover[p] - xsmr[p]
              #Increment plant N pools
              if (calc_nlimitation):
                nstor[p,v+1] = nstor[p,v] - parms['r_mort'] / 365.0 * nstor[p,v] + retransn[p] - plant_nalloc[p] + fpi*plant_ndemand[p]  

              #Calculate NPP
              npp[p,v+1] = gpp[p,v+1] - mr[p,v+1] - gr[p,v+1] - cstor_turnover[p]
              if (doy[v] == 1):
                annsum_npp[p] = annsum_npp_temp[p]
                annsum_npp_temp[p] = 0
                annsum_retransn[p] = annsum_retransn_temp[p]
                annsum_retransn_temp[p] = 0
                annsum_gpp[p] = annsum_gpp_temp[p]
                annsum_gpp_temp[p] = 0
                 
              annsum_npp_temp[p] = annsum_npp_temp[p]+npp[p,v]
              annsum_retransn_temp[p] = annsum_retransn_temp[p]+retransn[p]
              annsum_gpp_temp[p] = annsum_gpp_temp[p]+gpp[p,v]

            # ----------------- Litter and SOM decomposition model (CTC) --------------------
            ctc_input    = numpy.zeros([16,self.nsoil_layers],numpy.float)  #inputs to pool
            ctc_output   = numpy.zeros([16,self.nsoil_layers],numpy.float)  #Outputs from pool
            ctc_resp     = numpy.zeros([8,self.nsoil_layers],numpy.float)  #Respiration from pool
            #Litter inputs to the system
              #Carbon
            for nl in range(0,self.nsoil_layers):
              ctc_input[0,nl] = leafc_litter_vr[nl]*parms['lf_flab'] + frootc_litter_vr[nl]*parms['fr_flab']
              ctc_input[1,nl] = leafc_litter_vr[nl]*parms['lf_flig'] + frootc_litter_vr[nl]*parms['fr_flig']
              ctc_input[2,nl] = leafc_litter_vr[nl]*(1.0 - parms['lf_flab'] - parms['lf_flig']) + frootc_litter_vr[nl]* \
                           (1.0-parms['fr_flab']-parms['fr_flig'])
              ctc_input[7,nl] = livestemc_litter_vr[nl] + livecrootc_litter_vr[nl] + deadcrootc_litter_vr[nl] + deadstemc_litter_vr[nl] 
              #Nitrogen
              ctc_input[8,nl] = leafn_litter_vr[nl]*parms['lf_flab'] + \
                               frootc_litter_vr[nl]*parms['fr_flab'] / parms['frootcn'][p] 
              ctc_input[9,nl] = leafn_litter_vr[nl]*parms['lf_flig'] + \
                               frootc_litter_vr[nl]*parms['fr_flig'] / parms['frootcn'][p]
              ctc_input[10,nl] = leafn_litter_vr[nl]*(1.0 - parms['lf_flig'] - parms['lf_flab']) +  \
                           frootc_litter_vr[nl]*(1.0 - parms['fr_flig'] - parms['fr_flab']) / parms['frootcn'][p]
              ctc_input[15,nl] = (livestemc_litter_vr[nl] + livecrootc_litter_vr[nl]) / parms['livewdcn'][p] + \
                            (deadcrootc_litter_vr[nl] + deadstemc_litter_vr[nl]) / parms['deadwdcn'][p]

            ctc_to_sminn = numpy.zeros([self.nsoil_layers], numpy.float)
            if (s < spinup_cycles):
              spinup_factors = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 30.0, 3.0]
            else:
              spinup_factors = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            for nl in range(0,self.nsoil_layers):
              trate = parms['q10_hr']**((0.5*(tmax[v]+tmin[v])-10)/10.0)
              for p1 in range(0,8):
                if (p1 < 3):
                   if (calc_nlimitation):
                     ctc_output[p1,nl]   = k_ctc[p1]*ctcpools_vr[p1,nl,v]*depth_scalar[nl]*trate*fpi_vr[nl,v] 
                     ctc_output[p1+8,nl] = k_ctc[p1]*ctcpools_vr[p1,nl,v]*depth_scalar[nl]*trate*fpi_vr[nl,v] / ctc_cn[p1,nl]
                   else:
                     ctc_output[p1,nl]   = k_ctc[p1]*ctcpools_vr[p1,nl,v]*depth_scalar[nl]*trate*parms['fpi']
                     ctc_output[p1+8,nl] = k_ctc[p1]*ctcpools_vr[p1,nl,v]*depth_scalar[nl]*trate*parms['fpi'] / ctc_cn[p1,nl]
                else:
                   ctc_output[p1,nl]   = k_ctc[p1]*ctcpools_vr[p1,nl,v]*spinup_factors[p1]*depth_scalar[nl]*trate  #Decomposition (output)
                   ctc_output[p1+8,nl] = k_ctc[p1]*ctcpools_vr[p1,nl,v]*spinup_factors[p1]*depth_scalar[nl]*trate / ctc_cn[p1,nl]

                #Calculate HR and N mineralization from HR
                ctc_resp[p1,nl]   = ctc_output[p1,nl]*rf_ctc[p1]       
                if (ctcpools_vr[p1,nl,v] > 0):
                  ctc_to_sminn[nl] = ctc_to_sminn[nl] + ctc_resp[p1,nl] / ctc_cn[p1,nl] 
                for p2 in range(0,8):
                    #Transfer carbon from one pool to another
                    ctc_input[p2,nl] = ctc_input[p2,nl] + ctc_output[p1,nl]*tr_ctc[p1,p2]
                    if (p1 < 7):
                      ctc_input[p2+8,nl] = ctc_input[p2+8,nl] + ctc_output[p1,nl]*tr_ctc[p1,p2] / ctc_cn[p2,nl]
                      #Calculate nitrogen mineralized/immobilized
                      ctc_to_sminn[nl] = ctc_to_sminn[nl] + ctc_output[p1,nl]*tr_ctc[p1,p2] * \
                                        (1/ctc_cn[p1,nl] - 1/ctc_cn[p2,nl])
                    else:
                      #Fragmentation from CWD into litter - no immobilization/mineralization
                      ctc_input[p2+8,nl] = ctc_input[p2+8,nl] + ctc_output[p1,nl]*tr_ctc[p1,p2] / ctc_cn[p1,nl]
            hr[v+1]=0

            #Calculate inputs and outputs from advection
            advection_rate = 0.000  #m/yr
            if (self.nsoil_layers > 1 and advection_rate > 0.0):
              for p in range(0,16):
                for nl in range(0,self.nsoil_layers):
                  if (nl < self.nsoil_layers-1):
                    ctc_output[p,nl] = ctc_output[p,nl] + ctcpools_vr[p,nl,v] * spinup_factors[p % 8] * \
                                       advection_rate / 365.0 / soil_depth[nl]
                  if (nl > 0):
                    ctc_input[p,nl] = ctc_input[p,nl] + ctcpools_vr[p,nl-1,v] * spinup_factors[p % 8] * \
                                       advection_rate / 365.0 / soil_depth[nl-1]

            #Increment ctcpools
            for p in range(0,16):        #Handle both C and N
              for nl in range(0,self.nsoil_layers):
                ctcpools_vr[p,nl,v+1] = ctcpools_vr[p,nl,v] + ctc_input[p,nl] - ctc_output[p,nl]
                if (p < 8):
                  hr[v+1] = hr[v+1] + ctc_resp[p,nl]
 
            #Calculate NEE
            #nee[v+1]    = hr[v+1] - npp[v+1]
            #Total system carbon
            totecosysc_tmp = 0.0
            for p in range(0,npfts):
              totecosysc[v+1] = totecosysc_tmp + pftwt[p] * (leafc[p,v+1]+leafc_stor[p,v+1]+frootc[p,v+1]+ \
                                frootc_stor[p,v+1]+livestemc[p,v+1]+deadstemc[p,v+1]+livecrootc[p,v+1]+ \
                                cstor[p,v+1]+deadcrootc[p,v+1])
            for nl in range(0,self.nsoil_layers):
              totecosysc[v+1] = totecosysc[v+1] + sum(ctcpools_vr[:,nl,v+1])
            totlitc[v+1] = sum(ctcpools_vr[0,:,v+1])+sum(ctcpools_vr[1,:,v+1])+sum(ctcpools_vr[2,:,v+1])
            totsomc[v+1] = sum(ctcpools_vr[3,:,v+1])+sum(ctcpools_vr[4,:,v+1])+ \
                           sum(ctcpools_vr[5,:,v+1])+sum(ctcpools_vr[6,:,v+1])
            totlitn[v+1] = sum(ctcpools_vr[8,:,v+1])+sum(ctcpools_vr[9,:,v+1])+ \
                           sum(ctcpools_vr[10,:,v+1])+sum(ctcpools_vr[11,:,v+1])
            cwdc[v+1]    = sum(ctcpools_vr[7,:,v+1])
            nee[v+1] = totecosysc[v+1]-totecosysc[v]
 
            #Update soil mineral nitrogen
            ndep[v] = 0.115 / (365)
            nfix[v] = 0
            for p in range(0,self.npfts):
              nfix[v] = nfix[v] + pftwt[p] * (1.8 * (1.0 - math.exp(-0.003 * annsum_npp[p]))) / (365)            
            bdnr = 0.005
            if (calc_nlimitation):
              for nl in range(0,self.nsoil_layers):
                sminn_vr[nl,v+1] = max(sminn_vr[nl,v]*(1-bdnr) + nfix[v]*root_frac[0,nl] + ndep[v]*surf_prof[nl] - \
                                     fpi_vr[nl,v]*plant_ndemand_vr[nl] + ctc_to_sminn[nl], 0.0)

    def run_selm(self, spinup_cycles=0, lat_bounds=[-999,-999], lon_bounds=[-999,-999], \
                     do_monthly_output=False, do_output_forcings=False, pft=-1,          \
                     prefix='model', use_nn=False, ensemble=False, myoutvars=[], use_MPI=False):

        ens_torun=[]
        indx_torun=[]
        indy_torun=[]
        pftwt_torun = numpy.zeros([self.npfts, 10000000], numpy.float)
        n_active=0
        if (self.site == 'none'):
         if (use_MPI):
           from mpi4py import MPI
           comm=MPI.COMM_WORLD
           rank=comm.Get_rank()
           size=comm.Get_size()
           print size, 'i am', rank
         else:
           rank = 0
           size = 0
         if (rank == 0):
          mydomain = Dataset(oscm_dir+"/models/pftdata/domain.360x720_ORCHIDEE0to360.100409.nc4",'r')
          landmask = mydomain.variables['mask']
          myinput = Dataset(oscm_dir+"/models/pftdata/surfdata_360x720_DALEC.nc4")
          pct_pft    = myinput.variables['PCT_NAT_PFT']
          pct_natveg = myinput.variables['PCT_NATVEG']
          self.hdlatgrid = myinput.variables['LATIXY']
          self.hdlongrid = myinput.variables['LONGXY']
          self.x1 = int(round((lon_bounds[0]-0.25)*2))
          if (self.x1 < 0):
             self.x1 = self.x1+720
          self.x2 = int(round((lon_bounds[1]-0.25)*2))
          if (self.x2 < 0):
             self.x2 = self.x2+720
          self.nx = self.x2-self.x1+1
          self.y1 = int(round((lat_bounds[0]+89.75)*2))
          self.y2 = int(round((lat_bounds[1]+89.75)*2))
          self.ny = self.y2-self.y1+1
          lats_torun=[]
          lons_torun=[]
          vegfrac_torun=[]
          if (self.ne > 1 and size > 1 and size < self.nx*self.ny):
            all_ensembles_onejob = True
            k_max = 1
          else:
            all_ensembles_onejob = False
            k_max = self.ne
          for i in range(0,self.nx):
              for j in range(0,self.ny):
                vegfrac    = pct_natveg[self.y1+j,self.x1+i]
                bareground = pct_pft[0,self.y1+j,self.x1+i]
                if (bareground < 95.0 and vegfrac > 0.1 and landmask[self.y1+j,self.x1+i] > 0):
                  for k in range(0,k_max):
                    lons_torun.append(self.hdlongrid[self.y1+j,self.x1+i])
                    lats_torun.append(self.hdlatgrid[self.y1+j,self.x1+i])
                    if (pft < 0):
                      pftwt_torun[0, n_active] = sum(pct_pft[6:9,self.y1+j,self.x1+i])+pct_pft[3,self.y1+j,self.x1+i]
                      pftwt_torun[1, n_active] = sum(pct_pft[1:3,self.y1+j,self.x1+i])+pct_pft[4,self.y1+j,self.x1+i]+sum(pct_pft[9:12,self.y1+j,self.x1+i])
                      pftwt_torun[2, n_active] = sum(pct_pft[12:,self.y1+j,self.x1+i])
                    else:
                      pftwt_torun[pft, n_active] = 100.0
                    indx_torun.append(i)
                    indy_torun.append(j)
                    ens_torun.append(k)
                    vegfrac_torun.append((100.0-bareground)/100.0)
                    n_active = n_active+1

          #Load all forcing data into memory
          self.get_regional_forcings()
          #get forcings for one point to get relevant info
          self.load_forcings(lon=lons_torun[0], lat=lats_torun[0])
        else:
          #site forcing has already been loaded
          all_ensembles_onejob = False
          rank = 0
          size = 0
          n_active = self.ne
          if (n_active > 1 and use_MPI):
             from mpi4py import MPI
             comm=MPI.COMM_WORLD
             rank=comm.Get_rank()
             size=comm.Get_size()
          if (rank == 0):
            for k in range(0,self.ne):
               if (pft >= 0):
                 pftwt_torun[pft, k] = 100.0
               elif ('SPR'in self.site):
                 pftwt_torun[0:4, k] = 25.0
               indx_torun.append(0)
               indy_torun.append(0)
               ens_torun.append(k)
            self.nx = 1
            self.ny = 1

        if (rank == 0):
          print('%d simulation units to run'%(n_active))
          n_done=0
          if (do_monthly_output):
             self.nt = (self.end_year-self.start_year+1)*12
             istart=0
          else:
             self.nt = int(self.end_year-self.start_year+1)*365
             istart=1

          model_output={}
          if (len(myoutvars) == 0):
            myoutvars = self.outvars
            for v in self.forcvars:
              if (v != 'time'):
                myoutvars.append[v]

          for v in myoutvars:
            if (v in self.forcvars):
              do_output_forcings = True
              model_output[v] = numpy.zeros([self.nt,self.ny,self.nx], numpy.float)
            elif (v != 'ctcpools_vr' and (not '_vr' in v) and (not '_pft' in v) ):
                model_output[v] = numpy.zeros([self.ne,self.nt,self.ny,self.nx], numpy.float)
            elif ('_vr' in v):
                model_output[v] = numpy.zeros([self.ne,self.nsoil_layers,self.nt,self.ny,self.nx], numpy.float)              
            elif ('_pft' in v):
                model_output[v] = numpy.zeros([self.ne,self.npfts,self.nt,self.ny,self.nx], numpy.float)

          self.pftfrac = numpy.zeros([self.ny,self.nx,self.npfts], numpy.float)

          if (self.site == 'none'):
            self.load_forcings(lon=lons_torun[0], lat=lats_torun[0])

          if ((n_active == 1 and self.ne == 1) or size == 0):
            #No MPI
            for i in range(0,n_active):
                if (self.site == 'none'):
                  self.load_forcings(lon=lons_torun[i], lat=lats_torun[i])
                if (self.ne > 1):
                  for p in range(0,len(self.ensemble_pnames)):
                    self.parms[self.ensemble_pnames[p]][self.ensemble_ppfts[p]] = self.parm_ensemble[i,p]
                print 'Starting SLEM instance'
                self.selm_instance(self.parms, use_nn=use_nn, spinup_cycles=spinup_cycles, pftwt=pftwt_torun[:,i]/100.0)
                self.pftfrac[indy_torun[i],indx_torun[i],:] = pftwt_torun[:,i]
                for v in myoutvars:
                  if (v in self.outvars):
                    if (do_monthly_output):
                      if (v != 'ctcpools_vr' and (not '_vr' in v) and (not '_pft' in v) ):
                        model_output[v][ens_torun[i],:,indy_torun[i],indx_torun[i]] = \
                         utils.daily_to_monthly(self.output[v][1:])
                      elif ('_vr' in v):
                          model_output[v][ens_torun[i],:,:,indy_torun[i],indx_torun[i]] = \
                           utils.daily_to_monthly(self.output[v][:,1:])
                      elif ('_pft' in v):
                          model_output[v][ens_torun[i],:,:,indy_torun[i],indx_torun[i]] = \
                           utils.daily_to_monthly(self.output[v][:,1:])
                    else:
                      if (v != 'ctcpools_vr' and (not '_vr' in v) and (not '_pft' in v) ):
                        model_output[v][ens_torun[i],:,indy_torun[i],indx_torun[i]] = \
                         self.output[v][1:]
                      elif ('_vr' in v):
                        for nl in range(0,self.nsoil_layers):
                          model_output[v][ens_torun[i],nl,:,indy_torun[i],indx_torun[i]] = \
                           self.output[v][nl,1:]
                      elif ('_pft' in v):
                        for p in range(0,self.npfts):
                          model_output[v][ens_torun[i],p,:,indy_torun[i],indx_torun[i]] = \
                           self.output[v][p,1:]
                  elif (v in self.forcvars):
                    if (do_monthly_output):
                       model_output[v][ens_torun,:,indy_torun[i],indx_torun[i]] = \
                            utils.daily_to_monthly(self.forcings[v])
                    else:
                       model_output[v][ens_torun,:,indy_torun[i],indx_torun[i]] = \
                            self.forcings[v][:]
            self.write_nc_output(model_output, do_monthly_output=do_monthly_output, prefix=prefix)
          else:
           #send first np-1 jobs where np is number of processes
           for n_job in range(1,size):
            comm.send(n_job, dest=n_job, tag=1)
            comm.send(0,     dest=n_job, tag=2)
            if (self.site == 'none'):
              self.load_forcings(lon=lons_torun[n_job-1], lat=lats_torun[n_job-1])
            parms = self.parms
            if (not all_ensembles_onejob and self.ne > 1):
              for p in range(0,len(self.ensemble_pnames)):
                parms[self.ensemble_pnames[p]][self.ensemble_ppfts[p]] = self.parm_ensemble[ens_torun[n_job-1],p]

            comm.send(all_ensembles_onejob, dest=n_job, tag=300)
            comm.send(do_output_forcings,   dest=n_job, tag=400)
            comm.send(self.forcings,   dest = n_job, tag=6)
            comm.send(self.start_year, dest = n_job, tag=7)
            comm.send(self.end_year,   dest = n_job, tag=8)
            comm.send(self.nobs,       dest = n_job, tag=9)
            comm.send(self.lat,        dest = n_job, tag=10)
            comm.send(self.forcvars,   dest = n_job, tag=11)
            if (all_ensembles_onejob):
              comm.send(self.parm_ensemble,   dest=n_job, tag=100)
              comm.send(self.ensemble_pnames, dest=n_job, tag=101)
            else:
              comm.send(parms,           dest = n_job, tag=100)
            comm.send(myoutvars,         dest = n_job, tag=200)
            comm.send(pftwt_torun[:,n_job-1], dest=n_job, tag=500)

           #Assign rest of jobs on demand
           for n_job in range(size,n_active+1):
            process = comm.recv(source=MPI.ANY_SOURCE, tag=3)
            thisjob = comm.recv(source=process, tag=4)
            myoutput = comm.recv(source=process, tag=5)
            print('Received %d'%(thisjob))
            n_done = n_done+1
            comm.send(n_job, dest=process, tag=1)
            comm.send(0,     dest=process, tag=2)
            if (self.site == 'none'):
              self.load_forcings(lon=lons_torun[n_job-1], lat=lats_torun[n_job-1])
            if (not all_ensembles_onejob and self.ne > 1):
              for p in range(0,len(self.ensemble_pnames)):
                parms[self.ensemble_pnames[p]][self.ensemble_ppfts[p]] = self.parm_ensemble[ens_torun[n_job-1],p]

            comm.send(all_ensembles_onejob, dest=process, tag=300)
            comm.send(do_output_forcings,   dest=process, tag=400)
            comm.send(self.forcings,   dest = process, tag=6)
            comm.send(self.start_year, dest = process, tag=7)
            comm.send(self.end_year,   dest = process, tag=8)
            comm.send(self.nobs,       dest = process, tag=9)
            comm.send(self.lat,        dest = process, tag=10)
            comm.send(self.forcvars,   dest = process, tag=11)
            if (all_ensembles_onejob):
              comm.send(self.parm_ensemble,   dest=process, tag=100)
              comm.send(self.ensemble_pnames, dest=process, tag=101)
            else:
              comm.send(parms,           dest = process, tag=100)
            comm.send(myoutvars,         dest = process, tag=200)
            comm.send(pftwt_torun[:,n_job-1], dest=process, tag=500)
            #write output
            for v in myoutvars:
              if (all_ensembles_onejob):
                for k in range(0,self.ne):
                  model_output[v][k,pfts_torun[thisjob-1],:,indy_torun[thisjob-1],indx_torun[thisjob-1]] = \
                          myoutput[v][k,:]
              else:
                if ('_vr' in v or '_pft' in v):
                  model_output[v][ens_torun[thisjob-1],:,:,indy_torun[thisjob-1],indx_torun[thisjob-1]] \
                                    = myoutput[v][0,:,:]
                else:
                  model_output[v][ens_torun[thisjob-1],:,indy_torun[thisjob-1],indx_torun[thisjob-1]] \
                                    = myoutput[v][0,:]
            self.pftfrac[indy_torun[thisjob-1],indx_torun[thisjob-1],:] = pftwt_torun[:,thisjob-1]

           #receive remaining messages and finalize
           while (n_done < n_active):
            process = comm.recv(source=MPI.ANY_SOURCE, tag=3)
            thisjob = comm.recv(source=process, tag=4)
            myoutput = comm.recv(source=process, tag=5)
            vnum = 0
            print('Received %d'%(thisjob))
            n_done = n_done+1
            comm.send(-1, dest=process, tag=1)
            comm.send(-1, dest=process, tag=2)
            #write output
            for v in myoutvars:
              if (all_ensembles_onejob):
                for k in range(0,self.ne):
                  model_output[v][k,pfts_torun[thisjob-1],:,indy_torun[thisjob-1],indx_torun[thisjob-1]] = \
                        myoutput[v][k,:]
              else:
                if ('_vr' in v or '_pft' in v):
                  model_output[v][ens_torun[thisjob-1],:,:,indy_torun[thisjob-1],indx_torun[thisjob-1]] \
                                    = myoutput[v][0,:,:]
                else:
                  model_output[v][ens_torun[thisjob-1],:,indy_torun[thisjob-1],indx_torun[thisjob-1]] \
                                    = myoutput[v][0,:]
            self.pftfrac[indy_torun[thisjob-1],indx_torun[thisjob-1],:] = pftwt_torun[:,thisjob-1]
           self.write_nc_output(model_output, do_monthly_output=do_monthly_output, prefix=prefix)
           #MPI.Finalize()
        #Slave
        else:
          status=0
          while status == 0:
            myjob = comm.recv(source=0, tag=1)
            status = comm.recv(source=0, tag=2)
            if (status == 0):
              all_ensembles_onejob = comm.recv(source=0, tag=300)
              do_output_forcings   = comm.recv(source=0, tag=400)
              self.forcings = comm.recv(source = 0, tag = 6)
              self.start_year = comm.recv(source=0, tag=7)
              self.end_year   = comm.recv(source=0, tag=8)
              self.nobs       = comm.recv(source=0, tag=9)
              self.lat        = comm.recv(source=0, tag=10)
              self.forcvars   = comm.recv(source=0, tag=11)
              if (all_ensembles_onejob):
                self.parm_ensemble = comm.recv(source=0, tag=100)
                self.ensemble_pnames = comm.recv(source=0, tag=101)
              else:
                myparms         = comm.recv(source=0, tag=100)
              myoutvars = comm.recv(source=0, tag=200)
              mypftwt   = comm.recv(source=0, tag=500)
              #Initialize output arrays
              self.output = {}
              self.output_ens = {}
              if (all_ensembles_onejob):
                 k_max = self.ne
              else:
                 k_max = 1
              for var in self.outvars:
                if (var == 'ctcpools_vr'):
                   self.output[var] = numpy.zeros([16,self.nsoil_layers,self.nobs+1], numpy.float)
                elif ('_vr' in var):
                   self.output[var] = numpy.zeros([self.nsoil_layers,self.nobs+1], numpy.float)
                elif ('_pft' in var):
                   self.output[var] = numpy.zeros([self.npfts,self.nobs+1], numpy.float)
                else:
                   self.output[var] = numpy.zeros([self.nobs+1], numpy.float)
  
              thisoutput = {}
              thisoutput_ens = {}
              for k in range(0,k_max):
                if (all_ensembles_onejob):
                  myparms = self.pdefault
                  for p in range(0,len(self.ensemble_pnames)):
                    myparms[self.ensemble_pnames[p]] = self.parm_ensemble[k,p]
                self.selm_instance(myparms, use_nn=use_nn, spinup_cycles=spinup_cycles, pftwt=mypftwt/100.0 )
                for v in myoutvars:
                   if (v in self.outvars):
                     if (do_monthly_output):
                       if (('_vr' in v or '_pft' in v) and not 'ctcpools' in v):
                         thisoutput[v] = utils.daily_to_monthly(self.output[v][:,1:])
                       else:
                         thisoutput[v] = utils.daily_to_monthly(self.output[v][1:])
                     else:
                       if (('_vr' in v or '_pft' in v) and not 'ctcpools' in v):
                         thisoutput[v] = self.output[v][:,1:]
                       else:
                         thisoutput[v] = self.output[v][1:]
                   elif (v in self.forcvars):
                     if (do_monthly_output):
                         thisoutput[v] = utils.daily_to_monthly(self.forcings[v])
                     else:
                         thisoutput[v] = self.forcings[v]
                   if (k == 0):
                      if (('_vr' in v) and not 'ctcpools' in v):
                        thisoutput_ens[v] = numpy.zeros([k_max,self.nsoil_layers,max(thisoutput[v].shape)], numpy.float)
                      elif ('_pft' in v):
                        thisoutput_ens[v] = numpy.zeros([k_max,self.npfts,max(thisoutput[v].shape)], numpy.float)
                      else:
                        thisoutput_ens[v] = numpy.zeros([k_max, len(thisoutput[v])], numpy.float)
                   if (('_vr' in v) or ('_pft' in v)):
                     thisoutput_ens[v][k,:,:] = thisoutput[v]
                   else:
                     thisoutput_ens[v][k,:] = thisoutput[v]

              comm.send(rank,  dest=0, tag=3)
              comm.send(myjob, dest=0, tag=4)
              comm.send(thisoutput_ens, dest=0, tag=5)
          print('%d complete'%(rank))
          #MPI.Finalize()

    def write_nc_output(self, output, do_monthly_output=False, prefix='model'):
         #set up output file
         output_nc = Dataset(prefix+'_output.nc', 'w', format='NETCDF4')
         output_nc.createDimension('pft',self.npfts)
         output_nc.createDimension('soil',self.nsoil_layers)
         output_nc.createDimension('lon',self.nx)
         output_nc.createDimension('lat',self.ny)
         if (self.ne > 1):
           output_nc.createDimension('ensemble',self.ne)
           #ens_out = output_nc.createVariable('ensemble','i4',('ensemble',))
           #ens_out.axis="E"
           #ens_out.CoordinateAxisType = "Ensemble"
           pnum=0
           for p in self.ensemble_pnames:
             #write parameter values to file
             pvars={}
             pvars[p] = output_nc.createVariable(p+'_pft'+str(self.ensemble_ppfts[pnum]), 'f8', ('ensemble',))
             pvars[p][:] = self.parm_ensemble[:,pnum]
             pnum=pnum+1
         if (self.site == 'none'):
           lat_out = output_nc.createVariable('lat','f8',('lat',))
           lon_out = output_nc.createVariable('lon','f8',('lon',))
           lat_out[:] = self.hdlatgrid[self.y1:self.y2+1,self.x1]
           lon_out[:] = self.hdlongrid[self.y1,self.x1:self.x2+1]
         else:
           lat_out = self.latdeg
           lon_out = self.londeg
         pft_out = output_nc.createVariable('pft_frac','f4',('lat','lon','pft'))
         for n in range(0,self.ne):
           pft_out[:,:,:] = self.pftfrac[:,:,:]
         if (do_monthly_output):
            output_nc.createDimension('time',(self.end_year-self.start_year+1)*12)
            time = output_nc.createVariable('time','f8',('time',))
            dpm = [31,28,31,30,31,30,31,31,30,31,30,31]
            time[0] = 15.5
            mlast = 0
            for i in range(1,(self.end_year-self.start_year+1)*12):
              time[i] = time[i-1]+(dpm[mlast]+dpm[i % 12])/2.0
              mlast = i % 12
            istart=0
         else:
            output_nc.createDimension('time',(self.end_year-self.start_year+1)*365)
            time = output_nc.createVariable('time','f8',('time',))
            for i in range(0,self.nobs):
              time[i] = i+0.5
            istart=1
         time.units='Days since '+str(self.start_year)+'-01-01 00:00'

         ncvars={}
         for v in output:
            if (self.ne > 1):
              if (not 'ctcpools' in v and not '_pft' in v and not '_vr' in v):
                #ncvars[v] = output_nc.createVariable(v, 'f4',('ensemble','pft','time','lat','lon'))
                #Default - don't output PFT-level output for ensembles
                ncvars[v] = output_nc.createVariable(v, 'f4',('ensemble','time','lat','lon'))
                ncvars[v][:,:,:,:] = output[v][:,:,:,:]
              elif ('_vr' in v):
                ncvars[v] = output_nc.createVariable(v, 'f4',('ensemble','soil','time','lat','lon'))
                ncvars[v][:,:,:,:,:] = output[v][:,:,:,:,:] 
              elif ('_pft' in v):
                ncvars[v] = output_nc.createVariable(v, 'f4',('ensemble','pft','time','lat','lon'))
                ncvars[v][:,:,:,:,:] = output[v][:,:,:,:,:]
            else:
              if (not 'ctcpools' in v and not '_pft' in v and not '_vr' in v):
                ncvars[v] = output_nc.createVariable(v, 'f4',('time','lat','lon'))
                ncvars[v][:,:,:] = output[v][0,:,:,:].squeeze() 
              elif ('_vr' in v):
                ncvars[v] = output_nc.createVariable(v, 'f4',('soil','time','lat','lon'))
                ncvars[v][:,:,:,:] = output[v][0,:,:,:,:].squeeze()
              elif ('_pft' in v):
                ncvars[v] = output_nc.createVariable(v, 'f4',('pft','time','lat','lon'))
                ncvars[v][:,:,:,:] = output[v][0,:,:,:,:].squeeze()
         output_nc.close()

         #output for eden vis system - customize as needed
         if (self.ne > 1):
           eden_out = numpy.zeros([self.ne,pnum+1],numpy.float)
           for n in range(0,self.ne):
             eden_out[n,0:pnum]      = self.parm_ensemble[n,:]
             eden_out[n,pnum:pnum+1] = numpy.mean(output['gpp'][n,0,0:60,0,0])*365.
           numpy.savetxt("foreden.csv",eden_out,delimiter=",")

    def generate_synthetic_obs(self, parms, err, use_nn=False):
        #generate synthetic observations from model with Gaussian error
        self.obs = numpy.zeros([self.nobs], numpy.float)
        self.obs_err = numpy.zeros([self.nobs], numpy.float)+err
        self.selm_instance(parms, use_nn=use_nn)
        for v in range(0,self.nobs):
            self.obs[v] = self.output[v]+numpy.random.normal(0,err,1)
        self.issynthetic = True
        self.actual_parms = parms

    def get_regional_forcings(self):
        print('Loading regional forcings')
        self.regional_forc={}
        fnames = ['TMAX','TMIN','FSDS','BTRAN']
        vnames = ['TSA', 'TSA', 'FSDS','BTRAN']
        fnum=0
        for f in fnames:
          os.system('mkdir -p '+oscm_dir+'/models/elm_drivers')
          driver_path = os.path.abspath(oscm_dir+'/models/elm_drivers')
          myfile = "GSWP3_fromELMSP_"+f+"_1980-2009.nc4"
          if (not os.path.exists(driver_path+'/'+myfile)):
            print('%s not found.  Downloading.'%(myfile))
            os.system('wget --no-check-certificate https://acme-webserver.ornl.gov/dmricciuto/elm_drivers/'+myfile)
            os.rename(myfile, driver_path+'/'+myfile)
          print('%s/%s'%(driver_path,myfile))
          myinput = Dataset(driver_path+'/'+myfile,'r')
          self.regional_forc[f.lower()] = myinput.variables[vnames[fnum]][:,:,:]
          myinput.close()
          fnum = fnum+1
        print ('Loading complete')

    def load_forcings(self, site='none', lat=-999, lon=-999):
         #Get single point data from E3SM style cpl_bypass input files
         self.forcvars = ['tmax','tmin','rad','cair','doy','dayl','btran','time']
         self.forcings = {}
         self.site=site
         if (self.site != 'none'):
             #Get data for requested site
             myinput = Dataset('./forcing_data/'+self.site+'_forcing.nc4','r',format='NETCDF4')
             npts = myinput.variables['TBOT'].size              #number of half hours or hours
             tair = myinput.variables['TBOT'][0,:]              #Air temperature (K)
             fsds  = myinput.variables['FSDS'][0,:]             #Solar radiation (W/m2)
             btran = fsds * 0.0 + 1.0
             self.latdeg = myinput.variables['LATIXY'][0]            #site latitude
             self.londeg = myinput.variables['LONGXY'][0]            #site longitude
             self.start_year = int(myinput.variables['start_year'][:])    #starting year of data
             self.end_year   = int(myinput.variables['end_year'][:])   #ending year of data
             self.npd = int(npts/(self.end_year - self.start_year + 1)/365)   #number of obs per day
             self.nobs = int((self.end_year - self.start_year + 1)*365)  #number of days
             for fv in self.forcvars:
               self.forcings[fv] = []
             self.lat = self.latdeg*numpy.pi/180.
             #populate hourly forcings (for NN)
             #self.forcings['tair_hourly'] = myinput.variables['TBOT'][0,:].copy()-273.15
             #self.forcings['fsds_hourly'] = myinput.variables['FSDS'][0,:].copy()
             #self.forcings['rh_hourly']   = myinput.variables['RH'][0,:].copy()
             #self.forcings['psrf_hourly'] = myinput.variables['PSRF'][0,:].copy()
             #self.forcings['wind_hourly'] = myinput.variables['WIND'][0,:].copy()
             myinput.close()
             #populate daily forcings
             for d in range(0,self.nobs):
               self.forcings['tmax'].append(max(tair[d*self.npd:(d+1)*self.npd])-273.15)
               self.forcings['tmin'].append(min(tair[d*self.npd:(d+1)*self.npd])-273.15)
               self.forcings['rad'].append(sum(fsds[d*self.npd:(d+1)*self.npd]*(86400/self.npd)/1e6))
               self.forcings['btran'].append(1.0)
               self.forcings['cair'].append(360)
               self.forcings['doy'].append((float(d % 365)+1))
               self.forcings['time'].append(self.start_year+d/365.0)
               #Calculate day length
               dec  = -23.4*numpy.cos((360.*(self.forcings['doy'][d]+10.)/365.)*numpy.pi/180.)*numpy.pi/180.
               mult = numpy.tan(self.lat)*numpy.tan(dec)
               if (mult >= 1.):
                 self.forcings['dayl'].append(24.0)
               elif (mult <= -1.):
                  self.forcings['dayl'].append(0.)
               else:
                 self.forcings['dayl'].append(24.*numpy.arccos(-mult)/numpy.pi)

         elif (lat >= -90 and lon >= -180):
             #Get closest gridcell from reanalysis data
             self.latdeg=lat
             if (lon > 180):
                 lon=lon-360.
             if (lat > 9.5 and lat < 79.5 and lon > -170.5 and lon < -45.5):
                xg = int(round((lon + 170.25)*2))
                yg = int(round((lat - 9.75)*2))
                tmax = self.regional_forc['tmax'][:,yg,xg]
                tmin = self.regional_forc['tmin'][:,yg,xg]
                btran = self.regional_forc['btran'][:,yg,xg]
                fsds = self.regional_forc['fsds'][:,yg,xg]
             else:
                print('regions outside North America not currently supported')
                sys.exit(1)
             self.start_year = 1980
             self.end_year   = 2009
             self.npd = 1
             self.nobs = (self.end_year - self.start_year + 1)*365
             self.lat = self.latdeg*numpy.pi/180.

             #populate daily forcings
             self.forcings['tmax']  = tmax-273.15
             self.forcings['tmin']  = tmin-273.15
             self.forcings['btran'] = btran
             self.forcings['rad']   = fsds*86400/1e6
             self.forcings['cair']  = numpy.zeros([self.nobs], numpy.float) + 360.0
             self.forcings['doy']   = (numpy.cumsum(numpy.ones([self.nobs], numpy.float)) - 1) % 365 + 1
             self.forcings['time']  = self.start_year + (numpy.cumsum(numpy.ones([self.nobs], numpy.float)-1))/365.0
             self.forcings['dayl']  = numpy.zeros([self.nobs], numpy.float)
             for d in range(0,self.nobs):
               #Calculate day length
               dec  = -23.4*numpy.cos((360.*(self.forcings['doy'][d]+10.)/365.)*numpy.pi/180.)*numpy.pi/180.
               mult = numpy.tan(self.lat)*numpy.tan(dec)
               if (mult >= 1.):
                 self.forcings['dayl'][d] = 24.0
               elif (mult <= -1.):
                 self.forcings['dayl'][d] = 0.
               else:
                 self.forcings['dayl'][d] = 24.*numpy.arccos(-mult)/numpy.pi

         #define the x axis for plotting output (time, or one of the inputs)
         self.xlabel = 'Time (years)'
         #Initialize output arrays
         self.output = {}
         for var in self.outvars:
           if (var == 'ctcpools_vr'):
             self.output[var] = numpy.zeros([16,self.nsoil_layers,self.nobs+1], numpy.float)
           elif ('_vr' in var):
             self.output[var] = numpy.zeros([self.nsoil_layers,self.nobs+1], numpy.float)
           elif ('_pft' in var):
             self.output[var] = numpy.zeros([self.npfts,self.nobs+1], numpy.float)
           else:
             self.output[var] = numpy.zeros([self.nobs+1], numpy.float)

    #Load actual observations and uncertainties

    def generate_ensemble(self, n_ensemble, pnames, ppfts, fname='', normalized=False):
      self.parm_ensemble = numpy.zeros([n_ensemble,len(pnames)])
      self.ensemble_pnames = pnames
      self.ensemble_ppfts  = ppfts
      self.ne = n_ensemble
      if (fname != ''):
        #print('Generating parameter ensemble from %d'%(fname))
        inparms = open(fname,'r')
        lnum = 0
        for s in inparms:
          pvals = s.split()
          if (lnum < n_ensemble):
            for p in range(0,len(pnames)):
              if (normalized):
                self.parm_ensemble[lnum,p] = self.pmin[pnames[p]][ppfts[p]]+0.5* \
                     (float(pvals[p]))*(self.pmax[pnames[p]][ppfts[p]]-self.pmin[pnames[p]][ppfts[p]])
              else:
                self.parm_ensemble[lnum,p] = float(pvals[p])
          lnum=lnum+1
        inparms.close()
      else:
        for n in range(0,n_ensemble):
          # HACK: to have identical ensemble members uncomment line below
          # numpy.random.seed(2018)
          for p in range(0,len(pnames)):
            #Sample uniformly from the parameter space
            self.parm_ensemble[n,p] = numpy.random.uniform(low=self.pmin[pnames[p]][ppfts[p]], \
                  high=self.pmax[pnames[p]][ppfts[p]])
        #numpy.savetxt('inputs.txt', self.parm_ensemble)
