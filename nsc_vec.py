# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:11:02 2022

@author: fso
"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from sklearn.neural_network import MLPRegressor
import pickle 

elmparms = Dataset('./parameters/selm_default_parms.nc','r')
elm_parmlist = ['crit_dayl','ndays_on','ndays_off', \
                'flnr','slatop','leafcn','lflitcn','livewdcn','frootcn', \
                'deadwdcn','mbbopt','roota_par','rootb_par','fstor2tran', \
                'stem_leaf','croot_stem','flivewd','froot_leaf','grperc', \
                'br_mr','q10_mr','leaf_long','froot_long','season_decid', \
                'r_mort','lwtop_ann','q10_hr','k_l1','k_l2','k_l3','k_s1', \
                'k_s2','k_s3','k_s4','k_frag','rf_l1s1','rf_l2s2','rf_l3s3',\
                'rf_s1s2','rf_s2s3','rf_s3s4','cwd_flig','fr_flig','lf_flig', \
                'fr_flab','lf_flab','br_xr']


forcing = Dataset('./forcing_data/US-Ho1_forcing.nc4','r')

def GPP(rad, cair, tmin, tmax, lai, dayl= 12, parms=elmparms,p=1):
    a=np.array([10, 0.0156935, 4.22273, 208.868, 0.0453194,\
                0.37836, 7.19298, 0.011136, \
           2.1001, 0.789798])
    #Use the ACM model from DALEC
    rtot = 1.0
    myleafn = 1.0/(parms['leafcn'][p] * parms['slatop'][p])
    gs = 2**a[9]/((a[5]*rtot+(tmax-tmin)))
    pp = np.maximum(lai,0.5)*myleafn/gs*a[0]*np.exp(a[7]*tmax)
    qq = a[2]-a[3]
    #internal co2 concentration
    ci = 0.5*(cair+qq-pp+((cair+qq-pp)**2-4.*(cair*qq-pp*a[2]))**0.5)
    e0 = a[6]*np.maximum(lai,0.5)**2/(np.maximum(lai,0.5)**2+a[8])
    cps   = e0*rad*gs*(cair-ci)/(e0*rad+gs*(cair-ci))
    out = cps*(a[1]*dayl+a[4])
    #ACM is not valid for LAI < 0.5, so reduce GPP linearly for low LAI
    if (lai < 0.5):
        out *= lai/0.5
    return out

def make_GPP_nn():
    pkl_filename = './GPP_model_NN/bestmodel_daily.pkl'
    with open(pkl_filename, 'rb') as file:
      nnmodel = pickle.load(file)
    nsamples=20000
    nparms_nn = 14  #15
    ptrain_orig   = (np.loadtxt('./GPP_model_NN/ptrain_daily.dat'))[0:nsamples,:]
    pmin_nn = np.zeros([nparms_nn], np.float64)
    pmax_nn = np.zeros([nparms_nn], np.float64)
    for i in range(0,nparms_nn):
      pmin_nn[i] = min(ptrain_orig[:,i])
      pmax_nn[i] = max(ptrain_orig[:,i])
    
    def GPP_nn(rad, cair, tmin, tmax, lai, dayl= 0.5,dayl_factor = 1, btran =1, parms=elmparms,p=1):
        slatop = parms['slatop'][p]
        flnr = parms['flnr'][p]
        t10 = (tmax+tmin)/2.0+273.15
        
        #Use the NN trained on daily data
        args = [btran, lai, lai/4.0, tmax+273.15, tmin+273.15, t10, \
                        rad*1e6, 50.0, cair/10.0, dayl_factor, flnr, slatop, parms['leafcn'][p], parms['mbbopt'][p]]
        
        M = max(np.asarray(arg).shape for arg in args)
        met = np.empty( M +( nparms_nn,))
        for i in range(0,nparms_nn):   #normalize
          met[:, i] =(args[i] - pmin_nn[i])/(pmax_nn[i] - pmin_nn[i])
        gpp = np.maximum(nnmodel.predict(met), 0)
        return gpp
    return GPP_nn
        

def Rm(leafc, frootc, livecrootc, livestemc, temp, parms=elmparms,p=1):
    #Maintenance respiration
    trate = parms['q10_mr'][0]**((temp - 25.0)/25.0)
    leafn = leafc/parms['leafcn'][p]
    frootn = frootc/parms['frootcn'][p]
    woodn = (livecrootc+livestemc)/max(parms['livewdcn'][p],10.)
    out = (leafn + frootn  + woodn)*(parms['br_mr'][0]*24*3600)*trate
    return out

def LAI(leafc, parms, p=1):
    return leafc * parms['slatop'][p]

def ALLOC(availc, parms,p=1):
    frg  = parms['grperc'][p]
    flw  = parms['flivewd'][p]
    f1   = parms['froot_leaf'][p]
    # if (seasonal_rootalloc):
    #     if (annsum_gpp_temp[p]/annsum_gpp[p] > parms['froot_phen_peak'][p]- \
    #             parms['froot_phen_width'][p]/2.0 and annsum_gpp_temp[p]/annsum_gpp[p] \
    #             < parms['froot_phen_peak'][p]+parms['froot_phen_width'][p]/2.0):
    #       f1 = f1*1.0/(parms['froot_phen_width'][p])
    #     else:
    #       f1 = 0.0

    if (parms['stem_leaf'][p] < 0):
      f2   = max(-1.0*parms['stem_leaf'][p]/(1.0+np.exp(-0.004*(annsum_npp - \
                      300.0))) - 0.4, 0.1)
      f3   = parms['croot_stem'][p]
    else:
      f2 = parms['stem_leaf'][p]
      f3 = parms['croot_stem'][p]
    callom = (1.0+frg)*(1.0 + f1 + f2*(1+f3))
    nallom = 1.0 / parms['leafcn'][p] + f1 / parms['frootcn'][p] + \
          f2 * flw * (1.0 + f3) / max(parms['livewdcn'][p],10.) + \
          f2 * (1.0 - flw) * (1.0 + f3) / max(parms['deadwdcn'][p],10.)
    if (parms['season_decid'][p] == 1):
      leafc_alloc[p,v]      = 0.
      frootc_alloc[p,v]     = 0.
      leafcstor_alloc[p]  = availc * 1.0/callom
      frootcstor_alloc[p] = availc * f1/callom
    else:
      leafcstor_alloc[p]  = 0.
      frootcstor_alloc[p] = 0.
      leafc_alloc[p,v]      = availc * 1.0/callom
      frootc_alloc[p,v]     = availc * f1/callom
    livestemc_alloc[p,v]  = availc * flw*f2/callom
    deadstemc_alloc[p,v]  = availc * (1.0-flw) * f2/callom
    livecrootc_alloc[p] = availc * flw*(f2*f3)/callom
    deadcrootc_alloc[p] = availc * (1.0-flw) * f2*f3/callom
    
    plant_ndemand[p] = availc[p] * nallom[p]/callom[p] - annsum_retransn[p]*gpp[p,v+1]/annsum_gpp[p]


# def phenology(time,temp,gdd,dayl,parms,p=1):

#     if (parms['season_decid'][p] == 1):     #Decidous phenology
#       gdd_last = gdd[p]
#       dayl_last = dayl[v-1]
#       gdd_base = 0.0
#       doy = time * 365.
#       gdd[p] = (doy > 1) * (gdd[p] + max(temp-gdd_base, 0.0))
#       if (gdd[p] >= parms['gdd_crit'][p] and gdd_last < parms['gdd_crit'][p]):
#         leafon[p] = parms['ndays_on'][0]
#         leafc_trans_tot[p]  = leafc_stor[p,v]*parms['fstor2tran'][0]
#         frootc_trans_tot[p] = frootc_stor[p,v]*parms['fstor2tran'][0]
#       if (leafon[p] > 0):
#         leafc_trans[p]  = leafc_trans_tot[p]  / parms['ndays_on'][0]
#         frootc_trans[p] = frootc_trans_tot[p] / parms['ndays_on'][0]
#         leafon[p] = leafon[p] - 1
#       else:
#         leafc_trans[p] = 0.0
#         frootc_trans[p] = 0.0
#       #Calculate leaf off
#       if (dayl_last >= parms['crit_dayl'][0]/3600. and dayl[v] < parms['crit_dayl'][0]/3600.):
#          leafoff[p] = parms['ndays_off'][0]
#          leafc_litter_tot[p]  = leafc[p,v]
#          frootc_litter_tot[p] = frootc[p,v]
#       if (leafoff[p] > 0):
#          leafc_litter[p]  = min(leafc_litter_tot[p]  / parms['ndays_off'][0], leafc[p,v])
#          frootc_litter[p] = min(frootc_litter_tot[p] / parms['ndays_off'][0], frootc[p,v])
#          leafoff[p] = leafoff[p] - 1
#       else:
#          leafc_litter[p]  = 0.0
#          frootc_litter[p] = 0.0
#       leafn_litter[p] = leafc_litter[p] /parms['lflitcn'][p]
#       retransn[p] = leafc_litter[p] / parms['leafcn'][p] - leafn_litter[p]
#     else:               #Evergreen phenology / leaf mortality`
#       retransn[p] = leafc[p,v]  * 1.0 / (parms['leaf_long'][p]*365. ) * \
#                           (1.0 / parms['leafcn'][p] - 1.0 / parms['lflitcn'][p])
#       leafc_litter[p]  = parms['r_mort'][0] * leafc[p,v]/365.0  + leafc[p,v]  * 1.0 / (parms['leaf_long'][p]*365. )
#       leafn_litter[p]  = parms['r_mort'][0] * leafc[p,v]/365.0  / parms['leafcn'][p] +  \
#                      leafc[p,v]  * 1.0 / (parms['leaf_long'][p]*365. ) / parms['lflitcn'][p]
#       frootc_litter[p] = parms['r_mort'][0] * frootc[p,v]/365.0 + frootc[p,v] * 1.0 / (parms['froot_long'][p]*365.)    
# t  = np.arange(forcing['FSDS'][0,:].size)/24/2
# plt.plot(t-0.185,forcing['FSDS'][0,:])
x = np.linspace(0,20,101)
A = [GPP(500,400,15,25,x__,12) for x__ in x] 
plt.plot(x, A)

plt.grid()
plt.show()