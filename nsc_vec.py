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

def GPP(rad, cair, temp, lai, dayl= 0.5, parms=elmparms,p=1):
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

def LAI(leafc, params, p=1):
    return leafc * parms['slatop'][p]

# t  = np.arange(forcing['FSDS'][0,:].size)/24/2
# plt.plot(t-0.185,forcing['FSDS'][0,:])

GPP_nn = make_GPP_nn()
x = np.linspace(0,1000,101)
for i,  c in enumerate([200,300,400, 600, 800]):
    plt.plot(x,GPP(x,c,20,30,5), 'C{}'.format(i), label=c)
    
    # plt.plot(x,GPP_nn(x,c,20,30,5), 'C{}--'.format(i))
plt.legend()
plt.grid()
# plt.plot(x, Rm(x*0.1, x*0.2, x*0.2,x*0.3,10,25))
plt.show()
    # rs['cstor_turnover'] = kwargs['br_xr'] * (3600.*24.) * cstor * kwargs['trate']
    # rs['cstor_alloc'] = kwargs['availc'] * (1. - kwargs['fpg'])
    # rs['xsmr'] = max(kwargs['mr'] -kwargs['gpp'], 0.)
    # rs['c_mortality'] = kwargs['r_mort'] / 365. * cstor