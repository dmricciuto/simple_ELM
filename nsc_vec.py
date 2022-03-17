# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:11:02 2022

@author: fso
"""
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

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

def GPP(rad, cair, tmin, tmax, lai, dayl= 0.5, parms=elmparms,p=1):
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

def Rm(leafc, frootc, tmin,tmax,parms=elmparms,p=1):
    #Maintenance respiration
    trate = parms['q10_mr'][0]**((0.5*(tmax[v]+tmin[v])-25.0)/25.0)
    leafn = leafc/parms['leafcn'][p]
    frootn = frootc/parms['frootcn'][p]
         
    out = (leafn + frootn  + \
                      (livecrootc[p,v]+livestemc[p,v])/max(parms['livewdcn'][p],10.))* \
                      (parms['br_mr'][0]*24*3600)*trate
    return 
# t  = np.arange(forcing['FSDS'][0,:].size)/24/2
# plt.plot(t-0.185,forcing['FSDS'][0,:])
Ca = np.linspace(0,1000,101)
plt.plot(Ca, GPP(750, Ca, 15,25,2))
plt.plot(Ca, GPP(750, Ca, 10,20,2))
plt.plot(Ca, GPP(750, Ca, 10,20,4))
plt.show()
    # rs['cstor_turnover'] = kwargs['br_xr'] * (3600.*24.) * cstor * kwargs['trate']
    # rs['cstor_alloc'] = kwargs['availc'] * (1. - kwargs['fpg'])
    # rs['xsmr'] = max(kwargs['mr'] -kwargs['gpp'], 0.)
    # rs['c_mortality'] = kwargs['r_mort'] / 365. * cstor