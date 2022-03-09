# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:49:58 2022

@author: fso
"""


def stoich_v0(rates):
    vec = dict()
    vec['cstor'] = rates['cstor_alloc'] - rates['c_mortality']\
        - rates['cstor_turnover'] - rates['xsmr']
    return vec    

def rates_v0(cstor, **kwargs):
    rs = dict()
    rs['cstor_turnover'] = kwargs['br_xr'] * (3600.*24.) * cstor * kwargs['trate']
    rs['cstor_alloc'] = kwargs['availc'] * (1. - kwargs['fpg'])
    rs['xsmr'] = max(kwargs['mr'] -kwargs['gpp'], 0.)
    rs['c_mortality'] = kwargs['r_mort'] / 365. * cstor
    return rs

def nullcline_v0(**kwargs):
    U = kwargs['br_xr']*(3600.*24.)*kwargs['trate'] + kwargs['r_mort']/365.
    N = kwargs['availc']*(1. - kwargs['fpg']) - max(kwargs['mr']-kwargs['gpp'], 0.)
    return N/U
    
    


def stoich_v1(rates):
    vec = dict()
    vec['cstor'] = rates['photosynthesis'] - rates['growth_respiration']\
        - rates['maintenance_respiration'] - rates['growth'] - rates['excess_respiration']
    return vec

def rates_v1(cstor, **kwargs):
    rs = dict()
    rs['photosynthesis'] = kwargs['gpp']
    rs['growth_respiration'] = kwargs['gr']
    rs['maintenance_respiration'] = kwargs['mr']
    rs['growth'] = kwargs['fpg'] * kwargs['availc']
    rs['excess_respiration'] = kwargs['br_xr']*(3600.*24.)*cstor*kwargs['trate']
    return rs
 
def nsc_conc(output, pft, step):
    pools = ['leafc_pft', 'leafc_stor_pft', 'frootc_pft',
         'frootc_stor_pft', 'livestemc_pft', 'deadstemc_pft', 
         'livecrootc_pft','deadcrootc_pft']
    cmass = sum(output[pool][pft,step] for pool in pools)
    cmass += output['cstor_pft'][pft,step]
    return output['cstor_pft'][pft,step]/cmass


# ALIAS FOR ACCESS
stoich = stoich_v1
rates = rates_v1
nullcline = nullcline_v0

 #Nutrient limitation
 # availc[p]      = max(gpp[p,v+1]-mr[p,v+1],0.0)
 # xsmr[p] = max(mr[p,v+1]-gpp[p,v+1],0.0)

 # callom[p] = (1.0+frg)*(1.0 + f1 + f2*(1+f3))

 # if (calc_nlimitation):
 #   rc = 3.0 * max(annsum_npp[p] * nallom[p]/callom[p], 0.01)
 #   r = max(1.0, rc/max(nstor[p,v],1e-15))
 #   plant_nalloc[p] = (plant_ndemand[p] + annsum_retransn[p]*gpp[p,v+1]/annsum_gpp[p]) / r
 #   fpg[p,v] = 1/r    #Growth limiation due to npool resistance
 #   cstor_alloc[p] = availc[p] * (1.0 - fpg[p,v])
 # else:
 #   fpg[p,v] = parms['fpg'][p]
 #   cstor_alloc[p] = availc[p] * (1.0 - parms['fpg'][p])
 # gr[p,v+1] = availc[p] * fpg[p,v] * frg * (1.0 + f1+f2*(1+f3))/callom[p]
 
 
 # cstor_turnover[p] = parms['br_xr'][p] * (3600.*24.) * cstor[p,v] * trate
 
 # cstor[p,v+1]       = cstor[p,v] + cstor_alloc[p] - parms['r_mort'][0] / 365.0 * \
 #         cstor[p,v] - cstor_turnover[p] - xsmr[p]
 # #Increment plant N pools
 # if (calc_nlimitation):
 #   nstor[p,v+1] = nstor[p,v] - parms['r_mort'][0] / 365.0 * nstor[p,v] + \
 #           retransn[p] - plant_nalloc[p] + fpi*plant_ndemand[p]  
