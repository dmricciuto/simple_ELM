# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 11:48:31 2022

@author: fso
"""
import numpy as np
from nsc_vec import GPP, Rm, LAI
from scipy.special import softmax
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def build_dok():
    pools = ['leafc','leafc_stor','frootc','frootc_stor',
             'livestemc','deadstemc','livecrootc','deadcrootc', 
             'cstor', 'nstor']
    envs = ['temp']
    rates = ['GPP', 'resp', 'growth', 'plant_nalloc']
    for var in ['leafc', 'frootc']:
        for proc in ['growth', 'trans', 'litter']:
            rates.append(var + '_' + proc)
            if proc == 'growth':
                rates.append(var + '_stor_' + proc)
            
    for org in ['stemc', 'crootc']:
        for state in ['live', 'dead']:
            var = state + org
            for proc, val in zip(['growth','mortality'], [1,-1]):
                process = var + "_" + proc
                rates.append(process)
        rates.append('live'+org+'_turnover')
    

    rates.append('plant_ndemand')
    rates.append('retransn')
    
    rates.append('cstor_mortality')
    rates.append('nstor_mortality')

    pool_dok = { pool : n for n, pool in enumerate(pools)}
    rate_dok = { rate : n for n, rate in enumerate(rates)}
    env_dok = { env : n for n, env in enumerate(envs)}
    return pools, rates, envs, pool_dok, rate_dok, env_dok


#increment plant C pools
# leafc[p,v+1]       = leafc[p,v]       + fpg[p,v]*leafc_alloc[p,v] + leafc_trans[p] - leafc_litter[p]
# leafc_stor[p,v+1]  = leafc_stor[p,v]  + fpg[p,v]*leafcstor_alloc[p] - leafc_trans[p]
# frootc[p,v+1]      = frootc[p,v]      + fpg[p,v]*frootc_alloc[p,v] + frootc_trans[p] - frootc_litter[p]
# frootc_stor[p,v+1] = frootc_stor[p,v] + fpg[p,v]*frootcstor_alloc[p] - frootc_trans[p]
# livestemc[p,v+1]   = livestemc[p,v]   + fpg[p,v]*livestemc_alloc[p,v] - parms['r_mort'][0] \
#         / 365.0 * livestemc[p,v] - livestemc_turnover[p]
# deadstemc[p,v+1]   = deadstemc[p,v]   + fpg[p,v]*deadstemc_alloc[p,v] - parms['r_mort'][0] \
#         * mort_factor / 365.0 * deadstemc[p,v] + livestemc_turnover[p]
# livecrootc[p,v+1]  = livecrootc[p,v]  + fpg[p,v]*livecrootc_alloc[p] - parms['r_mort'][0] \
#         / 365.0 * livecrootc[p,v] - livecrootc_turnover[p]
# deadcrootc[p,v+1]  = deadcrootc[p,v]  + fpg[p,v]*deadcrootc_alloc[p] - parms['r_mort'][0] \
#         * mort_factor / 365.0 * deadcrootc[p,v] + livecrootc_turnover[p]
#  # self.output['cstor_nullcline_pft'][p,v] = nsc.nullcline(**kwargs)
#  cstor[p,v+1]       = cstor[p,v] + nsc.stoich(cstor_rates)['cstor']
#  #Increment plant N pools
#  if (calc_nlimitation):
#    nstor[p,v+1] = nstor[p,v] - parms['r_mort'][p] / 365.0 * nstor[p,v] + \
#            retransn[p] - plant_nalloc[p] + fpi*plant_ndemand[p]  

def build_stoich_matrix(pools, rates, 
                        pool_dok, rate_dok, sparse=False):
    shape = len(pools), len(rates)
    
    if not sparse:
        Gamma = np.zeros(shape)
    else:
        Gamma = sp.dok_matrix(shape,dtype=np.float64)

    # PLANT STRUCTURAL CARBON POOLS    
    for var in ['leafc', 'frootc']:
        for proc, val in zip(['growth', 'trans', 'litter'], [1,1,-1]):
            Gamma[pool_dok[var], rate_dok[var + '_' + proc]] = val
        
        Gamma[pool_dok[var + '_stor'], rate_dok[var + '_stor_growth']] = 1
        Gamma[pool_dok[var + '_stor'], rate_dok[var + '_trans']] = -1
    
    for org in ['stemc', 'crootc']:
        for state in ['live', 'dead']:
            var = state + org
            for proc, val in zip(['growth','mortality'], [1,-1]):
                process = var + "_" + proc
                Gamma[pool_dok[var], rate_dok[process]] = val
            if state == 'live':
                Gamma[pool_dok[var], rate_dok['live'+org+'_turnover']] = -1
            else:
                Gamma[pool_dok[var], rate_dok['live'+org+'_turnover']] = 1
    
    # PLANT LABILE POOLS
    
    Gamma[pool_dok['cstor'], rate_dok['GPP']] = 1
    Gamma[pool_dok['cstor'], rate_dok['resp']] = -1
    Gamma[pool_dok['cstor'], rate_dok['growth']] = -1    
    
    Gamma[pool_dok['nstor'], rate_dok['retransn']] = 1
    Gamma[pool_dok['nstor'], rate_dok['plant_ndemand']] = 1
    Gamma[pool_dok['nstor'], rate_dok['plant_nalloc']] = -1

    for pool in ['cstor', 'nstor']:
        Gamma[pool_dok[pool], rate_dok[pool + '_mortality']] = -1
    
    return Gamma

def build_rate_vec(rates,pool_dok,rate_dok,env_dok):
    def rates_vec(xs, es, ps):
        '''
        Parameters
        ----------
        xs : 1D array
            Array of states.
        es : 1D array
            Array of environmental variables.
        ps : 1D array
            Array of parameters.
    
        Returns
        -------
        rs : 1D array
            Array of process rates.
        '''
        rs = np.empty(len(rates))
        
        rs[rate_dok['GPP']] = GPP(
            es[env_dok['rad']],
            es[env_dok['cair']],
            es[env_dok['temp']],
            LAI(xs[pool_dok['leafc']],ps), 
            es[env_dok['dayl']],
            parms = ps
            )
    
        rs[rate_dok['resp']] = Rm(
            xs[pool_dok['leafc']], 
            xs[pool_dok['frootc']], 
            xs[pool_dok['livecrootc']], 
            xs[pool_dok['livestemc']], 
            es[env_dok['temp']]
            )
        
        for pool in ['stemc','crootc']:
            var = 'live' + pool
            c = ps['lwtop_ann'][0]/365.
            rs[rate_dok[var + '_turnover']] = c*xs[pool_dok[var]]
        
        for pool in ['livestemc','livecrootc','cstor', 'nstor']:
            rs[rate_dok[pool]] = ps['r_mort']/365. * xs[pool_dok[pool]]
            
        for pool in ['deadstemc','deadcrootc']:    
            cons = ps['r_mort']/365. * ps['mort_factor']
            rs[rate_dok[pool]] = cons * xs[pool_dok[pool]]
        return rs

def build_vec():
    pools,rates, envs, pool_dok,rate_dok,env_dok =build_dok()
    Gamma = build_stoich_matrix(pools, rates, pool_dok, rate_dok)
    rates_vec = build_rate_vec(rates, pool_dok, rate_dok, env_dok)
    def vec(xs, es, ps):
        rs = rates_vec(xs, es, ps)
        return Gamma.dot(rs)
    return vec



if __name__ == "__main__":
    pools,rates,pool_dok,rate_dok = build_dok()
    G = build_stoich_matrix(pools,rates,pool_dok,rate_dok)
    print(np.linalg.matrix_rank(G))
    print(G.shape)