##!/usr/bin/python
from numpy import lib
import model_sELM as models
import os, sys, math, random
import forcings
from utils import *
import matplotlib.pyplot as plt

site = 'US-SPR' #US-SPR; US-Ho1; US-MOz; US-MoT
pft = int(0)
lb = int(0)
rb = int(365)

fig, ax = plt.subplots(2,2)

#create model object
model = models.MyModel(site=site)
#Load model forcings
model.load_forcings(site=site)
# generate ensembles
#model.generate_ensemble(n_ensemble=int(2),pnames=['frootcn','leafcn'],ppfts=[2,2])
print('running the default model')
model.run_selm(use_nn=True, use_MPI=False, prefix='3R_default', seasonal_rootalloc=True, spinup_cycles=2)
ax[0,0].plot(model.output['lai_pft'][pft,lb:rb],'b')
ax[0,0].set_ylabel('LAI')
ax[0,1].plot(model.output['livestemc_pft'][pft,lb:rb],'b')
ax[0,1].set_ylabel('Live stem C')
ax[1,0].plot(model.output['frootc_pft'][pft,lb:rb],'b')
ax[1,0].set_ylabel('Froot C')
ax[1,1].plot(model.output['gpp_pft'][pft,lb:rb],'b')
ax[1,1].set_ylabel('GPP')

# #create model object
# model1 = models.MyModel(site=site)
# #Load model forcings
# model1.load_forcings(site=site)
# print('running model with fine root phenology')
# model1.parms['froot_phen_width'][:]=0.6
# model1.parms['froot_phen_peak'][:]=0.5
# model1.run_selm(use_nn=True, prefix='2R_default_ascend', seasonal_rootalloc=False, spinup_cycles=2, nfroot_orders=2)
# ax[0,0].plot(model1.output['lai_pft'][pft,lb:rb],'r')
# ax[0,1].plot(model1.output['leafc_pft'][pft,lb:rb],'r')
# ax[1,0].plot(model1.output['frootc_pft'][pft,lb:rb],'r')
# ax[1,1].plot(model1.output['deadstemc_pft'][pft,lb:rb],'r')
# #ax[1,1].legend(['Default','2-pool fine root'])


# #create model object
# model2 = models.MyModel(site=site)
# #Load model forcings
# model2.load_forcings(site=site,nfroot_orders=3)
# print('running model with fine root phenology')
# model2.parms['froot_phen_width'][:]=0.6
# model2.parms['froot_phen_peak'][:]=0.5
# model2.run_selm(use_nn=True, prefix='3R_default', seasonal_rootalloc=True, spinup_cycles=2, nfroot_orders=3)
# ax[0,0].plot(model2.output['lai_pft'][pft,lb:rb],'red')
# ax[0,1].plot(model2.output['livestemc_pft'][pft,lb:rb],'red')
# ax[1,0].plot(model2.output['frootc_pft'][pft,lb:rb],'red')
# ax[1,1].plot(model2.output['gpp_pft'][pft,lb:rb],'red')
# ax[1,1].legend(['Default', '3-pool fine root'])

plt.show()


#print(i, rmse)
