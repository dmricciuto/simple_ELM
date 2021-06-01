##!/usr/bin/python
import model_sELM as models
import os, sys, math, random
import forcings
from utils import *
import matplotlib.pyplot as plt

site = 'US-Ho1'
#create model object
model = models.MyModel(site=site)
#Load model forcings
model.load_forcings(site=site)

fig, ax = plt.subplots(2,2)
print('running the default model')
model.run_selm(use_nn=True, prefix='1R_default', seasonal_rootalloc=False, spinup_cycles=2,nfroot_orders=1)
ax[0,0].plot(model.output['lai_pft'].squeeze()[0:365],'b')
ax[0,0].set_ylabel('LAI')
ax[0,1].plot(model.output['leafc_pft'].squeeze()[0:365],'b')
ax[0,1].set_ylabel('Leaf C')
ax[1,0].plot(model.output['frootc_pft'].squeeze()[0:365],'b')
ax[1,0].set_ylabel('Froot C')
ax[1,1].plot(model.output['deadstemc_pft'].squeeze()[0:365],'b')
ax[1,1].set_ylabel('Dead stem C')

#create model object
model1 = models.MyModel(site=site)
#Load model forcings
model1.load_forcings(site=site)

print('running model with fine root phenology')
model1.parms['froot_phen_width'][:]=0.6
model1.parms['froot_phen_peak'][:]=0.5
model1.run_selm(use_nn=True, prefix='2R_default', seasonal_rootalloc=False, spinup_cycles=2, nfroot_orders=1)
ax[0,0].plot(model1.output['lai_pft'].squeeze()[0:365],'r')
ax[0,1].plot(model1.output['leafc_pft'].squeeze()[0:365],'r')
ax[1,0].plot(model1.output['frootc_pft'].squeeze()[0:365],'r')
ax[1,1].plot(model1.output['deadstemc_pft'].squeeze()[0:365],'r')
ax[1,1].legend(['Default','Root phenology'])

plt.show()


#print(i, rmse)
