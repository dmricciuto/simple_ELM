##!/usr/bin/python
import model_sELM_v0 as models
# import os, sys, math, random
# import forcings
# from utils import *
# import matplotlib.pyplot as plt
from numpy import ndarray
from json import dump

site = 'US-Ho1'
#create model object
model = models.MyModel(site=site)
#Load model forcings
model.load_forcings(site=site)

# fig, ax = plt.subplots(2,2)
print('running the default model')

kwargs = dict(use_nn=True, seasonal_rootalloc=False, spinup_cycles=3)
model.run_selm(**kwargs)
# ax[0,0].plot(model.output['lai_pft'].squeeze()[0:365],'b')
# ax[0,0].set_ylabel('LAI')
# ax[0,1].plot(model.output['leafc_pft'].squeeze()[0:365],'b')
# ax[0,1].set_ylabel('Leaf C')
# ax[1,0].plot(model.output['frootc_pft'].squeeze()[0:365],'b')
# ax[1,0].set_ylabel('Froot C')
# ax[1,1].plot(model.output['deadstemc_pft'].squeeze()[0:365],'b')
# ax[1,1].set_ylabel('Dead stem C')



data = {}
for key in model.output:
    datum = model.output[key]
    if isinstance(datum, ndarray):
        datum = datum.tolist()
    
    data[key] = datum

# ADD site and kwargs
data['site'] = site
data['kwargs'] = kwargs

with open('../simple_ELM-output/std_test.json', 'w') as jsonfile:
    dump(data, jsonfile)
