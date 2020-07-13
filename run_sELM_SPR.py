#!/usr/bin/python
import model_sELM as models
import os, sys, math, random
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import argparse
import forcings

from utils import *

# default parameters
#nens  = 5   # ensemble size

parser = argparse.ArgumentParser()
site = 'US-SPR'

print('Processing site: %s'%(site))

#create model object
model = models.MyModel()

#Load model forcings
forcings.load(model, site=site)

# ------------------- Run and analyze the default model ---------
#Run model with default parameters
myoutvars=['cpool_pft', 'sminn_vr', 'fpi_vr', 'fpg_pft', 'gpp_pft','lai_pft','npp_pft','totsomc','deadstemc_pft']
model.run_selm(spinup_cycles=6, use_nn=True, do_monthly_output=True,myoutvars=myoutvars) #try this

