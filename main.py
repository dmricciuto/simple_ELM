##!/usr/bin/python
import model_DALEC as models
import eval_DALEC 
import os, sys, math, random
import forcings
from utils import *

site = 'US-SPR'
#create model object
model = models.MyModel()
#Load model forcings
forcings.load(model, site=site)

#run with the default parameters  (see above for definitions of the parameters)
parms = [20,  20, -2, -2,  15, 15, 15, 15, 0.3, 0.3, 0.5, 0.7, 0.05, 50, 5, 2, 2.52e-6, 2, 2, 3, 50, 0.01]
rmse = eval_DALEC.rmse(parms, model)

print rmse
