import numpy

def rmse(parms, model):
  #Set model parameters from the input array
  model.parms['gdd_min'][1] = parms[0]  #Growing degree day threshold for Larch
  model.parms['gdd_min'][2] = parms[1]  #Growing degree day threshold for Shruyb
  model.parms['tsmin'][1] = parms[2]    #Leaf off temperature for Larch
  model.parms['tsmin'][2] = parms[3]    #Leaf off temperature for Shrub
  model.parms['nue'][0] = parms[4]   #NUE for Black Spruce
  model.parms['nue'][1] = parms[5]   #NUE for Larch
  model.parms['nue'][2] = parms[6]   #NUE for shrub
  model.parms['nue'][3] = parms[7]   #NUE for Sphagnum
  model.parms['aleaf'][0] = parms[8] #Leaf allocation fraction, black spruce
  model.parms['aleaf'][1] = parms[9] #Leaf allocation fraction, Larch
  model.parms['aleaf'][2] = parms[10] #Leaf allocation fraction, Shrub
  model.parms['astem'][0:2] = parms[11]  #Stem allocation fraction, both tree types
  model.parms['astem'][2] = parms[12]    #Stem allocation fraction, shrub
  model.parms['tstem'][0:2] = parms[13]  #Stem turnover for trees
  model.parms['tstem'][2] = parms[14]    #Stem turnover for shrubs
  model.parms['troot'][0:3] = parms[15]  #Root turnover (tree, shrubs)
  model.parms['br_mr'][0] = parms[16]    #base rate for MR
  model.parms['q10_mr'][0] = parms[17]   #Q10 for MR
  model.parms['q10_hr'][0] = parms[18]   #Q10 for HR
  model.parms['br_lit'][0] = parms[19]   #litter turnover time
  model.parms['br_som'][0] = parms[20]   #SOM turnover time
  model.parms['dr'][0] = parms[21]       #Decomposition rate

  #Set the minimum allowable values for these parameters
  pmin = [  5,  5, -5, -5,  3, 3, 3, 3, 0.05, 0.05, 0.20, 0.2, 0.01, 20, 2, 1, 1e-6, 1.2, 1.2, 1, 20, 0.001]
  #Set the maximum allowable value for these parameters
  pmax = [200,200,  5,  5, 20,20,20,20, 0.50, 0.50, 0.80, 0.7, 0.20,100,10, 5, 4e-6, 3.6, 2.6, 5, 1000, 0.1]

  #Check if provided parameters are in range
  inbounds = True
  for p in range(0, len(parms)):
    if parms[p] < pmin[p] or parms[p] > pmax[p]:
      inbounds = False

  if (inbounds):
    #Run the model simulation with the specified parameters
    model.run_dalec(model.parms, ad_cycles = 8, final_cycles=8, trans_startyear=1970)

    #Calculate the cost function for these parameters based on the flux chamber data
    nee_lc_light_obs = numpy.loadtxt('constraints/nee_lc_light.txt', skiprows=1)
    nee_lc_light_model = model.fluxes['nee_lc_light'][-365*7-1:]
    ndata = nee_lc_light_obs.shape[0]
    sse = 0    #Sum of squared errors (weighted by uncertainty)
    ngood = 0

    for i in range(0, ndata):
      index = int((nee_lc_light_obs[i,0]-2013)*365 + nee_lc_light_obs[i,1]-1)
      if (nee_lc_light_obs[i,3] > -999):
        sse = sse + ((nee_lc_light_obs[i,3] - nee_lc_light_model[index]) / nee_lc_light_obs[i,4])**2
        ngood = ngood + 1

    #Calculate the root mean squared error
    rmse = numpy.sqrt(sse/ngood)
  else:
    rmse = 9.9e10

  return rmse
