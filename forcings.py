from netCDF4 import Dataset
import numpy

def load(model, site='none', lat=-999, lon=-999):
         #Get single point data from E3SM style cpl_bypass input files
         model.forcvars = ['tmax','tmin','rad','cair','doy','dayl','btran','time']
         model.forcings = {}
         model.site=site

         if (model.site != 'none'):
             #Get data for requested site
             myinput = Dataset('./forcing_data/'+model.site+'_forcing.nc4','r',format='NETCDF4')
             npts = myinput.variables['TBOT'].size              #number of half hours or hours
             tair = myinput.variables['TBOT'][0,:]              #Air temperature (K)
             fsds  = myinput.variables['FSDS'][0,:]             #Solar radiation (W/m2)
             btran = fsds * 0.0 + 1.0
             model.latdeg = myinput.variables['LATIXY'][0]            #site latitude
             model.londeg = myinput.variables['LONGXY'][0]            #site longitude
             model.start_year = int(myinput.variables['start_year'][:])    #starting year of data
             model.end_year   = int(myinput.variables['end_year'][:])   #ending year of data
             model.npd = int(npts/(model.end_year - model.start_year + 1)/365)   #number of obs per day
             model.nobs = int((model.end_year - model.start_year + 1)*365)  #number of days
             for fv in model.forcvars:
               model.forcings[fv] = []
             model.lat = model.latdeg*numpy.pi/180.
             myinput.close()
             #populate daily forcings

             for d in range(0,model.nobs):
               model.forcings['tmax'].append(max(tair[d*model.npd:(d+1)*model.npd])-273.15)
               model.forcings['tmin'].append(min(tair[d*model.npd:(d+1)*model.npd])-273.15)
               model.forcings['rad'].append(sum(fsds[d*model.npd:(d+1)*model.npd]*(86400/model.npd)/1e6))
               model.forcings['btran'].append(1.0)
               model.forcings['cair'].append(360)
               model.forcings['doy'].append((float(d % 365)+1))
               model.forcings['time'].append(model.start_year+d/365.0)
               #Calculate day length
               dec  = -23.4*numpy.cos((360.*(model.forcings['doy'][d]+10.)/365.)*numpy.pi/180.)*numpy.pi/180.
               mult = numpy.tan(model.lat)*numpy.tan(dec)
               if (mult >= 1.):
                 model.forcings['dayl'].append(24.0)
               elif (mult <= -1.):
                  model.forcings['dayl'].append(0.)
               else:
                 model.forcings['dayl'].append(24.*numpy.arccos(-mult)/numpy.pi)

         elif (lat >= -90 and lon >= -180):
             #Get closest gridcell from reanalysis data
             model.latdeg=lat
             if (lon > 180):
                 lon=lon-360.
             if (lat > 9.5 and lat < 79.5 and lon > -170.5 and lon < -45.5):
                xg = int(round((lon + 170.25)*2))
                yg = int(round((lat - 9.75)*2))
                tmax = model.regional_forc['tmax'][:,yg,xg]
                tmin = model.regional_forc['tmin'][:,yg,xg]
                btran = model.regional_forc['btran'][:,yg,xg]
                fsds = model.regional_forc['fsds'][:,yg,xg]
             else:
                print('regions outside North America not currently supported')
                sys.exit(1)
             model.start_year = 1980
             model.end_year   = 2009
             model.npd = 1
             model.nobs = (model.end_year - model.start_year + 1)*365
             model.lat = model.latdeg*numpy.pi/180.

             #populate daily forcings
             model.forcings['tmax']  = tmax-273.15
             model.forcings['tmin']  = tmin-273.15
             model.forcings['btran'] = btran
             model.forcings['rad']   = fsds*86400/1e6
             model.forcings['cair']  = numpy.zeros([model.nobs], numpy.float) + 360.0
             model.forcings['doy']   = (numpy.cumsum(numpy.ones([model.nobs], numpy.float)) - 1) % 365 + 1
             model.forcings['time']  = model.start_year + (numpy.cumsum(numpy.ones([model.nobs], numpy.float)-1))/365.0
             model.forcings['dayl']  = numpy.zeros([model.nobs], numpy.float)
             for d in range(0,model.nobs):
               #Calculate day length
               dec  = -23.4*numpy.cos((360.*(model.forcings['doy'][d]+10.)/365.)*numpy.pi/180.)*numpy.pi/180.
               mult = numpy.tan(model.lat)*numpy.tan(dec)
               if (mult >= 1.):
                 model.forcings['dayl'][d] = 24.0
               elif (mult <= -1.):
                 model.forcings['dayl'][d] = 0.
               else:
                 model.forcings['dayl'][d] = 24.*numpy.arccos(-mult)/numpy.pi

