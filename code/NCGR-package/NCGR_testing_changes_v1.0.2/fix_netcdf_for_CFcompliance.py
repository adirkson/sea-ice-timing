#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 14:41:37 2018

@author: arlan
"""

import os
import numpy as np
import netCDF4
from datetime import datetime

######################################## Change these variables accordingly
event = 'ifd'

im='06'
years = np.arange(1979,2018+1)

# change this to where data are
path = '/home/arlan/Dropbox/UQAM_Postdoc/Projects/IFD_FUD/Code/Probabilistic/NCGR_cnorm/sea-ice-timing/code/NCGR-package/NCGR_testing_changes/Data'

##########################################
obs=True
fname_in = 'ifd_1979_2018_timeseries_im06_filled_1_CanCM_masked.nc'
fname_out = 'ifd_obs_1979_2018_im06.nc'

# fname_in = 'ifd_1979_2018_timeseries_im06_NEW_original_grid.nc'
# fname_out = 'ifd_hcsts_1979_2018_im06.nc'



def get_nc_time(im,years):  
    units = 'days since 1900-01-01'
    calendar = 'standard'
        
    nc_time = np.array([]) # the time variable that will be outputed in the concatenated netcdf

    for year in years:
        nc_time = np.append(nc_time, datetime(year,im,1))
        
    nc_time = netCDF4.date2num(nc_time, units=units, calendar=calendar)
    
    return nc_time



# Load and re-write netCDFs
os.chdir(path)
if os.path.exists(fname_out):
    os.remove(fname_out)

with netCDF4.Dataset(fname_in) as src,  netCDF4.Dataset(fname_out, "w") as dst:
    # copy global attributes all at once via dictionary
    dst.setncatts(src.__dict__)
    # copy dimensions except replace the time_counter with just 'time' and remove the original time dimension
    for name, dimension in src.dimensions.items():

            dst.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None)) 

            
    # copy all file data except for time and ifd/fud (fix these)
    for name, variable in src.variables.items():                           
        if name==event:
            if obs==True:
                outVar = dst.createVariable(name, variable.datatype, ('time','latitude','longitude'))
            else:
                outVar = dst.createVariable(name, variable.datatype, ('time','ensemble','latitude','longitude'))
                
            outVar.long_name = 'ice-free date'
            outVar[:] = src[name][:]
           
        else:

            if name=='time':
                outVar = dst.createVariable('time', np.float32, ('time'))
                outVar[:] = get_nc_time(int(im),years).astype(np.float32)
                outVar.units = 'days since 1900-01-01'
                outVar.calendar = 'standard'
                outVar.long_name = 'time'
            else:            
                dst.createVariable(name, variable.datatype, variable.dimensions)
                dst[name][:] = src[name][:]
                dst[name].setncatts(src[name].__dict__)
            
            