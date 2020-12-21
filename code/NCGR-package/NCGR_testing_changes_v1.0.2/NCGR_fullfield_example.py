#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:12:28 2020

@author: arlan
"""

# from NCGR import ncgr
import ncgr
from sitdates import sitdates
import time


# input filenames
hc_netcdf = './Data/fud_hc_1981_2019_im10.nc' 
obs_netcdf = './Data/fud_obs_1981_2019_im10.nc' 
fcst_netcdf = './Data/fud_fcst_2020_im10.nc' 
clim_netcdf = './Data/fud_clim_2011_2019_im10.nc'   

# output filename (this usually doesn't exist yet)
out_netcdf = './Data/fud_fcst_ncgr_2020_im10.nc'

# Dictionary defining the relevant variables/dimensions for hc_netcdf and fcst_netcdf
model_dict = ({'event_vn' : 'fud', # variable name for the ice-free date or freeze-up date variable
                   'time_vn' : 'time'}, # variable name for the time coordinate
                  {'time_dn' : 'time', # dimension name for the time coordinate
                   'ens_dn' : 'ensemble'}) # dimension name for the forecast realization/ensemble coordinate

# Dictionary defining the relevant variables/dimensions for obs_netcdf
obs_dict = ({'event_vn' : 'fud',  # variable name for the ice-free date or freeze-up date variable
                 'time_vn' : 'time'}, # variable name for the time coordinate
                {'time_dn' : 'time'}) # dimension name for the time coordinate

im = 10 # initialization month
si_time = sitdates(event='fud')
a = si_time.get_min(im)
b = si_time.get_max(im)

# # calibrate 
start = time.time()
ncgr.ncgr_fullfield(fcst_netcdf, hc_netcdf, obs_netcdf, out_netcdf,
                  a, b, model_dict, obs_dict, 
                  clim_netcdf=clim_netcdf) 
end = time.time()

print("time elapsed (minutes)", (end-start)/60.)