#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:12:28 2020

@author: arlan
"""

# from NCGR import ncgr
import ncgr
import sitdates
import time


# input filenames
hc_netcdf = './Data/ifd_hc_1979_2017_im06.nc' 
obs_netcdf = './Data/ifd_obs_1979_2017_im06.nc' 
fcst_netcdf = './Data/ifd_fcst_2018_im06.nc' 
clim_netcdf = './Data/ifd_clim_2008_2017_im06.nc' 

# output filename (this usually doesn't exist yet)
out_netcdf = './Data/ifd_fcst_2018_im06_ncgr_0.05.nc'

# Dictionary defining the relevant variables/dimensions hc_netcdf and fcst_netcdf
model_dict = ({'event_vn' : 'ifd',
                   'time_vn' : 'time'},
                  {'time_dn' : 'time',
                   'ens_dn' : 'ensemble'})

# Dictionary defining the relevant variables/dimensions obs_netcdf
obs_dict = ({'event_vn' : 'ifd',
                 'time_vn' : 'time'},
                {'time_dn' : 'time'})

im = 6

si_time = sitdates.sitdates(event='ifd')
a = si_time.pre_occurence(im)
b = si_time.non_occurence(im)

# calibrate 
start = time.time()
ncgr.ncgr_fullfield(fcst_netcdf,hc_netcdf, obs_netcdf, out_netcdf,
                  a, b, model_dict, obs_dict,
                  clim_netcdf=clim_netcdf) 
end = time.time()

print("time elapsed (minutes)", (end-start)/60.)