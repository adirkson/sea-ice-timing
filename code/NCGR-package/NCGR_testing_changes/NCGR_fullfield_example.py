#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:12:28 2020

@author: arlan
"""

# from NCGR import ncgr
import ncgr
import sitdates


# input filenames
hc_netcdf = './Data/ifd_hc_1979_2017_im06.nc' 
obs_netcdf = './Data/ifd_obs_1979_2017_im06.nc' 
fcst_netcdf = './Data/ifd_fcst_2018_im06.nc' 
clim_netcdf = './Data/ifd_clim_2008_2017_im06.nc' 

# output filename (this usually doesn't exist yet)
out_netcdf = './Data/ifd_fcst_2018_im06_ncgr.nc'

event='ifd'
im = 6

si_time = sitdates.sitdates(event=event)
a = si_time.pre_occurence(im)
b = si_time.non_occurence(im)

# calibrate 
ncgr.ncgr_fullfield(hc_netcdf, obs_netcdf, fcst_netcdf, out_netcdf, event,
                  a, b, clim_netcdf=clim_netcdf) 