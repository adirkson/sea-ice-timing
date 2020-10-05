#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:00:53 2019

@author: arlan

Just load an IFD file and explore the data a bit
"""

import sitdates

import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import cm
import matplotlib as mpl


###############################################################################
################## User-defined variables #####################################
###############################################################################

event = 'ifd'
tc = 0.5 # threshold SIC for the event (needed to apply observation mask)

# initialization month
im = 7
im_id = "%02d" % im

# minimum and maximum possible dates (see seaice_timing_time documentation)
si_time = sitdates.sitdates(event=event)
a = si_time.pre_occurence(im)
b = si_time.non_occurence(im)


# Climatology years (for computing forecast anomalies and event probabilities; 
# also just for creating filenames)
clim_yr_s = 2008
clim_yr_f = 2017

# Forecast year (also just for creating filenames)
fcst_yr = 2018

# the directories where the data are located and where the output file will be saved
dir_out = './Data/'

ncgr_netcdf = dir_out+event+'_fcst_'+str(fcst_yr)+'_im'+im_id+'_ncgr_0.05.nc'

# load observed SIC for day prior to the initialization date
sic_obs = Dataset(dir_out+'seaice_conc_daily_nh_f17_20180531_v03r01_rg.nc').variables['seaice_conc_cdr'][:][0]


###########################
fcst_file = Dataset(ncgr_netcdf)
ncgr_p_en = fcst_file['prob_EN'][:][0]
ncgr_p_nn = fcst_file['prob_NN'][:][0]
ncgr_p_ln = fcst_file['prob_LN'][:][0]

fill_value = fcst_file['prob_EN']._FillValue

ncgr_p_en[ncgr_p_en==fill_value] = np.nan
ncgr_p_nn[ncgr_p_nn==fill_value] = np.nan
ncgr_p_ln[ncgr_p_ln==fill_value] = np.nan

ncgr_p_en[sic_obs<tc] = np.nan
ncgr_p_nn[sic_obs<tc] = np.nan
ncgr_p_ln[sic_obs<tc] = np.nan


lat = fcst_file['latitude'][:]
lon = fcst_file['longitude'][:]

LON, LAT = np.meshgrid(lon,lat)


# prep arrays to be filled with most likely category probability
ncgr_p_en_new = np.zeros(ncgr_p_en.shape)
ncgr_p_nn_new = np.zeros(ncgr_p_nn.shape)
ncgr_p_ln_new = np.zeros(ncgr_p_ln.shape)

# if category is most likely, set to the probability for that category (else it will be zero)
ncgr_p_en_new[ncgr_p_en==np.nanmax(np.array([ncgr_p_en,ncgr_p_nn,ncgr_p_ln]),axis=0)] = ncgr_p_en[ncgr_p_en==np.nanmax(np.array([ncgr_p_en,ncgr_p_nn,ncgr_p_ln]),axis=0)]
ncgr_p_nn_new[ncgr_p_nn==np.nanmax(np.array([ncgr_p_en,ncgr_p_nn,ncgr_p_ln]),axis=0)] = ncgr_p_nn[ncgr_p_nn==np.nanmax(np.array([ncgr_p_en,ncgr_p_nn,ncgr_p_ln]),axis=0)]
ncgr_p_ln_new[ncgr_p_ln==np.nanmax(np.array([ncgr_p_en,ncgr_p_nn,ncgr_p_ln]),axis=0)] = ncgr_p_ln[ncgr_p_ln==np.nanmax(np.array([ncgr_p_en,ncgr_p_nn,ncgr_p_ln]),axis=0)]


########### Plotting  #####################
def Add_Lon_Data(data,LAT,LON):
    # adds a synthetic longitude to data at 360 in order to get
    # rid of plotting discontinuity at 0/360
    lat_new = LAT[:,0]
    lon_new = np.append(LON[0,:],360.)
    LON_new, LAT_new = np.meshgrid(lon_new,lat_new)
    data_new = np.zeros(LON_new.shape)
    data_new[:,:-1] = np.copy(data)
    data_new[:,-1] = np.copy(data[:,-1])
    return data_new

def Add_Lon_Grid(LAT,LON):
    # adds a synthetic longitude to the LAT and LON matrices
    lat_new = LAT[:,0]
    lon_new = np.append(LON[0,:],360.)
    LON_new, LAT_new = np.meshgrid(lon_new,lat_new)
    
    return LAT_new, LON_new, LAT_new.shape[0], LAT_new.shape[1]

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # cuts off the ends of cmap colors at minval and maxval
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
    cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def set_up_subplot(fig,subplot=111):
    crs_np = ccrs.NorthPolarStereo(central_longitude=-45)
    ax = fig.add_subplot(subplot,projection=crs_np)
    
    xll, yll = crs_np.transform_point(279.26,33.92, ccrs.Geodetic())
    xur, yur = crs_np.transform_point(102.34,31.37, ccrs.Geodetic())
    
    ax.set_extent([xll,xur,yll,yur],crs=crs_np)

    ax.add_feature(cfeature.OCEAN,facecolor='c', zorder=1)
    ax.add_feature(cfeature.LAND,facecolor='0.75', zorder=3)
    ax.add_feature(cfeature.LAKES,facecolor='c',linestyle='-', edgecolor='k',zorder=3)
    ax.coastlines(resolution='110m',linewidth=1,color='k',zorder=3)       
    return ax

clevs = [0.4, 0.6, 0.8, 1.0]
clevs_lab = ['40','60','80','100']
clevs_lab = [n+'%' for n in clevs_lab]
clevs_ticks = np.array(clevs)

# colormaps for each category
cmap_en = cm.YlOrRd
cmap_nn = cm.Greens
cmap_ln = cm.Blues

cmap_en = truncate_colormap(cmap_en,0.3,1.0)
cmap_nn = truncate_colormap(cmap_nn,0.3,1.0)
cmap_ln = truncate_colormap(cmap_ln,0.3,1.0)

cmap_en.set_under('0.75')
cmap_nn.set_under('0.75')
cmap_ln.set_under('0.75')

cmap_en.set_over(cm.YlOrRd(256))
cmap_nn.set_over(cm.Greens(256))
cmap_ln.set_over(cm.Blues(256))

norm_en = mpl.colors.BoundaryNorm(clevs, cmap_en.N)
norm_nn = mpl.colors.BoundaryNorm(clevs, cmap_nn.N)
norm_ln = mpl.colors.BoundaryNorm(clevs, cmap_ln.N)

LAT_new, LON_new, nlat_new, nlon_new = Add_Lon_Grid(LAT+2.767273/2., LON)
ncgr_p_en_new = Add_Lon_Data(ncgr_p_en_new, LAT, LON)
ncgr_p_nn_new = Add_Lon_Data(ncgr_p_nn_new, LAT, LON)
ncgr_p_ln_new = Add_Lon_Data(ncgr_p_ln_new, LAT, LON)


fig = plt.figure(num=1,figsize=(8.5,9))
plt.clf()

#############################################################
ax = set_up_subplot(fig)
datain1 = np.copy(ncgr_p_en_new)
masked_array1 = np.ma.array(datain1, mask=datain1==0.0)

datain2 = np.copy(ncgr_p_nn_new)
masked_array2 = np.ma.array(datain2, mask=datain2==0.0)

datain3 = np.copy(ncgr_p_ln_new)
masked_array3 = np.ma.array(datain3, mask=datain3==0.0)


im1 = ax.pcolormesh(LON_new,LAT_new,masked_array1,vmin=0.4,vmax=1.0,
              cmap=cmap_en,norm=norm_en,rasterized=True,transform=ccrs.PlateCarree(), zorder=2) 

im2 = ax.pcolormesh(LON_new,LAT_new,masked_array2,vmin=0.4,vmax=1.0,
              cmap=cmap_nn,norm=norm_nn,rasterized=True,transform=ccrs.PlateCarree(), zorder=2)

im3 = ax.pcolormesh(LON_new,LAT_new,masked_array3,vmin=0.4,vmax=1.0,
              cmap=cmap_ln,norm=norm_ln,rasterized=True,transform=ccrs.PlateCarree(), zorder=2)


ax.outline_patch.set_linewidth(1.5)

# ax.scatter(LON[6,78],LAT[6,78], marker='o',s=20, c='k',
#            transform=ccrs.PlateCarree(),zorder=2)


cbar_ax1 = fig.add_axes([0.035, 0.04, 0.025, 0.25])
cb1 = fig.colorbar(im1,cax=cbar_ax1,orientation='vertical',format='%d', ticks=clevs_ticks,
             spacing='uniform', drawedges=True, extend='min', boundaries=[0]+clevs, extendfrac='auto')
cbar_ax1.set_yticklabels(clevs_lab,fontsize=10)

cbar_ax2 = fig.add_axes([0.035, 0.34, 0.025, 0.25])
cb2 = fig.colorbar(im2,cax=cbar_ax2,orientation='vertical',format='%d', ticks=clevs_ticks,
             spacing='uniform',drawedges=True, extend='min', boundaries=[0]+clevs, extendfrac='auto')
cbar_ax2.set_yticklabels(clevs_lab,fontsize=10)

cbar_ax3 = fig.add_axes([0.035, 0.63, 0.025, 0.25])
cb3 = fig.colorbar(im3,cax=cbar_ax3,orientation='vertical',format='%d', ticks=clevs_ticks,
             spacing='uniform',drawedges=True, extend='min', boundaries=[0]+clevs,  extendfrac='auto')

cbar_ax1.set_yticklabels(clevs_lab,fontsize=14)
cbar_ax2.set_yticklabels(clevs_lab,fontsize=14)
cbar_ax3.set_yticklabels(clevs_lab,fontsize=14)

cbar_ax1.set_ylabel('Early', fontsize=16,fontweight='semibold')
cbar_ax2.set_ylabel('Near-normal', fontsize=16,fontweight='semibold')
cbar_ax3.set_ylabel('Late', fontsize=16,fontweight='semibold')

cbar_ax1.yaxis.set_label_position('left')
cbar_ax2.yaxis.set_label_position('left')
cbar_ax3.yaxis.set_label_position('left')


cb1.outline.set_linewidth(2)
cb1.outline.set_edgecolor('k')
cb1.dividers.set_color('w')
cb1.dividers.set_linewidth(1.5)

cb2.outline.set_linewidth(2)
cb2.outline.set_edgecolor('k')
cb2.dividers.set_color('w')
cb2.dividers.set_linewidth(1.5)

cb3.outline.set_linewidth(2)
cb3.outline.set_edgecolor('k')
cb3.dividers.set_color('w')
cb3.dividers.set_linewidth(1.5)

fig.text(0.065,0.06,'EC',fontsize=14)
fig.text(0.065,0.36,'EC',fontsize=14)
fig.text(0.065,0.65,'EC',fontsize=14)

fig.subplots_adjust(left=0.05, right=0.98, top=0.91, bottom=0.01)

ax.set_title('Probability for Early, Near-normal, or Late '+event.upper()+' \n From '+im_id+'/'+str(fcst_yr)+' (cf '+str(clim_yr_s)+'-'+str(clim_yr_f)+')',
        fontsize=20,pad=10.)

# plt.savefig(dir_out+'/figure.png',dpi=500)