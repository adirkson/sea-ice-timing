#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:00:53 2019

@author: arlan

Just load an IFD file and explore the data a bit
"""

from sitdates import sitdates

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

event = 'fud'
tc = 0.5 # threshold SIC for the event (needed to apply observation mask)

# initialization month
im = 10
im_id = "%02d" % im


# Climatology years (for computing forecast anomalies and event probabilities; 
# also just for creating filenames)
clim_yr_s = 2011
clim_yr_f = 2019

# Forecast year (also just for creating filenames)
fcst_yr = 2020

# the directories where the data are located and where the produced figure will be saved
dir_out = './Data/'

# path+filename of NCGR-calibrated forecast 
ncgr_netcdf = dir_out+event+'_fcst_ncgr_'+str(fcst_yr)+'_im'+im_id+'.nc'

# load observed SIC for day prior to the initialization date 
sic_obs = Dataset(dir_out+'seaice_conc_daily_icdr_nh_f18_20200930_v01r00_rg.nc').variables['seaice_conc_cdr'][:][0]

###########################
si_time = sitdates(event=event)
a = si_time.get_min(im)
b = si_time.get_max(im)

fcst_file = Dataset(ncgr_netcdf)
ncgr_p_en = fcst_file['prob_EN'][:][0]
ncgr_p_nn = fcst_file['prob_NN'][:][0]
ncgr_p_ln = fcst_file['prob_LN'][:][0]

clim_terc_low = fcst_file['clim_1_3'][:][0]
clim_terc_up = fcst_file['clim_2_3'][:][0]

fcst_pre = fcst_file['prob_pre'][:][0]
fcst_non = fcst_file['prob_non'][:][0]

fill_value = fcst_file['prob_EN']._FillValue
    
ncgr_p_en[ncgr_p_en==fill_value] = np.nan
ncgr_p_nn[ncgr_p_nn==fill_value] = np.nan
ncgr_p_ln[ncgr_p_ln==fill_value] = np.nan

if event=='fud':
    ncgr_p_en[sic_obs>tc] = np.nan
    ncgr_p_nn[sic_obs>tc] = np.nan
    ncgr_p_ln[sic_obs>tc] = np.nan
if event=='ifd':
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

# mask for when the climatological 2/3 tercile is equal to the last day of the forecast or season.
clim_ut_mask = np.zeros(ncgr_p_nn_new.shape)
clim_ut_mask[(clim_terc_up==b)&(ncgr_p_nn_new>0.0)] = 1.0

# mask for white color in areas where the IFD (FUD) does not (has already) occurr(ed) at the end (start) of the forecast
ice_mask = np.zeros(ncgr_p_nn_new.shape)
if event=='ifd':
    ice_mask[(fcst_non==1.0)|(sic_obs<tc)] = 1.0
if event=='fud':
    ice_mask[(fcst_pre==1.0)|(sic_obs>tc)] = 1.0
    
########### Plotting  #####################
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
    ax.add_feature(cfeature.LAND,facecolor='0.3', zorder=3)
    ax.add_feature(cfeature.LAKES,facecolor='c',linestyle='-', edgecolor='k',zorder=3)
    ax.coastlines(resolution='110m',linewidth=1,color='k',zorder=3)       
    return ax

clevs = [0.4, 0.6, 0.8, 1.0]
clevs_lab = ['40','60','80','100']
clevs_lab = [n+'%' for n in clevs_lab]
clevs_ticks = np.array(clevs)

# colormaps for each category
if event=='fud':
    cmap_ln = cm.YlOrRd
    cmap_nn = cm.Greens
    cmap_en = cm.Blues
if event=='ifd':
    cmap_en = cm.YlOrRd
    cmap_nn = cm.Greens
    cmap_ln = cm.Blues
    
cmap_en = truncate_colormap(cmap_en,0.3,1.0)
cmap_nn = truncate_colormap(cmap_nn,0.3,1.0)
cmap_ln = truncate_colormap(cmap_ln,0.3,1.0)

cmap_en.set_under('0.75')
cmap_nn.set_under('0.75')
cmap_ln.set_under('0.75')

norm_en = mpl.colors.BoundaryNorm(clevs, cmap_en.N)
norm_nn = mpl.colors.BoundaryNorm(clevs, cmap_nn.N)
norm_ln = mpl.colors.BoundaryNorm(clevs, cmap_ln.N)


#############################################################

fig = plt.figure(num=1,figsize=(8.5,9))
plt.clf()

ax = set_up_subplot(fig)

datain_masked = np.ma.array(ice_mask, mask=ice_mask==0.0)

masked_array1 = np.ma.array(ncgr_p_en_new, mask=ncgr_p_en_new==0.0)
masked_array2 = np.ma.array(ncgr_p_nn_new, mask=ncgr_p_nn_new==0.0)
masked_array3 = np.ma.array(ncgr_p_ln_new, mask=ncgr_p_ln_new==0.0)


ax.pcolormesh(LON,LAT,datain_masked, cmap=cm.Greys,
              rasterized=True,transform=ccrs.PlateCarree(), zorder=2) 

im1 = ax.pcolormesh(LON,LAT,masked_array1,vmin=0.4,vmax=1.0,
              cmap=cmap_en,norm=norm_en,rasterized=True,transform=ccrs.PlateCarree(), zorder=2) 

im2 = ax.pcolormesh(LON,LAT,masked_array2,vmin=0.4,vmax=1.0,
              cmap=cmap_nn,norm=norm_nn,rasterized=True,transform=ccrs.PlateCarree(), zorder=2)

im3 = ax.pcolormesh(LON,LAT,masked_array3,vmin=0.4,vmax=1.0,
              cmap=cmap_ln,norm=norm_ln,rasterized=True,transform=ccrs.PlateCarree(), zorder=2)

    
########### hatching for when the upper tercile for climatology includes the last day ############
label = 'Probability includes \n no '+event.upper()
masked_array4 = np.ma.array(clim_ut_mask, mask=clim_ut_mask==0.0)

plt.rcParams['hatch.color'] = 'white'
plt.rcParams['hatch.linewidth'] = 0.5
cs1 = ax.pcolor(LON,LAT,masked_array4, hatch='xxxxx', alpha=0.,
              rasterized=True,transform=ccrs.PlateCarree(), zorder=2, label=label)

cs2 = mpl.patches.Patch(alpha=0.0, hatch=cs1._hatch, label=label)
l = ax.legend(handles=[cs2], loc='upper right', frameon=False)
for text in l.get_texts():
    text.set_color('w')

######### colorbars
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

ax.outline_patch.set_linewidth(1.5)


fig.subplots_adjust(left=0.05, right=0.98, top=0.91, bottom=0.01)

ax.set_title('Probability for Early, Near-normal, or Late '+event.upper()+' \n From '+im_id+'/'+str(fcst_yr)+' (cf '+str(clim_yr_s)+'-'+str(clim_yr_f)+')',
        fontsize=20,pad=10.)

plt.savefig(dir_out+'/'+event.upper()+'_im'+str(im)+'_3category.png',dpi=700)