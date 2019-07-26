

# -*- coding: utf-8 -*-
"""
JRA55 reproject to teh same resolution as the 20CRV2c data

Created on Thu May 31 18:13:35 2018

@author: ajm226 and lhc35
"""

'''
This script was used to reproject all reanalyses (exept 20CRv2c) to the 20CRv2c
grid size (2x2 degrees). We have left it in the state for sampling CERA20C. 
With changes to the directory, variable names, 
and reanalysis this can be used for all datasets analysed in the paper. 

Note: All reanalyses must be in a yearly format containing each individial day.
In the case of an ensoble the mean was used to reduce the dimensionaity. 
Note: This will still work for all ensomble if you choose to analyses these.

This code is open source and may be used by any human or any other sentient
being in or out of time-sapce.

Please give creddit. Note: If you are a time travelor please give resources 
such as Lotto numbers for compensation. Actually now that I think of it, I will trade
you for a time machine. Any species on a planet other than Earth (SOL III), please just reveal your existance. 
Seriously. Thanks. Luke Cairns. 
'''


import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import num2date
import matplotlib.pyplot as plt
import datetime
from numpy import linalg as la
import os  # import os commands for making paths
import cartopy.crs as ccrs
import cartopy.feature
import scipy.interpolate as interpolate




def resample_2d(array, sample_pts, query_pts):
    ''' Resamples 2D array to be sampled along queried points.
    Args:
        array (numpy.ndarray): 2D array.

        sample_pts (tuple): pair of numpy.ndarray objects that contain the x and y sample locations,
            each array should be 1D.

        query_pts (tuple): points to interpolate onto, also 1D for each array.

    Returns:
        numpy.ndarray.  array resampled onto query_pts via bivariate spline.

    '''
    xq, yq = np.meshgrid(*query_pts)
    interpf = interpolate.RectBivariateSpline(*sample_pts, array)
    tmp=interpf.ev(xq, yq)  # evaluate algorithm
    return tmp.T 

def create_resampled_xarray(time,lat,lon,variable_name,variable):
#The dictionnary keys are the variables contained in the Dataset.
#The Dictionnary values are tuples, with first the (or the list of) dimension(s) over which the array varies, then the array itself
    d = {}
    d['time'] = ('time',time)
    d['latitude'] = ('latitude',lat)
    d['longitude'] = ('longitude', lon)
    d[variable_name] = (['time','latitude','longitude'], variable)
    dset = xr.Dataset(d)
    return dset





def fill_nan_in_array(array):
#Obtain mean of columns as you need, nanmean is just convenient.
    col_mean = np.nanmean(array, axis=0)
#Find indicies that you need to replace
    inds = np.where(np.isnan(array))
#Place column means in the indices. Align the arrays using take
    array[inds] = np.take(col_mean, inds[1])
    return array


## main code here

## main code here
for year in range(2010,2011):

    print('Resampling year:'+str(year))
    directoryname1='E:/datasets/20CRv2c/monolevel/'  
    filename1='uwnd.sig995.'+str(year)+'.nc' # arbitrary file from 20CRV2c for lat/long coords, 
    #but need to be aware of likely leap year problem 
    
    ncid1 =xr.open_dataset(os.path.join(directoryname1,filename1)) 
    
    
    latitude1=ncid1.lat.values  
    longitude1=ncid1.lon.values
    
    time1=ncid1.time.values
    
    query_pts=(latitude1[::-1],longitude1)  # reverse latitude array as need ascending values for interpolation
    
    #
    #
    directoryname='E:/datasets/CERA/CERA_years/CERA_mean/'  
    
    variable_name='u10'
    filename= 'CERA_mean_'+str(year)+'.nc'
    ncid2 =xr.open_dataset(os.path.join(directoryname,filename)) 
    
    
    
    time2=ncid2.time.values
    
    
    latitude2=ncid2.latitude.values  # need strictly ascending values for interpolation
    
    longitude2=ncid2.longitude.values
    
    sample_pts=(latitude2[::-1],longitude2) # reverse latitude array as need ascending values for interpolation
    
    
    
    t0=datetime.datetime.now()
    
    output=resample_2d(np.squeeze(ncid2.u10.values[0,:,:]), sample_pts, query_pts)
    for i in range(1,365):
        tmp_array=fill_nan_in_array(np.squeeze(ncid2.u10.values[i,:,:]))
        if(np.sum(np.isnan(tmp_array))):
            print('Broken')
        else:
            tmp=resample_2d(tmp_array, sample_pts, query_pts)
        output=np.concatenate((output,tmp))
    output=np.reshape(output,(365,latitude1.shape[0],longitude1.shape[0]))
    t1=datetime.datetime.now()
    print(t1-t0)
        
    resampled_array=create_resampled_xarray(time1[0:365],latitude1,longitude1,variable_name,output)
        
    filename= variable_name+'_resampledu_'+str(year)+'.nc'
    
    resampled_array.to_netcdf(directoryname+filename)








