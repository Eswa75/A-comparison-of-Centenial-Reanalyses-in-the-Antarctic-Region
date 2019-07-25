# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 22:10:53 2018

@author: Luke
"""

import numpy as np
import xarray as xr
#from netCDF4 import num2date
import os  # import os commands for making paths

def calculate_bmu(test_array, input_array):   

    # use euclidean distance to calculate the BMU

    test_array_tile = np.tile(test_array, (input_array.shape[0], 1))
    return  np.sqrt(np.nansum(np.square(input_array-test_array_tile),axis=1))
    

def read_u(year_array):
    # this function reads in 10m zonal (east-west) velocity from ERA-Interim dataset
    directoryname='E:/datasets/ERA20C/'
    variable_name='u10'
    filename=variable_name+'_resampledu_'+str(year_array[0])+'.nc'
    print('Reading year:'+str(year_array[0]))
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename)) 
    for year in year_array[1:]:
        print('Reading year:'+str(year))
        filename=variable_name+'_resampledu_'+str(year)+'.nc'
        ncid2 =xr.open_dataset(os.path.join(directoryname,filename)) 
        ncid1=xr.merge((ncid1,ncid2))

    # calculate day of year climatology 
    climatology = ncid1.u10.groupby('time.dayofyear').mean('time')
    # calculate the anomaly from the 20 year climatology
    anomalies = ncid1.u10.groupby('time.dayofyear') - climatology
    # output the raw xarray (ncid1) and anomaly patterns and the climatology
    return anomalies,climatology,ncid1


def read_v(year_array):
    # this function reads in 10m meridional (north-south) velocity from ERA-Interim dataset
    directoryname='E:/datasets/ERA20C/'  
    variable_name='v10'
    print('Reading year:'+str(year_array[0]))
    filename=variable_name+'_resampledu_'+str(year_array[0])+'.nc'  
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename)) 
    for year in year_array[1:]:
        print('Reading year:'+str(year))
        filename=variable_name+'_resampledu_'+str(year)+'.nc'
        ncid2 =xr.open_dataset(os.path.join(directoryname,filename)) 
        ncid1=xr.merge((ncid1,ncid2))

    # calculate day of year climatology 
    climatology = ncid1.v10.groupby('time.dayofyear').mean('time')
    # calculate the anomaly from the 20 year climatology
    anomalies = ncid1.v10.groupby('time.dayofyear') - climatology
    # output the raw xarray (ncid1) and anomaly patterns and the climatology
    return anomalies,climatology,ncid1

def create_multi_xarray_time_series(time,variable_name1,variable1,variable_name2,variable2):
# create multiple variables in an xarray object 
    #The dictionnary keys are the variables contained in the Dataset.
#The Dictionnary values are tuples, with first the (or the list of) dimension(s) over which the array varies, then the array itself
    d = {}
    d['time'] = ('time',time)
    d[variable_name1] = (['time'], variable1)
    d[variable_name2] = (['time'], variable2)
    dset = xr.Dataset(d)
    return dset

def create_xarray_histogram_series(time,node_numebr,variable_name,variable):
# create xarray histogram data
    d = {}
    d['time'] = ('time',time)
    d['node_number'] = ('node_number',node_number)
    d[variable_name] = (['time','node_number'], variable)
    dset = xr.Dataset(d)
    return dset

def create_year_histogram(node_dataset,som_number,year_array):
    # create istogram data for the year
    histogram2=np.zeros(1)
    for year in year_array:
        time_slice1=str(year)+'-01-01'
        time_slice2=str(year)+'-12-31'
        selection=node_dataset.sel(time=slice(time_slice1,time_slice2))
        histogram1=np.histogram(selection.node_number.values,som_number+1)
        histogram2=np.concatenate((histogram2,histogram1[0]),axis=0)        
        time_array=str(year)+'-06-01'
    histogram2=np.reshape(histogram2[1:],(year_array.shape[0],som_number+1))
    variable_name='histogram2'
    histogram_xarray=create_xarray_histogram_series(time_array,np.arange(0,18),variable_name,histogram2)
    return histogram_xarray



# start of main program
    

    
    

for period in range(6):
    if period == 0:
        year_array=np.arange(1900,1919+1) # identify time period for analysis
    elif period == 1:
        year_array=np.arange(1920,1939+1)
    elif period == 2:
        year_array=np.arange(1940,1959+1)
    elif period == 3:
        year_array=np.arange(1960,1979+1) # identify time period for analysis
    elif period == 4:
        year_array=np.arange(1980,1999+1) # identify time period for analysis    
    else:
        year_array=np.arange(2000,2011) # identify time period for analysis
        
    # read in ERA-INterim data
    anomalies_u,climatology_u,xarray_u=read_u(year_array)
    anomalies_v,climatology_v,xarray_v=read_v(year_array)

    year_climatology_u=np.mean(climatology_u,axis=0)
    year_climatology_v=np.mean(climatology_v,axis=0)
    
    # read the topographic map in asociated with the region
    directoryname2='E:/datasets/20CRv2c/monolevel/'  
    topography_filename='topography.nc'
    topography_data=xr.open_dataset(os.path.join(directoryname2,topography_filename)) 


    # define the latitude and longitude region of the map you wish to consider
    # this is hardwired at the moment, but can be changed to represent things around the peninsula
    # must be same as in the RISES7 python code
    lat_index=76
    lon_index1=139 
    lon_index2=158
    # creating numpy subset for u    
    u_subset=anomalies_u.values[:,lat_index:,lon_index1:lon_index2]  # -50S degree subset
    # creating numpy subset for u    
    v_subset=anomalies_v.values[:,lat_index:,lon_index1:lon_index2]  # -50S degree subset



    # reshaping numpy matrix so that it can be compared correctly 
    u_subset=np.reshape(u_subset,(u_subset.shape[0],u_subset.shape[1]*u_subset.shape[2]))
    v_subset=np.reshape(v_subset,(v_subset.shape[0],v_subset.shape[1]*v_subset.shape[2]))


 # read SOM patterns from RISES7 code
    directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
    variable_name='variable_u'
    filename=variable_name+'_RISES7.nc'
    som_u=xr.open_dataset(os.path.join(directoryname,filename)) 
    
    directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
    variable_name='variable_v'
    filename=variable_name+'_RISES7.nc'
    som_v=xr.open_dataset(os.path.join(directoryname,filename)) 


    # calucalte best matching units - i.e. SOM pattern that best matches the data for a particular day 
    column=0
    bmu_u=np.ones((som_u.variable_u.shape[0],u_subset.shape[0]))*np.nan
    bmu_v=np.ones((som_u.variable_u.shape[0],u_subset.shape[0]))*np.nan
    for column in som_u.SOM_node:   
        bmu_u[column,:]=calculate_bmu(som_u.variable_u[column,:], u_subset)
        bmu_v[column,:]=calculate_bmu(som_v.variable_v[column,:], v_subset)


    # reformatting data
    node_number=np.argmin(bmu_u+bmu_v,axis=0)
    euclidean=np.min(bmu_u+bmu_v,axis=0)
    time1=xarray_u.time.values
    variable_name1='node_number'
    variable_name2='euclidean'
    # creating datset in xarray format
    node_dataset=create_multi_xarray_time_series(time1,variable_name1,np.reshape(node_number,(node_number.shape[0],)),variable_name2,np.reshape(euclidean,(euclidean.shape[0],)))




    # writing xarray to a netcdf file.
    directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
    filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    node_dataset.to_netcdf(directoryname+filename)  
    print(period)
   
    
        





