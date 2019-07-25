# -*- coding: utf-8 -*-
"""
RMSD_plot1.py

REad in all the  time series of BMU (best matching unit) and EUclidean ditance from 
ERA-INterim, MERRA2, JRA55 and 20th Century Version 2c. 
Then make a time series diagram

Created on Tue Jun 12 12:08:22 2018
*eddited on 16/08/2018 @ 19:32:10

@authors: ajm226, lhc35
"""

#these are the packages that are used in this code:
# namely numpy, pandas, xarray, matplotlib, os and datetime
# you will likely have to do install of pandas and xarray
import numpy as np
import pandas as pd
import xarray as xr
#from netCDF4 import num2date
import matplotlib.pyplot as plt
import os  # import os commands for making paths
import datetime as datetime

def create_xarray_histogram_series(time,node_number,variable_name,variable):
#this function makes an xarray object which is made up of the time and the node numbers
    d = {}
    d['time'] = ('time',time)
    d['node_number'] = ('node_number',node_number)
    d[variable_name] = (['time','node_number'], variable)    
    dset = xr.Dataset(d) 
    return dset


def create_year_histogram(node_dataset,som_number,year_array):
# create histogram for yearly slices based on the node number
    histogram2=np.zeros(1)
    for year in year_array:
        time_slice1=str(year)+'-01-01'
        time_slice2=str(year+1)+'-01-01'
        selection=node_dataset.sel(time=slice(time_slice1,time_slice2))
        histogram1=np.histogram(selection.node_number.values,som_number)
        histogram2=np.concatenate((histogram2,histogram1[0]),axis=0)

# create time array
    start_year=str(year_array[0])+'-01-01'
    end_year =str(year_array[-1]+1)+'-01-01'
    time_array=pd.date_range(start_year,end_year,freq='BAS-JUL')

# create output xarray  with the mean pattern and the anomaly pattern for graphing
    histogram2=np.reshape(histogram2[1:],(year_array.shape[0],som_number))
    histogram2_mean=np.mean(histogram2,axis=0)
    histogram2_anom=histogram2-histogram2_mean
    histogram2_anom=(histogram2_anom/histogram2_mean)*100.0
    variable_name='histogram2'
    histogram_anom_xarray=create_xarray_histogram_series(time_array,np.arange(0,som_number),variable_name,histogram2_anom)
    histogram_xarray=create_xarray_histogram_series(time_array,np.arange(0,som_number),variable_name,histogram2)
    return histogram_anom_xarray,histogram_xarray,histogram2_mean

def RMSD(histogram2,histogram_mean):
        # histogram2 is a a 2d histogram xarray (number of years, versus node number)
        # we then calculate a time series of the Root mean square differeence as a function of time for output
        difference=(histogram2.histogram2-histogram_mean)/3.65
        squared_difference=xr.ufuncs.square(difference)
        mean_squared_difference=squared_difference.mean(axis=1)
        return np.sqrt(mean_squared_difference)   
        


# start of main program
        
    
som_shape=[4,3] # define the shape of teh self organizing map (a 4 by 3 grid in this case)


# merge two ECMWF datasets created by a previous program called RIS_EOF_SOM7_ECMWF
year_array=np.arange(1980,1999+1)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(2000,2011)
filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1980,2011)
ECMWF_anom_histogram2,ECMWF_histogram2,ECMWF_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)  
ECMWF_rmsd=RMSD(ECMWF_histogram2,ECMWF_mean)



# merge two ERA5 datasets created by a previous program called RIS_EOF_SOM7_ECMWF
year_array=np.arange(1980,1999+1)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_ERA5-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(2000,2011)
filename='Node_dataset_ERA5-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1980,2011)
ERA5_anom_histogram2,ERA5_histogram2,ERA5_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)  
ERA5_rmsd=RMSD(ERA5_histogram2,ECMWF_mean)

# merge the two MERRA2 datasets created by a previous program RIS_EOF_SOM7_ECMWF

year_array=np.arange(1980,1999+1)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_NCEP_NCAR2-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(2000,2011)
filename='Node_dataset_NCEP_NCAR2-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1980,2011)
NCEP_NCAR2_anom_histogram2,NCEP_NCAR2_histogram2,NCEP_NCAR2_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)  

NCEP_NCAR2_rmsd=RMSD(NCEP_NCAR2_histogram2,ECMWF_mean)

# merge the two MERRA2 datasets created by a previous program RIS_EOF_SOM7_ECMWF

year_array=np.arange(1980,1999+1)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_MERRA2-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(2000,2011)
filename='Node_dataset_MERRA2-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1980,2011)
MERRA2_anom_histogram2,MERRA2_histogram2,MERRA2_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)  

MERRA2_rmsd=RMSD(MERRA2_histogram2,ECMWF_mean)

# merge 20CRV2c datasets

year_array=np.arange(1900,1920)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_20CRV2c-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(1920,1940)
filename='Node_dataset_20CRv2c-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1940,1960)
filename='Node_dataset_20CRv2c-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1960,1979+1)
filename='Node_dataset_20CRv2c-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
filename='Node_dataset_20CRV2c-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1980,1999+1)
filename='Node_dataset_20CRv2c-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(2000,2011)
filename='Node_dataset_20CRv2c-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1900,2011)
Twenty_anom_histogram2,Twenty_histogram2,Twenty_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)  


Twenty_rmsd=RMSD(Twenty_histogram2,ECMWF_mean)
        
#merge CERA Datasets 

year_array=np.arange(1902,1919+1)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_CERA-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(1920,1939+1)
filename='Node_dataset_CERA-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1940,1959+1)
filename='Node_dataset_CERA-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1960,1979+1)
filename='Node_dataset_CERA-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
filename='Node_dataset_CERA-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1980,1999+1)
filename='Node_dataset_CERA-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(2000,2011)
filename='Node_dataset_CERA-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1902,2011)
CERA_anom_histogram2,CERA_histogram2,CERA_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)  


CERA_rmsd=RMSD(CERA_histogram2,ECMWF_mean)


#merge ERA20C Datasets 

year_array=np.arange(1900,1920)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(1920,1940)
filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1940,1960)
filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1960,1979+1)
filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1980,1999+1)
filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(2000,2011)
filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1900,2011)
ERA20C_anom_histogram2,ERA20C_histogram2,ERA20C_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)  


ERA20C_rmsd=RMSD(ERA20C_histogram2,ECMWF_mean)


# merge the three JRA55 datasets
year_array=np.arange(1958,1979+1)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_JRA55-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(1980,1999+1)
filename='Node_dataset_JRA55-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(2000,2011)
filename='Node_dataset_JRA55-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1958,2011)
JRA55_anom_histogram2,JRA55_histogram2,JRA55_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)  


JRA55_rmsd=RMSD(JRA55_histogram2,ECMWF_mean)

        
# create a nice plot
plt.figure(figsize=(10,4))  #smaller than A4 which would be 210 297
plt.plot(ECMWF_rmsd.time,ECMWF_rmsd,color='blue',label='ERA-Interim')
plt.plot(Twenty_rmsd.time,Twenty_rmsd,color='cyan',label='20CRV2c')
plt.plot(CERA_rmsd.time,CERA_rmsd,color='orange',label='CERA')
plt.plot(ERA20C_rmsd.time,ERA20C_rmsd,color='red',label='ERA20C')
#plt.plot(ERA5_rmsd.time,ERA5_rmsd,color='black',label='ERA5 ')
#plt.plot(NCEP_NCAR2_rmsd.time,NCEP_NCAR2_rmsd,color='green',label='NCEP_NCAR2')
#plt.plot(MERRA2_rmsd.time,MERRA2_rmsd,color='magenta',label='MERRA2')
#plt.plot(JRA55_rmsd.time,JRA55_rmsd,color='purple',label='JRA55')


plt.xlabel('Year')
plt.ylabel('RMSD (%)')
plt.grid()
plt.axis([datetime.datetime(1900,1,1), datetime.datetime(2011,1,1), 0 ,10])
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('RMSD_plot_AP2_43.png',dpi=600) 
    