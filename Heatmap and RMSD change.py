# -*- coding: utf-8 -*-
"""
Euclidean_plot_ts2.py

REad in all the euclidean distance based time series of Best MAtcing Unit and EUclidean ditance from 
ERA-INterim, MERRA2, jRA55 and 20th Century Version 2c. Then make some nice time series diagrams and
a histgram plot


@authors: ajm226, lhc35

"""

#these are the packages that are used in this code:
# namely numpy, pandas, xarray, matplotlib, os, datetime and seaborn
# you will likely have to do install pandas, xarray and seaborn


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os  # import os commands for making paths
import datetime as datetime
import seaborn as sns

def create_xarray_histogram_series(time,node_number,variable_name,variable):
#The dictionnary keys are the variables contained in the Dataset.
#The Dictionnary values are tuples, with first the (or the list of) dimension(s) over which the array varies, then the array itself
    d = {}
    d['time'] = ('time',time)
    d['node_number'] = ('node_number',node_number)
    d[variable_name] = (['time','node_number'], variable)    
    dset = xr.Dataset(d) 
    return dset


def create_year_histogram(node_dataset,som_number,year_array):
    
    # create histogram for yearly slices based on som number
    histogram2=np.zeros(1)
    for year in year_array:
        time_slice1=str(year)+'-01-01'
        time_slice2=str(year)+'-12-31'
        selection=node_dataset.sel(time=slice(time_slice1,time_slice2))
        histogram1=np.histogram(selection.node_number.values,som_number)
        histogram2=np.concatenate((histogram2,histogram1[0]),axis=0)


# create time array
    start_year=str(year_array[0])+'-01-01'
    end_year =str(year_array[-1]+1)+'-01-01'
    time_array=pd.date_range(start_year,end_year,freq='BAS-JUL')
    
    

# create output xarray for graphing
    histogram2=np.reshape(histogram2[1:],(year_array.shape[0],som_number))
    histogram2_mean=np.mean(histogram2,axis=0)
    Relative_frequency=histogram2/histogram2_mean
    histogram2_RFO=np.log(Relative_frequency)
    variable_name='histogram2'
    histogram_xarray=create_xarray_histogram_series(time_array,np.arange(0,som_number),variable_name,histogram2_RFO)
    return histogram_xarray,histogram2_mean



som_shape=[4,3]


# merge the two MERRA2 datasets
year_array=np.arange(1980,1999+1)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_MERRA2-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(2000,2011)
filename='Node_dataset_MERRA2-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1980,2011)
MERRA2_histogram2,MERRA2_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)  



# merge three 20CRV2c datasets

year_array=np.arange(1900,1919+1)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_20CRV2c-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(1920,1939+1)
filename='Node_dataset_20CRv2c-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1940,1959+1)
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
Twenty_histogram2,Twenty_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)  


# merge two ECMWF datasets
year_array=np.arange(1980,1999+1)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(2000,2011)
filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1980,2011)
ECMWF_histogram2,ECMWF_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)  


# merge two ERA5 datasets
year_array=np.arange(1980,1999+1)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_ERA5-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(2000,2011)
filename='Node_dataset_ERA5-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1980,2011)
ERA5_histogram2,ERA5_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)

# merge two NCEP/NCAR2 datasets
year_array=np.arange(1980,1999+1)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
filename='Node_dataset_NCEP_NCAR2-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(2000,2011)
filename='Node_dataset_NCEP_NCAR2-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1980,2011)
NCEP_histogram2,NCEP_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)


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
JRA55_histogram2,JRA55_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)  

# merge the three CERA datasets
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
year_array=np.arange(1980,1999+1)
filename='Node_dataset_CERA-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(2000,2011)
filename='Node_dataset_CERA-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1902,2011)
CERA_histogram2,CERA_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)

# merge the three ERA20C datasets
year_array=np.arange(1900,1919+1)
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'
filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
year_array=np.arange(1920,1939+1)
filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1940,1959+1)
filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
ncid1=xr.merge((ncid1,ncid2))
year_array=np.arange(1960,1979+1)
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
ERA20C_histogram2,ERA20C_mean=create_year_histogram(ncid1,som_shape[0]*som_shape[1],year_array)



plt.figure(figsize=(10,8)) #smaller than A4 which would be 210 297
iter=0
letters = 'abcdefghijkl'
for i in range(0,som_shape[0]):
    for j in range(0,som_shape[1]):
        ax1 = plt.subplot(som_shape[0],som_shape[1],iter+1)
        plt.plot(ECMWF_histogram2.time.values,ECMWF_histogram2.histogram2.values[:,iter],linewidth=2.5,color='black')
        plt.plot(Twenty_histogram2.time.values,(Twenty_histogram2.histogram2.values[:,iter]),linewidth=2.5,color='slateblue')
        plt.plot(MERRA2_histogram2.time.values,(MERRA2_histogram2.histogram2.values[:,iter]),linewidth=2.5,)
        plt.plot(JRA55_histogram2.time.values,(JRA55_histogram2.histogram2.values[:,iter]),linewidth=2.5)
        plt.plot(CERA_histogram2.time.values,(CERA_histogram2.histogram2.values[:,iter]),linewidth=2.5,color='palegreen')
        plt.plot(ERA20C_histogram2.time.values,(ERA20C_histogram2.histogram2.values[:,iter]),linewidth=2.5,color='steelblue')
        plt.plot(ERA5_histogram2.time.values,(ERA5_histogram2.histogram2.values[:,iter]),linewidth=2.5)
        plt.plot(NCEP_histogram2.time.values,(NCEP_histogram2.histogram2.values[:,iter]),linewidth=2.5)
        plt.xticks([datetime.datetime(1900,1,1),datetime.datetime(1920,1,1),datetime.datetime(1940,1,1),datetime.datetime(1960,1,1),
        datetime.datetime(1980,1,1),datetime.datetime(2000,1,1)])
        plt.ylim((-2,2))
        plt.xlim((datetime.datetime(1900,1,1),datetime.datetime(2015,1,1)))
        plt.grid(linestyle=':',linewidth=2)
        if(i<som_shape[0]-1):
            ax1.tick_params(labelbottom='off')
        else:
            ax1.tick_params(labelbottom='on')
            plt.xlabel('Year')
        if((j==0)):
            ax1.tick_params(labelleft='on')
            if (i%2==0):
                plt.ylabel(r'$\Delta ln\left(RFO/\overline{RFO}\right)$',fontsize='x-large')
        else:
            ax1.tick_params(labelleft='off')
        ax1.annotate('('+letters[iter]+')'+' Node ' + str(iter+1), xy=(0.01, 0.85), xycoords="axes fraction",fontsize='x-large')
        iter=iter+1
        
        
plt.subplots_adjust(hspace=0.3)      
plt.savefig('EP_ts2_RISES7_figure1.png',dpi=1200)        



# create a second diagram showing the average patterns of teh relative frequency of occurrence for every reanalyses
plt.figure(figsize=(8,6))  #smaller than A4 which would be 210 297
plt.subplot(2,4,1)
cmap=sns.cubehelix_palette(start=0.1,as_cmap=True)
g=sns.heatmap(np.reshape((ECMWF_mean/3.6524),(som_shape[0],som_shape[1])), annot=True, fmt="3.1f", linewidths=.5,cbar=False,cmap=cmap)
g.set(xticks=[],yticks=[],title='(a) ERA-Interim',)
plt.subplot(2,4,2)
g=sns.heatmap(np.reshape((MERRA2_mean/3.652425),(som_shape[0],som_shape[1])), annot=True, fmt="3.1f", linewidths=.5,cbar=False,cmap=cmap)
g.set(xticks=[],yticks=[],title='(b) MERRA2')
plt.subplot(2,4,3)
g=sns.heatmap(np.reshape((Twenty_mean/3.652425),(som_shape[0],som_shape[1])), annot=True, fmt="3.1f", linewidths=.5,cbar=False,cmap=cmap)
g.set(xticks=[],yticks=[],title='(c) 20CRV2c')
plt.subplot(2,4,4)
g=sns.heatmap(np.reshape((CERA_mean/3.652425),(som_shape[0],som_shape[1])), annot=True, fmt="3.1f", linewidths=.5,cbar=False,cmap=cmap)
g.set(xticks=[],yticks=[],title='(d) CERA')
plt.subplot(2,4,5)
g=sns.heatmap(np.reshape((ERA20C_mean/3.652425),(som_shape[0],som_shape[1])), annot=True, fmt="3.1f", linewidths=.5,cbar=False,cmap=cmap)
g.set(xticks=[],yticks=[],title='(e) ERA20C')
plt.subplot(2,4,6)
g=sns.heatmap(np.reshape((ERA5_mean/3.652425),(som_shape[0],som_shape[1])), annot=True, fmt="3.1f", linewidths=.5,cbar=False,cmap=cmap)
g.set(xticks=[],yticks=[],title='(f) ERA5')
plt.subplot(2,4,7)
g=sns.heatmap(np.reshape((NCEP_mean/3.652425),(som_shape[0],som_shape[1])), annot=True, fmt="3.1f", linewidths=.5,cbar=False,cmap=cmap)
g.set(xticks=[],yticks=[],title='(g) NCEP')
plt.subplot(2,4,8)
g=sns.heatmap(np.reshape((JRA55_mean/3.652425),(som_shape[0],som_shape[1])), annot=True, fmt="3.1f", linewidths=.5,cbar=False,cmap=cmap)
g.set(xticks=[],yticks=[],title='(h) JRA55')
plt.savefig('EP_ts2_RISES7_figure2.png',dpi=600)     

