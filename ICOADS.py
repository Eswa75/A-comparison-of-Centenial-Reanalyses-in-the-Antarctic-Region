# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:41:57 2019

@author: Luke
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os  # import os commands for making paths
import datetime as datetime
import seaborn as sns
import time
import statistics as st

# Loads in ICOADS mothly total SLP observation number data
directoryname = 'C:/Users/Luke/Desktop/Antarctic Research/Programs'
filename = 'slp.nobs.nc'
ncid = xr.open_dataset(os.path.join(directoryname,filename))
slp_obs = ncid.slp


#Creating a DataArray for each year's total value
times = np.arange(1900,1990)
data = np.zeros(len(times))
data_AP2 = np.zeros(len(times))
data_RS = np.zeros(len(times))
total_obs = xr.DataArray(data, coords=[times], dims=['time'])
total_obs_AP2 = xr.DataArray(data_AP2, coords=[times], dims=['time'])
total_obs_RS = xr.DataArray(data_RS, coords=[times], dims=['time'])


#Selects Antarctic Peninsula Region 1 and sorts data into a yearly format
Ob_count = 0 
year_array = times
for year in year_array:
    year_slice = slice(str(year)+'-01-01',str(year)+'-12-31')
    year_obs = slp_obs.sel(time=year_slice)
    for month in range(0,11):
        for lat in range(76,90):
            for lon in range(139,159):
                if year_obs[month,lat,lon] > 0:
                    Ob_count = Ob_count + year_obs[month,lat,lon]
    total_obs.loc[dict(time=year)] = np.log(Ob_count)
    Ob_count = 0 
    print("Currently on : " + str(year))               
           
      
#Selects Antarctic Peninsula Region 2 and sorts data into a yearly format            
Ob_count = 0 
year_array = times
for year in year_array:
    year_slice = slice(str(year)+'-01-01',str(year)+'-12-31')
    year_obs = slp_obs.sel(time=year_slice)
    for month in range(0,11):
        for lat in range(73,90):
            for lon in range(141,157):
                if year_obs[month,lat,lon] > 0:
                    Ob_count = Ob_count + year_obs[month,lat,lon]
    total_obs_AP2.loc[dict(time=year)] = np.log(Ob_count)
    Ob_count = 0 
    print("Currently on : " + str(year))     
    
    
    
#Selects Antarctic Ross Sea and sorts data into a yearly format    
Ob_count = 0 
year_array = times
for year in year_array:
    year_slice = slice(str(year)+'-01-01',str(year)+'-12-31')
    year_obs = slp_obs.sel(time=year_slice)
    for month in range(0,11):
        for lat in range(75,90):
            for lon in range(70,111):
                if year_obs[month,lat,lon] > 0:
                    Ob_count = Ob_count + year_obs[month,lat,lon]
    total_obs_RS.loc[dict(time=year)] = np.log(Ob_count)
    Ob_count = 0 
    print("Currently on : " + str(year))   
    
    
    
    
    
plt.bar(total_obs_RS.time,total_obs_RS,color='orange',label='RS Region SLP Observations')
plt.bar(total_obs_AP2.time,total_obs_AP2,color='cyan',label = 'AP2 Region SLP Observations')
plt.bar(total_obs.time,total_obs,color='magenta',label='AP1 Region SLP Observations')

ax = plt.gca()
ax.set_facecolor('xkcd:light grey')
plt.xlabel('year')
plt.ylabel('Number of Observations')
plt.legend(loc='upper left',fontsize='medium')
plt.savefig("Surface Based Observations",dpi=600)