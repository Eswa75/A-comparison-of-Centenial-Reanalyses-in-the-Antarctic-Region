# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:35:49 2019

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
import csv

'''
SOM32
'''
directoryname='C:/Users/Luke/Desktop/Antarctic Research/Final Results/Entropy_Coefficents/SOM32' 
filename='ERA20C_Permutations.nc'
ERA2032 =xr.open_dataset(os.path.join(directoryname,filename))
filename='CERA_Permutations.nc'
CERA32 =xr.open_dataset(os.path.join(directoryname,filename))


'''
SOM43
'''

directoryname='C:/Users/Luke/Desktop/Antarctic Research/Final Results/Entropy_Coefficents/SOM43' 
filename='ERA20C_Permutations.nc'
ERA2043 =xr.open_dataset(os.path.join(directoryname,filename))
filename='CERA_Permutations.nc'
CERA43 =xr.open_dataset(os.path.join(directoryname,filename))

'''
SOM86
'''

directoryname='C:/Users/Luke/Desktop/Antarctic Research/Final Results/Entropy_Coefficents/SOM86' 
filename='ERA20C_Permutations.nc'
ERA2086 =xr.open_dataset(os.path.join(directoryname,filename))
filename='CERA_Permutations.nc'
CERA86 =xr.open_dataset(os.path.join(directoryname,filename))

''' 
Let the ploting begin
'''

colors = cm.rainbow(np.linspace(0, 1, 15))
font = {'family' : 'DejaVu Sans',
        'weight' : 'medium',
        'size'   : 14}

plt.rc('font', **font)

plt.figure(figsize=(20,25))

'''
CERA32
'''
plt.subplot(4,2,1)
plt.plot(CERA32.time,CERA32.MERRA2,color='orange',marker="^",label="CERA vs. MERRA2",linewidth=3,markersize=7)
plt.plot(CERA32.time,CERA32.JRA55,color='green',marker="^",label="CERA vs. JRA55",linewidth=3,markersize=7)
plt.plot(CERA32.time,CERA32.ERA_Interim,color='red',marker="^",label="CERA vs. ERA-I",linewidth=3,markersize=7)
plt.plot(CERA32.time,CERA32.ncep,color=colors[12,:],marker="^",label="CERA vs. NCEP_NCAR2",linewidth=3,markersize=7)
plt.plot(CERA32.time,CERA32.ERA20C,color='magenta',marker="^",label="CERA vs. ERA20C",linewidth=3,markersize=7)
plt.plot(CERA32.time,CERA32.CR20,color=colors[4,:],marker="^",label="CERA vs. 20CR",linewidth=3,markersize=7)
plt.plot(CERA32.time,CERA32.ERA5,color='black',marker="^",label="CERA vs. ERA5",linewidth=3,markersize=7)


plt.title('(a) Entropy Coefficient: CERA, SOM = 3x2')
plt.xlabel('Year')
plt.ylabel('Entropy Coefficient')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()


'''
CERA43
'''

plt.subplot(4,2,3)
plt.plot(CERA43.time,CERA43.MERRA2,color='orange',marker="^",label="CERA vs. MERRA2",linewidth=3,markersize=7)
plt.plot(CERA43.time,CERA43.JRA55,color='green',marker="^",label="CERA vs. JRA55",linewidth=3,markersize=7)
plt.plot(CERA43.time,CERA43.ERA_Interim,color='red',marker="^",label="CERA vs. ERA-I",linewidth=3,markersize=7)
plt.plot(CERA43.time,CERA43.ncep,color=colors[12,:],marker="^",label="CERA vs. NCEP_NCAR2",linewidth=3,markersize=7)
plt.plot(CERA43.time,CERA43.ERA20C,color='magenta',marker="^",label="CERA vs. ERA20C",linewidth=3,markersize=7)
plt.plot(CERA43.time,CERA43.CR20,color=colors[4,:],marker="^",label="CERA vs. 20CR",linewidth=3,markersize=7)
plt.plot(CERA43.time,CERA43.ERA5,color='black',marker="^",label="CERA vs. ERA5",linewidth=3,markersize=7)


plt.title('(b) Entropy Coefficient: CERA, SOM = 4x3')
plt.xlabel('Year')
plt.ylabel('Entropy Coefficient ')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()


'''
CERA86
'''

plt.subplot(4,2,5)
plt.plot(CERA86.time,CERA86.MERRA2,color='orange',marker="^",label="CERA vs. MERRA2",linewidth=3,markersize=7)
plt.plot(CERA86.time,CERA86.JRA55,color='green',marker="^",label="CERA vs. JRA55",linewidth=3,markersize=7)
plt.plot(CERA86.time,CERA86.ERA_Interim,color='red',marker="^",label="CERA vs. ERA-I",linewidth=3,markersize=7)
plt.plot(CERA86.time,CERA86.ncep,color=colors[12,:],marker="^",label="CERA vs. NCEP_NCAR2",linewidth=3,markersize=7)
plt.plot(CERA86.time,CERA86.ERA20C,color='magenta',marker="^",label="CERA vs. ERA20C",linewidth=3,markersize=7)
plt.plot(CERA86.time,CERA86.CR20,color=colors[4,:],marker="^",label="CERA vs. 20CR",linewidth=3,markersize=7)
plt.plot(CERA86.time,CERA86.ERA5,color='black',marker="^",label="CERA vs. ERA5",linewidth=3,markersize=7)


plt.title('(c) Entropy Coefficient: CERA, SOM = 8x6')
plt.xlabel('Year')
plt.ylabel('Entropy Coefficient')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()



'''
ERA20C 32
'''

plt.subplot(4,2,2)
plt.plot(ERA2032.time,ERA2032.MERRA2,color='orange',marker="^",label="CERA vs. MERRA2",linewidth=3,markersize=7)
plt.plot(ERA2032.time,ERA2032.JRA55,color='green',marker="^",label="CERA vs. JRA55",linewidth=3,markersize=7)
plt.plot(ERA2032.time,ERA2032.ERA_Interim,color='red',marker="^",label="CERA vs. ERA-I",linewidth=3,markersize=7)
plt.plot(ERA2032.time,ERA2032.ncep,color=colors[12,:],marker="^",label="CERA vs. NCEP_NCAR2",linewidth=3,markersize=7)
plt.plot(ERA2032.time,ERA2032.CERA,color='magenta',marker="^",label="CERA vs. ERA20C",linewidth=3,markersize=7)
plt.plot(ERA2032.time,ERA2032.CR20,color=colors[1,:],marker="^",label="CERA vs. 20CR",linewidth=3,markersize=7)
plt.plot(ERA2032.time,ERA2032.ERA5,color='black',marker="^",label="CERA vs. ERA5",linewidth=3,markersize=7)

plt.title('(d) Entropy Coefficient: ERA20C, SOM = 3x2')
plt.xlabel('Year')
plt.ylabel('Entropy Coefficient')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()

'''
ERA20C 43
'''

plt.subplot(4,2,4)
plt.plot(ERA2043.time,ERA2043.MERRA2,color='orange',marker="^",label="CERA vs. MERRA2",linewidth=3,markersize=7)
plt.plot(ERA2043.time,ERA2043.JRA55,color='green',marker="^",label="CERA vs. JRA55",linewidth=3,markersize=7)
plt.plot(ERA2043.time,ERA2043.ERA_Interim,color='red',marker="^",label="CERA vs. ERA-I",linewidth=3,markersize=7)
plt.plot(ERA2043.time,ERA2043.ncep,color=colors[12,:],marker="^",label="CERA vs. NCEP_NCAR2",linewidth=3,markersize=7)
plt.plot(ERA2043.time,ERA2043.CERA,color='magenta',marker="^",label="CERA vs. ERA20C",linewidth=3,markersize=7)
plt.plot(ERA2043.time,ERA2043.CR20,color=colors[1,:],marker="^",label="CERA vs. 20CR",linewidth=3,markersize=7)
plt.plot(ERA2043.time,ERA2043.ERA5,color='black',marker="^",label="CERA vs. ERA5",linewidth=3,markersize=7)

plt.title('(e) Entropy Coefficient: ERA20C, SOM = 4x3 ')
plt.xlabel('Year')
plt.ylabel('Entropy Coefficient')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()

'''
ERA20C 86
'''

plt.subplot(4,2,6)
plt.plot(ERA2086.time,ERA2086.MERRA2,color='orange',marker="^",label="CERA vs. MERRA2",linewidth=3,markersize=7)
plt.plot(ERA2086.time,ERA2086.JRA55,color='green',marker="^",label="CERA vs. JRA55",linewidth=3,markersize=7)
plt.plot(ERA2086.time,ERA2086.ERA_Interim,color='red',marker="^",label="CERA vs. ERA-I",linewidth=3,markersize=7)
plt.plot(ERA2086.time,ERA2086.ncep,color=colors[12,:],marker="^",label="CERA vs. NCEP_NCAR2",linewidth=3,markersize=7)
plt.plot(ERA2086.time,ERA2086.CERA,color='magenta',marker="^",label="CERA vs. ERA20C",linewidth=3,markersize=7)
plt.plot(ERA2086.time,ERA2086.CR20,color=colors[1,:],marker="^",label="CERA vs. 20CR",linewidth=3,markersize=7)
plt.plot(ERA2086.time,ERA2086.ERA5,color='black',marker="^",label="CERA vs. ERA5",linewidth=3,markersize=7)

plt.title('(f) Entropy Coefficient: ERA20C, SOM = 8x6')
plt.xlabel('Year')
plt.ylabel('Entropy Coefficient')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()

plt.savefig('Entropy_SOM_Comparison')





