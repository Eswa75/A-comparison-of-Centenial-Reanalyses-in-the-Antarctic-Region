# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 17:37:18 2019

@author: Luke
"""

# -*- codinE: utf-8 -*-
"""
Euclidean_plot_ts2.py

REad in all the euclidean distance based time series of Best MAtcing Unit and EUclidean ditance from 
ERA-INterim, ERA20C, MERRA2, jRA55 and 20th Century Version 2c. Then make some nice time series diagrams and
a histgram plot


@author: ajm226
"""

#these are the packages that are used in this code:
# namely numpy, pandas, xarray, matplotlib, os, datetime and seaborn
# you will likely have to do install pandas, xarray and seaborn


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

def read_ensemble_data(ensemble_no,directoryname):
    filename='Node_dataset_20CRV2c_ensemble_'+str(ensemble_no)+'_'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))    
    for year_span in np.arange(0,5):
        if(year_span==0):
            year_array=np.arange(1920,1939+1)
        elif(year_span==1):
            year_array=np.arange(1940,1959+1)            
        elif(year_span==2):
            year_array=np.arange(1960,1979+1)
        elif(year_span==3):
            year_array=np.arange(1980,1999+1)
        else:
            year_array=np.arange(2000,2011)
             # writing xarray to a netcdf file.
        filename='Node_dataset_20CRV2c_ensemble_'+str(ensemble_no)+'_'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
        print('READING' +filename)
        ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
        ncid1=xr.merge((ncid1,ncid2))
    return ncid1


def read_20CRV2c_data():
    directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
    year_array=np.arange(1900,1920)
    filename='Node_dataset_20CRV2c-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))    
    for year_span in np.arange(0,5):
        if(year_span==0):
            year_array=np.arange(1920,1939+1)
        elif(year_span==1):
            year_array=np.arange(1940,1959+1)            
        elif(year_span==2):
            year_array=np.arange(1960,1979+1)
        elif(year_span==3):
            year_array=np.arange(1980,1999+1)
        else:
            year_array=np.arange(2000,2011)
             # writing xarray to a netcdf file.
        filename='Node_dataset_20CRV2c-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
        print('READING' +filename)
        ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
        ncid1=xr.merge((ncid1,ncid2))
    return ncid1




def read_ERA20C_data():
    directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
    year_array=np.arange(1900,1919+1)
    filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))    
    for year_span in np.arange(0,5):
        if(year_span==0):
            year_array=np.arange(1920,1939+1)
        elif(year_span==1):
            year_array=np.arange(1940,1959+1)            
        elif(year_span==2):
            year_array=np.arange(1960,1979+1)
        elif(year_span==3):
            year_array=np.arange(1980,1999+1)
        else:
            year_array=np.arange(2000,2011)
             # writing xarray to a netcdf file.
        filename='Node_dataset_ERA20C-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
        print('READING' +filename)
        ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
        ncid1=xr.merge((ncid1,ncid2))
    return ncid1


def read_JRA55_data():
    directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
    year_array=np.arange(1958,1979+1)
    filename='Node_dataset_JRA55-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))    
    for year_span in np.arange(0,2):
        if(year_span==0):
            year_array=np.arange(1980,1999+1)
        else:
            year_array=np.arange(2000,2011)
             # writing xarray to a netcdf file.
        filename='Node_dataset_JRA55-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
        print('READING' +filename)
        ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
        ncid1=xr.merge((ncid1,ncid2))
    return ncid1


def read_MERRA2_data():
    directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
    year_array=np.arange(1980,1999+1)
    filename='Node_dataset_MERRA2-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))    
    year_array=np.arange(2000,2011)
    filename='Node_dataset_MERRA2-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
    ncid1=xr.merge((ncid1,ncid2))
    return ncid1


def read_NCEP_NCAR2_data():
    directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
    year_array=np.arange(1980,1999+1)
    filename='Node_dataset_NCEP_NCAR2-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))    
    year_array=np.arange(2000,2011)
    filename='Node_dataset_NCEP_NCAR2-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
    ncid1=xr.merge((ncid1,ncid2))
    return ncid1


def read_ECMWF_data():
    year_array=np.arange(1980,1999+1)
    directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
    filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))    
    year_array=np.arange(2000,2011)
    filename='Node_dataset_ECMWF-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
    ncid1=xr.merge((ncid1,ncid2))
    return ncid1

def read_ERA5_data():
    year_array=np.arange(1980,1999+1)
    directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/'  
    filename='Node_dataset_ERA5-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))    
    year_array=np.arange(2000,2011)
    filename='Node_dataset_ERA5-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
    ncid1=xr.merge((ncid1,ncid2))
    return ncid1



def read_CERA_data():
    directoryname='C:/Users/Luke/Desktop/Antarctic Research/Synoptic Patterns and Comparisons/' 
    year_array=np.arange(1902,1920)
    filename='Node_dataset_CERA-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
    print('READING' +filename)
    ncid1 =xr.open_dataset(os.path.join(directoryname,filename))    
    for year_span in np.arange(0,5):
        if(year_span==0):
            year_array=np.arange(1920,1939+1)
        elif(year_span==1):
            year_array=np.arange(1940,1959+1)            
        elif(year_span==2):
            year_array=np.arange(1960,1979+1)
        elif(year_span==3):
            year_array=np.arange(1980,1999+1)
        else:
            year_array=np.arange(2000,2011)
             # writing xarray to a netcdf file.
        filename='Node_dataset_CERA-'+str(year_array[0])+'-'+str(year_array[-1])+'_RISES7.nc'
        print('READING' +filename)
        ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
        ncid1=xr.merge((ncid1,ncid2))
    return ncid1



def contigency_entropy(x,y):  #x,y):

#Given a two-dimensional contingency table in the form of an integer array nn[i][j], where i
#labels the x variable and ranges from 1 to ni, j labels the y variable and ranges from 1 to nj,
#this routine returns the entropy h of the whole table, the entropy hx of the x distribution, the
#entropy hy of the y distribution, the entropy hygx of y given x, the entropy hxgy of x given y,
#the dependency uygx of y on x (eq. 14.4.15), the dependency uxgy of x on y (eq. 14.4.16),
#and the symmetrical dependency uxy (eq. 14.4.17).


    TINY=1.0E-06

    table=np.asarray(pd.crosstab(x,y))


    ni=table.shape[0]
    nj=table.shape[1]


    sumi=np.sum(table,1)
    sumj=np.sum(table,0)
    sum_total=np.sum(table)

# expectation and chi2 calculation
    expectation=np.zeros((ni,nj))

    for i in np.arange(0,sumi.shape[0]):   #% rows
        for j in np.arange(0,sumj.shape[0]):   #% columns
            expectation[i,j]=(sumi[i]*sumj[j])/sum_total
             
    chi2=np.sum( ( (table-expectation)**2.0)/expectation) 
    cramer=np.sqrt(chi2/(sum_total*min(ni-1,nj-1)))



    TINY=1.0E-06
# see section 14.4 in Numerical Recipes

    hx=0.0 #Entropy of the x distribution,
    for i in np.arange(0,ni):
        if (sumi[i]):
            p=sumi[i]/sum_total
            hx =hx -p*np.log(p)
            
        
    hy=0.0 #%Entropy of the y distribution.
    for j in np.arange(0,nj):
        if (sumj[j]):
            p=sumj[j]/sum_total
            hy =hy- p*np.log(p)
    
    h=0.0
    for i in np.arange(0,ni): # Total entropy: loop over both x and y
        for j in np.arange(0,nj):
            if (table[i,j]):
                p=table[i,j]/sum_total
                h =h- p*np.log(p)
                
    hygx=(h)-(hx) #Uses equation (14.4.18),
    hxgy=(h)-(hy) #as does this.
    uygx=(hy-hygx)/(hy+TINY) #Equation (14.4.15).
    uxgy=(hx-hxgy)/(hx+TINY) #%Equation (14.4.16).
    uxy=2.0*(hx+hy-h)/(hx+hy+TINY) #Equation (14.4.17).
    
    dependency=uxy
    

    return table,cramer,dependency


def create_xarray_association_series(time,variable_name1,variable1,variable_name2,variable2):
#The dictionnary keys are the variables contained in the Dataset.
#The Dictionnary values are tuples, with first the (or the list of) dimension(s) over which the array varies, then the array itself
    d = {}
    d['time'] = ('time',time)
    d[variable_name1] = ('time', variable1)
    d[variable_name2] = ('time', variable2)
    dset = xr.Dataset(d) 
    return dset



def create_year_association(node_number1,node_number2,year_array):
    
    # create histogram for yearly slices based on som number
    cramer=np.zeros(year_array.shape[0])
    dependency=np.zeros(year_array.shape[0])
    i=0
    for year in year_array:
        time_slice1=str(year)+'-01-01'
        time_slice2=str(year)+'-12-30'
        
        selection1=node_number1.sel(time=slice(time_slice1,time_slice2))
        selection2=node_number2.sel(time=slice(time_slice1,time_slice2))
        table,cramer[i],dependency[i]=contigency_entropy(selection1,selection2)
        i=i+1
        
# create time array
    start_year=str(year_array[0])+'-01-01'
    end_year =str(year_array[-1]+1)+'-01-01'
    time_array=pd.date_range(start_year,end_year,freq='BAS-JUL')

# create output xarray for graphing
    variable_name1='cramer'
    variable_name2='dependency'
    association_xarray=create_xarray_association_series(time_array,variable_name1,cramer,variable_name2,dependency)
    return association_xarray

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

year_array_ECMWF=np.arange(1980,2011)
ncid_ECMWF=read_ECMWF_data()


year_array_20CR=np.arange(1900,2011)
ncid_20CR=read_20CRV2c_data()


year_array_MERRA2=np.arange(1980,2011)
ncid_MERRA2=read_MERRA2_data()


year_array_JRA55=np.arange(1958,2011)
ncid_JRA55=read_JRA55_data()

year_array_ERA20C=np.arange(1900,2011)
ncid_ERA20C=read_ERA20C_data()

year_array_CERA=np.arange(1902,2011)
ncid_CERA=read_CERA_data()

year_array_NCEP=np.arange(1980,2011)
ncid_NCEP=read_NCEP_NCAR2_data()

year_array_ERA5=np.arange(1980,2011)
ncid_ERA5=read_ERA5_data()


#output=create_year_association(ncid_ECMWF.node_number,ncid_MERRA2.node_number,year_array_ECMWF)


#output=create_year_association(ncid_JRA55.node_number,ncid_20CR.node_number,year_array_JRA55)


colors = cm.rainbow(np.linspace(0, 1, 15))
font = {'family' : 'DejaVu Sans',
        'weight' : 'medium',
        'size'   : 14}

plt.rc('font', **font)

plt.figure(figsize=(20,25))  #smaller than A4 which would be 210 297



'''
20CR
'''

plt.subplot(4,2,1)

output=create_year_association(ncid_JRA55.node_number,ncid_20CR.node_number,year_array_JRA55)
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^",label="JRA55 vs. 20CRV2c",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='JRA55')
Twenty_cr_output = output

output=create_year_association(ncid_ECMWF.node_number,ncid_20CR.node_number,year_array_ECMWF)
plt.plot(output.time,output.dependency.values,color='red',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='red',marker="^",label="ERA-Interim vs. 20CRV2c",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA_Interim')
Twenty_cr_output = xr.merge([Twenty_cr_output,output])


output=create_year_association(ncid_MERRA2.node_number,ncid_20CR.node_number,year_array_MERRA2)
plt.plot(output.time,output.dependency.values,color='orange',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='orange',marker="^",label="20CRV2c vs. MERRA2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='MERRA2')
Twenty_cr_output = xr.merge([Twenty_cr_output,output])

output=create_year_association(ncid_ERA20C.node_number,ncid_20CR.node_number,year_array_ERA20C)
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^",label="ERA20C vs. 20CRV2c",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA20C')
Twenty_cr_output = xr.merge([Twenty_cr_output,output])

output=create_year_association(ncid_20CR.node_number,ncid_NCEP.node_number,year_array_NCEP)
plt.plot(output.time,output.dependency.values,color='purple',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='purple',marker="^",label="NCEP_NCAR2 vs. 20CR",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ncep')
Twenty_cr_output = xr.merge([Twenty_cr_output,output])

output=create_year_association(ncid_ERA5.node_number,ncid_20CR.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color='black',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='black',marker="^",label="ERA5 vs. 20CRV2c",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA5')
Twenty_cr_output = xr.merge([Twenty_cr_output,output])

output=create_year_association(ncid_CERA.node_number,ncid_20CR.node_number,year_array_CERA)
plt.plot(output.time,output.dependency.values,color='green',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='green',marker="^",label="CERA vs. 20CRV2c",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CERA')
Twenty_cr_output = xr.merge([Twenty_cr_output,output])


directoryname='C:/Users/Luke/Desktop/Antarctic Research/Final Results/Entropy_Coefficents/'
filename='CR20_Permutations.nc'
Twenty_cr_output.to_netcdf(directoryname+filename)


plt.title('(a) Entropy Coefficient: 20CR2vc')

plt.ylabel('Entropy Coefficient')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()


'''
ERA_Interim Permutations
'''

plt.subplot(4,2,2)


output=create_year_association(ncid_ECMWF.node_number,ncid_NCEP.node_number,year_array_NCEP)
plt.plot(output.time,output.dependency.values,color='purple',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='purple',marker="^",label="NCEP_NCAR2 vs. ERA-Interim",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ncep')
ERA_Interim_output = output


output=create_year_association(ncid_ERA20C.node_number,ncid_ECMWF.node_number,year_array_ECMWF)
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^",label="ERA20C vs. ERA-Interim",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA20C')
ERA_Interim_output = xr.merge([ERA_Interim_output,output])


output=create_year_association(ncid_ECMWF.node_number,ncid_20CR.node_number,year_array_ECMWF)
plt.plot(output.time,output.dependency.values,color='red',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='red',marker="^",label="ERA-Interim vs. 20CRV2c",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CR20')
ERA_Interim_output = xr.merge([ERA_Interim_output,output])


output=create_year_association(ncid_MERRA2.node_number,ncid_ECMWF.node_number,year_array_ECMWF)
plt.plot(output.time,output.dependency.values,color='orange',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='orange',marker="^",label="MERRA2 vs. ERA-Interim",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='MERRA2')
ERA_Interim_output = xr.merge([ERA_Interim_output,output])


output=create_year_association(ncid_JRA55.node_number,ncid_ECMWF.node_number,year_array_ECMWF)
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^",label="JRA55 vs. ERA-Interim",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='JRA55C')
ERA_Interim_output = xr.merge([ERA_Interim_output,output])


output=create_year_association(ncid_ERA5.node_number,ncid_ECMWF.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color='black',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='black',marker="^",label="ERA5 vs. ERA-Interim",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA5')
ERA_Interim_output = xr.merge([ERA_Interim_output,output])


output=create_year_association(ncid_CERA.node_number,ncid_ECMWF.node_number,year_array_ECMWF)
plt.plot(output.time,output.dependency.values,color='green',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='green',marker="^",label="CERA vs. ERA-Interim",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CERA')
ERA_Interim_output = xr.merge([ERA_Interim_output,output])


directoryname='C:/Users/Luke/Desktop/Antarctic Research/Final Results/Entropy_Coefficents/'
filename='ERA_Interim_Permutations.nc'
ERA_Interim_output.to_netcdf(directoryname+filename)



plt.title('(b) Entropy Coefficient: ERA-Interim')
plt.ylabel('Entropy Coefficient')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()


'''
JRA55 Permutations
'''

plt.subplot(4,2,3)

output=create_year_association(ncid_JRA55.node_number,ncid_ECMWF.node_number,year_array_ECMWF)
plt.plot(output.time,output.dependency.values,color='magenta',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='magenta',marker="^",label="JRA55 vs. ERA-Interim",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA_Interim')
JRA55_output = output


output=create_year_association(ncid_JRA55.node_number,ncid_20CR.node_number,year_array_JRA55)
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^",label="JRA55 vs. 20CRV2c",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CR20')
JRA55_output = xr.merge([JRA55_output,output])

output=create_year_association(ncid_ERA20C.node_number,ncid_JRA55.node_number,year_array_JRA55)
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^",label="ERA20C vs. JRA55",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA20C')
JRA55_output = xr.merge([JRA55_output,output])

output=create_year_association(ncid_JRA55.node_number,ncid_MERRA2.node_number,year_array_MERRA2)
plt.plot(output.time,output.dependency.values,color='orange',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='orange',marker="^",label="JRA55 vs. MERRA2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='MERRA2')
JRA55_output = xr.merge([JRA55_output,output])

output=create_year_association(ncid_JRA55.node_number,ncid_NCEP.node_number,year_array_NCEP)
plt.plot(output.time,output.dependency.values,color='purple',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='purple',marker="^",label="NCEP_NCAR2 vs. JRA55",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ncep')
JRA55_output = xr.merge([JRA55_output,output])


output=create_year_association(ncid_ERA5.node_number,ncid_JRA55.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color='black',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='black',marker="^",label="ERA5 vs. JRA55",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA5')
JRA55_output = xr.merge([JRA55_output,output])


output=create_year_association(ncid_CERA.node_number,ncid_JRA55.node_number,year_array_JRA55)
plt.plot(output.time,output.dependency.values,color='green',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='green',marker="^",label="CERA vs. JRA55",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CERA')
JRA55_output = xr.merge([JRA55_output,output])


directoryname='C:/Users/Luke/Desktop/Antarctic Research/Final Results/Entropy_Coefficents/'
filename='JRA55_Permutations.nc'
JRA55_output.to_netcdf(directoryname+filename) 

plt.title('(c) Entropy Coefficient: JRA55')
plt.ylabel('Entropy Coefficient')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()

'''
NCEP Permutations
'''

plt.subplot(4,2,4)

output=create_year_association(ncid_JRA55.node_number,ncid_NCEP.node_number,year_array_NCEP)
plt.plot(output.time,output.dependency.values,color='purple',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='purple',marker="^",label="NCEP_NCAR2 vs. JRA55",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='JRA55')
NCEP2_output = output

output=create_year_association(ncid_ECMWF.node_number,ncid_NCEP.node_number,year_array_NCEP)
plt.plot(output.time,output.dependency.values,color='red',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='red',marker="^",label="NCEP_NCAR2 vs. ERA-Interim",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA_Interim')
NCEP2_output = xr.merge([NCEP2_output,output])

output=create_year_association(ncid_20CR.node_number,ncid_NCEP.node_number,year_array_NCEP)
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^",label="NCEP_NCAR2 vs. 20CR",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CR20')
NCEP2_output = xr.merge([NCEP2_output,output])

output=create_year_association(ncid_ERA20C.node_number,ncid_NCEP.node_number,year_array_NCEP)
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^",label="NCEP_NCAR2 vs. ERA20C",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA20C')
NCEP2_output = xr.merge([NCEP2_output,output])

output=create_year_association(ncid_MERRA2.node_number,ncid_NCEP.node_number,year_array_NCEP)
plt.plot(output.time,output.dependency.values,color='orange',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='orange',marker="^",label="NCEP_NCAR2 vs. MERRA2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='MERRA2')
NCEP2_output = xr.merge([NCEP2_output,output])


output=create_year_association(ncid_ERA5.node_number,ncid_NCEP.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color='black',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='black',marker="^",label="NCEP_NCAR2 vs. ERA5",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA5')
NCEP2_output = xr.merge([NCEP2_output,output])


output=create_year_association(ncid_CERA.node_number,ncid_NCEP.node_number,year_array_NCEP)
plt.plot(output.time,output.dependency.values,color='green',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='green',marker="^",label="NCEP_NCAR2 vs. CERA",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CERA')
NCEP2_output = xr.merge([NCEP2_output,output])



directoryname='C:/Users/Luke/Desktop/Antarctic Research/Final Results/Entropy_Coefficents/'
filename='NCEP2_Permutations.nc'
NCEP2_output.to_netcdf(directoryname+filename) 

plt.title('(d) Entropy Coefficient: NCEP/NCAR2')
plt.ylabel('Entropy Coefficient')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()




'''
MERRA2 Permutations
'''


plt.subplot(4,2,5)

output=create_year_association(ncid_MERRA2.node_number,ncid_NCEP.node_number,year_array_NCEP)
plt.plot(output.time,output.dependency.values,color='purple',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='purple',marker="^",label="NCEP_NCAR2 vs. MERRA2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ncep')
MERRA2_output = output

output=create_year_association(ncid_JRA55.node_number,ncid_MERRA2.node_number,year_array_MERRA2)
plt.plot(output.time,output.dependency.values,color='orange',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='orange',marker="^",label="JRA55 vs. MERRA2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='JRA55')
MERRA2_output = xr.merge([MERRA2_output,output])

output=create_year_association(ncid_MERRA2.node_number,ncid_ECMWF.node_number,year_array_ECMWF)
plt.plot(output.time,output.dependency.values,color='red',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='red',marker="^",label="MERRA2 vs. ERA-Interim",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA_Interim')
MERRA2_output = xr.merge([MERRA2_output,output])


output=create_year_association(ncid_MERRA2.node_number,ncid_20CR.node_number,year_array_MERRA2)
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^",label="20CRV2c vs. MERRA2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CR20')
MERRA2_output = xr.merge([MERRA2_output,output])

output=create_year_association(ncid_ERA20C.node_number,ncid_MERRA2.node_number,year_array_MERRA2)
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^",label="ERA20C vs. MERRA2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA20C')
MERRA2_output = xr.merge([MERRA2_output,output])

output=create_year_association(ncid_ERA5.node_number,ncid_MERRA2.node_number,year_array_MERRA2)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color='black',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='black',marker="^",label="ERA5 vs. MERRA2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA5')
MERRA2_output = xr.merge([MERRA2_output,output])


output=create_year_association(ncid_CERA.node_number,ncid_MERRA2.node_number,year_array_MERRA2)
plt.plot(output.time,output.dependency.values,color='green',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='green',marker="^",label="CERA vs. MERRA2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CERA')
MERRA2_output = xr.merge([MERRA2_output,output])



directoryname='C:/Users/Luke/Desktop/Antarctic Research/Final Results/Entropy_Coefficents/'
filename='MERRA2_Permutations.nc'
MERRA2_output.to_netcdf(directoryname+filename) 

plt.title('(e) Entropy Coefficient: MERRA2')
plt.xlabel('Year')
plt.ylabel('Entropy Coefficient')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()





'''
ERA20C Permutations 
'''


plt.subplot(4,2,6)



output=create_year_association(ncid_ERA20C.node_number,ncid_MERRA2.node_number,year_array_MERRA2)
plt.plot(output.time,output.dependency.values,color='orange',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='orange',marker="^",label="ERA20C vs. MERRA2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='MERRA2')
ERA20C_output = output

output=create_year_association(ncid_ERA20C.node_number,ncid_NCEP.node_number,year_array_NCEP)
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^",label="NCEP_NCAR2 vs. ERA20C",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ncep')
ERA20C_output = xr.merge([ERA20C_output,output])

output=create_year_association(ncid_ERA20C.node_number,ncid_JRA55.node_number,year_array_JRA55)
plt.plot(output.time,output.dependency.values,color='green',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='green',marker="^",label="ERA20C vs. JRA55",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='JRA55')
ERA20C_output = xr.merge([ERA20C_output,output])

output=create_year_association(ncid_ERA20C.node_number,ncid_ECMWF.node_number,year_array_ECMWF)
plt.plot(output.time,output.dependency.values,color='red',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='red',marker="^",label="ERA20C vs. ERA-Interim",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA_Interim')
ERA20C_output = xr.merge([ERA20C_output,output])

output=create_year_association(ncid_ERA20C.node_number,ncid_20CR.node_number,year_array_ERA20C)
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^",label="ERA20C vs. 20CRV2c",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CR20')
ERA20C_output = xr.merge([ERA20C_output,output])

output=create_year_association(ncid_ERA20C.node_number,ncid_ERA5.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color='black',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='black',marker="^",label="ERA5 vs. ERA20C",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA5')
ERA20C_output = xr.merge([ERA20C_output,output])

output=create_year_association(ncid_ERA20C.node_number,ncid_CERA.node_number,year_array_CERA)
plt.plot(output.time,output.dependency.values,color='magenta',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='magenta',marker="^",label="CERA vs. ERA20C",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CERA')
ERA20C_output = xr.merge([ERA20C_output,output])


directoryname='C:/Users/Luke/Desktop/Antarctic Research/Final Results/Entropy_Coefficents/'
filename='ERA20C_Permutations.nc'
ERA20C_output.to_netcdf(directoryname+filename) 

plt.plot()
plt.title('(c) Entropy Coefficient: ERA20C')
plt.xlabel('Year')
plt.ylabel('Entropy Coefficient')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()

''' 
CERA Permutations
'''

plt.subplot(4,2,7)



output=create_year_association(ncid_CERA.node_number,ncid_MERRA2.node_number,year_array_MERRA2)
plt.plot(output.time,output.dependency.values,color='orange',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='orange',marker="^",label="CERA vs. MERRA2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='MERRA2')
CERA_output = output

output=create_year_association(ncid_CERA.node_number,ncid_NCEP.node_number,year_array_NCEP)
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^",label="CERA vs. NCEP_NCAR2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ncep')
CERA_output = xr.merge([CERA_output,output])

output=create_year_association(ncid_CERA.node_number,ncid_JRA55.node_number,year_array_JRA55)
plt.plot(output.time,output.dependency.values,color='green',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='green',marker="^",label="CERA vs. JRA55",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='JRA55')
CERA_output = xr.merge([CERA_output,output])

output=create_year_association(ncid_CERA.node_number,ncid_ECMWF.node_number,year_array_ECMWF)
plt.plot(output.time,output.dependency.values,color='red',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='red',marker="^",label="CERA vs. ERA-Interim",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA_Interim')
CERA_output = xr.merge([CERA_output,output])

output=create_year_association(ncid_CERA.node_number,ncid_20CR.node_number,year_array_CERA)
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^",label="CERA vs. 20CRV2c",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CR20')
CERA_output = xr.merge([CERA_output,output])

output=create_year_association(ncid_CERA.node_number,ncid_ERA5.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color='black',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='black',marker="^",label="CERA vs. ERA5",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA5')
CERA_output = xr.merge([CERA_output,output])

output=create_year_association(ncid_CERA.node_number,ncid_ERA20C.node_number,year_array_CERA)
plt.plot(output.time,output.dependency.values,color='magenta',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='magenta',marker="^",label="CERA vs. ERA20C",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA20C')
CERA_output = xr.merge([CERA_output,output])

directoryname='C:/Users/Luke/Desktop/Antarctic Research/Final Results/Entropy_Coefficents/'
filename='CERA_Permutations.nc'
CERA_output.to_netcdf(directoryname+filename) 

plt.plot()
plt.title('(d) Entropy Coefficient: CERA')
plt.xlabel('Year')
plt.ylabel('Entropy Coefficient')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()


''' 
ERA5 Permutations
'''


plt.subplot(4,2,8)



output=create_year_association(ncid_ERA5.node_number,ncid_MERRA2.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color='orange',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='orange',marker="^",label="ERA5 vs. MERRA2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='MERRA2')
ERA5_output = output

output=create_year_association(ncid_ERA5.node_number,ncid_NCEP.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[12,:],marker="^",label="ERA5 vs. NCEP_NCAR2",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ncep')
ERA5_output = xr.merge([ERA5_output,output])


output=create_year_association(ncid_ERA5.node_number,ncid_JRA55.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color='purple',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='purple',marker="^",label="ERA5 vs. JRA55",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='JRA55')
ERA5_output = xr.merge([ERA5_output,output])

output=create_year_association(ncid_ERA5.node_number,ncid_ECMWF.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color='red',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='red',marker="^",label="ERA5 vs. ERA-Interim",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA_Interim')
ERA5_output = xr.merge([ERA5_output,output])

output=create_year_association(ncid_ERA5.node_number,ncid_20CR.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[4,:],marker="^",label="ERA5 vs. 20CRv2c",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CR20')
ERA5_output = xr.merge([ERA5_output,output])

output=create_year_association(ncid_ERA5.node_number,ncid_ERA20C.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color=colors[9,:],marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color=colors[9,:],marker="^",label="ERA5 vs. ERA20C",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='ERA20C')
ERA5_output = xr.merge([ERA5_output,output])

output=create_year_association(ncid_ERA5.node_number,ncid_CERA.node_number,year_array_ERA5)
output.dependency[-19]=np.nan
plt.plot(output.time,output.dependency.values,color='green',marker="^", linestyle="none")
plt.plot(output.time,output.dependency.values,color='green',marker="^",label="ERA5 vs. CERA",linewidth=3,markersize=7)
output = output.dependency.to_dataset(name='CERA')
ERA5_output = xr.merge([ERA5_output,output])

directoryname='C:/Users/Luke/Desktop/Antarctic Research/Final Results/Entropy_Coefficents/'
filename='ERA5_Permutations.nc'
ERA5_output.to_netcdf(directoryname+filename)  




plt.plot()
plt.title('(h) Entropy Coefficient: ERA5')
plt.xlabel('Year')
plt.ylabel('Entropy Coefficent')
plt.ylim(0, 1)
plt.legend(loc='upper left',fontsize='small')
plt.grid()



plt.savefig('Entropy_Coefficents_AP1')






