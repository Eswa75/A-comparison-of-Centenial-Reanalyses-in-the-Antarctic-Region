# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:15:11 2019

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



directoryname='C:/Users/Luke/Desktop/Antarctic Research/Final Results/Entropy_Coefficents/SOM43' 
filename='ERA5_Permutations.nc'
ncid1 =xr.open_dataset(os.path.join(directoryname,filename))
filename='CR20_Permutations.nc'
ncid2 =xr.open_dataset(os.path.join(directoryname,filename))
filename='ERA_Interim_Permutations.nc'
ncid3 =xr.open_dataset(os.path.join(directoryname,filename))
filename='ERA20C_Permutations.nc'
ncid4 =xr.open_dataset(os.path.join(directoryname,filename))
filename='CERA_Permutations.nc'
ncid5 =xr.open_dataset(os.path.join(directoryname,filename))
filename='MERRA2_Permutations.nc'
ncid6 =xr.open_dataset(os.path.join(directoryname,filename))
filename='NCEP2_Permutations.nc'
ncid7 =xr.open_dataset(os.path.join(directoryname,filename))
filename='JRA55_Permutations.nc'
ncid8 =xr.open_dataset(os.path.join(directoryname,filename))





'''
ERA5
''' 

#Post 1979
Period = slice('1980-01-01','2009-12-31')
MERRA2 = ncid1.MERRA2.sel(time=Period).mean()
ERA20C = ncid1.ERA20C.sel(time=Period).mean()
NCAR_NCEP2 = ncid1.ncep.sel(time=Period).mean()
JRA55 = ncid1.JRA55.sel(time=Period).mean()
ERA_Interim = ncid1.ERA_Interim.sel(time=Period).mean()
CERA = ncid1.CERA.sel(time=Period).mean()
CR20 = ncid1.CR20.sel(time=Period).mean()

print('ERA5 1979')
ERA5_post_1979 = xr.concat([MERRA2,ERA20C,NCAR_NCEP2,JRA55,ERA_Interim,CERA,CR20])
output = ERA5_post_1979.mean().to_dataset(name='ERA5_post_1979')
Table_data = output


'''
ERA-Interim
'''
#Post 1979
Period = slice('1980-01-01','2009-12-31')
MERRA2 = ncid3.MERRA2.sel(time=Period).mean()
ERA20C = ncid3.ERA20C.sel(time=Period).mean()
NCAR_NCEP2 = ncid3.ncep.sel(time=Period).mean()
JRA55 = ncid3.JRA55C.sel(time=Period).mean()
ERA5 = ncid3.ERA5.sel(time=Period).mean()
CERA = ncid3.CERA.sel(time=Period).mean()
CR20 = ncid3.CR20.sel(time=Period).mean()


ERA_Interim_post_1979 = xr.concat([MERRA2,ERA20C,NCAR_NCEP2,JRA55,ERA5,CERA,CR20])
output = ERA_Interim_post_1979.mean().to_dataset(name='ERA_Interim__post_1979')
Table_data = xr.merge([Table_data,output])




'''
ERA20C
'''

#Post 1979
Period = slice('1979-01-01','2009-12-31')
MERRA2 = ncid4.MERRA2.sel(time=Period).mean()
ERA5 = ncid4.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid4.ncep.sel(time=Period).mean()
JRA55 = ncid4.JRA55.sel(time=Period).mean()
ERA_Interim = ncid4.ERA_Interim.sel(time=Period).mean()
CERA = ncid4.CERA.sel(time=Period).mean()
CR20 = ncid4.CR20.sel(time=Period).mean()


ERA20C_post_1979 = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,CERA,CR20])

output = ERA20C_post_1979.mean().to_dataset(name='ERA20C_post_1979')
Table_data = xr.merge([Table_data,output])

#Post 1957
Period = slice('1957-01-01','1978-12-31')
MERRA2 = ncid4.MERRA2.sel(time=Period).mean()
ERA5 = ncid4.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid4.ncep.sel(time=Period).mean()
JRA55 = ncid4.JRA55.sel(time=Period).mean()
ERA_Interim = ncid4.ERA_Interim.sel(time=Period).mean()
CERA = ncid4.CERA.sel(time=Period).mean()
CR20 = ncid4.CR20.sel(time=Period).mean()


ERA20C_post_1957 = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,CERA,CR20])

output = ERA20C_post_1957.mean().to_dataset(name='ERA20C_1957-1979')
Table_data = xr.merge([Table_data,output])

#Post 1900
Period = slice('1900-01-01','1956-12-31')
MERRA2 = ncid4.MERRA2.sel(time=Period).mean()
ERA5 = ncid4.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid4.ncep.sel(time=Period).mean()
JRA55 = ncid4.JRA55.sel(time=Period).mean()
ERA_Interim = ncid4.ERA_Interim.sel(time=Period).mean()
CERA = ncid4.CERA.sel(time=Period).mean()
CR20 = ncid4.CR20.sel(time=Period).mean()


ERA20C_post_1900 = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,CERA,CR20])
output = ERA20C_post_1900.mean().to_dataset(name='ERA20C_1900-1957')
Table_data = xr.merge([Table_data,output])


# 1900 - 1917
Period = slice('1900-01-01','1916-12-31')
MERRA2 = ncid4.MERRA2.sel(time=Period).mean()
ERA5 = ncid4.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid4.ncep.sel(time=Period).mean()
JRA55 = ncid4.JRA55.sel(time=Period).mean()
ERA_Interim = ncid4.ERA_Interim.sel(time=Period).mean()
CERA = ncid4.CERA.sel(time=Period).mean()
CR20 = ncid4.CR20.sel(time=Period).mean()


ERA20C1910s = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,CERA,CR20])
output = ERA20C1910s.mean().to_dataset(name='ERA20C1910')
Table_data = xr.merge([Table_data,output])

#1978-1985
Period = slice('1978-01-01','1985-12-31')
MERRA2 = ncid4.MERRA2.sel(time=Period).mean()
ERA5 = ncid4.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid4.ncep.sel(time=Period).mean()
JRA55 = ncid4.JRA55.sel(time=Period).mean()
ERA_Interim = ncid4.ERA_Interim.sel(time=Period).mean()
CERA = ncid4.CERA.sel(time=Period).mean()
CR20 = ncid4.CR20.sel(time=Period).mean()


ERA20C1980 = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,CERA,CR20])
output = ERA20C1980.mean().to_dataset(name='ERA20C1980')
Table_data = xr.merge([Table_data,output])


'''
CERA
'''

#Post 1979
Period = slice('1979-01-01','2009-12-31')
MERRA2 = ncid5.MERRA2.sel(time=Period).mean()
ERA5 = ncid5.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid5.ncep.sel(time=Period).mean()
JRA55 = ncid5.JRA55.sel(time=Period).mean()
ERA_Interim = ncid5.ERA_Interim.sel(time=Period).mean()
CR20 = ncid5.CR20.sel(time=Period).mean()
ERA20C = ncid5.ERA20C.sel(time=Period).mean()


CERA_post_1979 = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,ERA20C,CR20])
output = CERA_post_1979.mean().to_dataset(name='CERA_post_1979')
Table_data = xr.merge([Table_data,output])

#Post 1957
Period = slice('1957-01-01','1978-12-31')
MERRA2 = ncid5.MERRA2.sel(time=Period).mean()
ERA5 = ncid5.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid5.ncep.sel(time=Period).mean()
JRA55 = ncid5.JRA55.sel(time=Period).mean()
ERA_Interim = ncid5.ERA_Interim.sel(time=Period).mean()
ERA20C = ncid5.ERA20C.sel(time=Period).mean()
CR20 = ncid5.CR20.sel(time=Period).mean()


CERA_post_1957 = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,ERA20C,CR20])
output = CERA_post_1957.mean().to_dataset(name='CERA_1957-1979')
Table_data = xr.merge([Table_data,output])

#Post 1900
Period = slice('1900-01-01','1956-12-31')
MERRA2 = ncid5.MERRA2.sel(time=Period).mean()
ERA5 = ncid5.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid5.ncep.sel(time=Period).mean()
JRA55 = ncid5.JRA55.sel(time=Period).mean()
ERA_Interim = ncid5.ERA_Interim.sel(time=Period).mean()
ERA20C = ncid5.ERA20C.sel(time=Period).mean()
CR20 = ncid5.CR20.sel(time=Period).mean()


CERA_post_1900 = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,ERA20C,CR20])
print(CERA_post_1979.mean())
output = CERA_post_1900.mean().to_dataset(name='CERA_1900-1957')
Table_data = xr.merge([Table_data,output])


#Post 1900-1917
Period = slice('1900-01-01','1916-12-31')
MERRA2 = ncid5.MERRA2.sel(time=Period).mean()
ERA5 = ncid5.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid5.ncep.sel(time=Period).mean()
JRA55 = ncid5.JRA55.sel(time=Period).mean()
ERA_Interim = ncid5.ERA_Interim.sel(time=Period).mean()
ERA20C = ncid5.ERA20C.sel(time=Period).mean()
CR20 = ncid5.CR20.sel(time=Period).mean()


CERA1910 = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,ERA20C,CR20])
output = CERA1910.mean().to_dataset(name='CERA1910')
Table_data = xr.merge([Table_data,output])

#1978-1985
Period = slice('1978-01-01','1985-12-31')
MERRA2 = ncid5.MERRA2.sel(time=Period).mean()
ERA5 = ncid5.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid5.ncep.sel(time=Period).mean()
JRA55 = ncid5.JRA55.sel(time=Period).mean()
ERA_Interim = ncid5.ERA_Interim.sel(time=Period).mean()
ERA20C = ncid5.ERA20C.sel(time=Period).mean()
CR20 = ncid5.CR20.sel(time=Period).mean()


CERA1980 = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,ERA20C,CR20])
output = CERA1980.mean().to_dataset(name='CERA1980')
Table_data = xr.merge([Table_data,output])


'''
20CRv2c
'''

#Post 1979
Period = slice('1979-01-01','2009-12-31')
MERRA2 = ncid2.MERRA2.sel(time=Period).mean()
ERA5 = ncid2.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid2.ncep.sel(time=Period).mean()
JRA55 = ncid2.JRA55.sel(time=Period).mean()
ERA_Interim = ncid2.ERA_Interim.sel(time=Period).mean()
CERA = ncid2.CERA.sel(time=Period).mean()
ERA20C = ncid2.ERA20C.sel(time=Period).mean()

CR20_post_1979 = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,ERA20C,CERA])
output = CR20_post_1979.mean().to_dataset(name='CR20_post_1979')
Table_data = xr.merge([Table_data,output])

#Post 1957
Period = slice('1957-01-01','1978-12-31')
MERRA2 = ncid2.MERRA2.sel(time=Period).mean()
ERA5 = ncid2.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid2.ncep.sel(time=Period).mean()
JRA55 = ncid2.JRA55.sel(time=Period).mean()
ERA_Interim = ncid2.ERA_Interim.sel(time=Period).mean()
ERA20C = ncid2.ERA20C.sel(time=Period).mean()
CERA = ncid2.CERA.sel(time=Period).mean()

CR20_post_1957 = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,ERA20C,CERA])
output = CR20_post_1957.mean().to_dataset(name='CR20_1957-1979')
Table_data = xr.merge([Table_data,output])

#Post 1900
Period = slice('1900-01-01','1956-12-31')
MERRA2 = ncid2.MERRA2.sel(time=Period).mean()
ERA5 = ncid2.ERA5.sel(time=Period).mean()
NCAR_NCEP2 = ncid2.ncep.sel(time=Period).mean()
JRA55 = ncid2.JRA55.sel(time=Period).mean()
ERA_Interim = ncid2.ERA_Interim.sel(time=Period).mean()
ERA20C = ncid2.ERA20C.sel(time=Period).mean()
CERA = ncid2.CERA.sel(time=Period).mean()

CR20_post_1900 = xr.concat([MERRA2,ERA5,NCAR_NCEP2,JRA55,ERA_Interim,ERA20C,CERA])
output = CR20_post_1900.mean().to_dataset(name='CR20_1900-1957')
Table_data = xr.merge([Table_data,output])

'''
MERRA2
'''

#Post 1979
Period = slice('1980-01-01','2009-12-31')
ERA_Interim = ncid6.ERA_Interim.sel(time=Period).mean()
ERA20C = ncid6.ERA20C.sel(time=Period).mean()
NCAR_NCEP2 = ncid6.ncep.sel(time=Period).mean()
JRA55 = ncid6.JRA55.sel(time=Period).mean()
ERA5 = ncid6.ERA5.sel(time=Period).mean()
CERA = ncid6.CERA.sel(time=Period).mean()
CR20 = ncid6.CR20.sel(time=Period).mean()


MERRA2_post_1979 = xr.concat([ERA_Interim,ERA20C,NCAR_NCEP2,JRA55,ERA5,CERA,CR20])
output = MERRA2_post_1979.mean().to_dataset(name='MERRA2_post_1979')
Table_data = xr.merge([Table_data,output])


'''
NCEP/NCAR2
'''

#Post 1979
Period = slice('1980-01-01','2009-12-31')
ERA_Interim = ncid7.ERA_Interim.sel(time=Period).mean()
ERA20C = ncid7.ERA20C.sel(time=Period).mean()
MERRA2 = ncid7.MERRA2.sel(time=Period).mean()
JRA55 = ncid7.JRA55.sel(time=Period).mean()
ERA5 = ncid7.ERA5.sel(time=Period).mean()
CERA = ncid7.CERA.sel(time=Period).mean()
CR20 = ncid7.CR20.sel(time=Period).mean()

NCEP_NCAR2_post_1979 = xr.concat([ERA_Interim,ERA20C,MERRA2,JRA55,ERA5,CERA,CR20])
output = NCEP_NCAR2_post_1979.mean().to_dataset(name='NCEP_NCAR2_post_1979')
Table_data = xr.merge([Table_data,output])


'''
JRA55
'''

#Post 1979
Period = slice('1980-01-01','2009-12-31')
ERA_Interim = ncid8.ERA_Interim.sel(time=Period).mean()
ERA20C = ncid8.ERA20C.sel(time=Period).mean()
MERRA2 = ncid8.MERRA2.sel(time=Period).mean()
NCEP_NCAR2 = ncid8.ncep.sel(time=Period).mean()
ERA5 = ncid8.ERA5.sel(time=Period).mean()
CERA = ncid8.CERA.sel(time=Period).mean()
CR20 = ncid8.CR20.sel(time=Period).mean()

JRA55_post_1979 = xr.concat([ERA_Interim,ERA20C,MERRA2,NCEP_NCAR2,ERA5,CERA,CR20])
output = JRA55_post_1979.mean().to_dataset(name='JRA55_post_1979')
Table_data = xr.merge([Table_data,output])



#Post 1979
Period = slice('1958-01-01','1978-12-31')
ERA_Interim = ncid8.ERA_Interim.sel(time=Period).mean()
ERA20C = ncid8.ERA20C.sel(time=Period).mean()
MERRA2 = ncid8.MERRA2.sel(time=Period).mean()
NCEP_NCAR2 = ncid8.ncep.sel(time=Period).mean()
ERA5 = ncid8.ERA5.sel(time=Period).mean()
CERA = ncid8.CERA.sel(time=Period).mean()
CR20 = ncid8.CR20.sel(time=Period).mean()

JRA55_post_1958 = xr.concat([ERA_Interim,ERA20C,MERRA2,NCEP_NCAR2,ERA5,CERA,CR20])
output = JRA55_post_1958.mean().to_dataset(name='JRA55_1957-1979')
Table_data = xr.merge([Table_data,output])

print(Table_data)



































































