# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 01:36:21 2019

@author: Luke


The intention of this program is to create a master file to run the Self Organising Map 
series along with the later sorting prosseses and graphs. First input is the desired SOM size. P

"""


import importlib
import gc

'''
AP1 Index

lat_index=76
lon_index1=139 
lon_index2=158


AP2 Index
lat_index=73
lon_index1=141 
lon_index2=156
'''

'''
Map Coordinates for AP2 = 280,310,-54,-75

Map Coordinates for AP1 = 276,314,-60,-75
'''



# Initiates the Self Organising Map
importlib.import_module('Synoptic Pattern Generator (SOM)')
gc.collect()

#Sorts each day for each reanalysis into the clossest SOM generated pattern.
print('Running 20CRv2c')
importlib.import_module('Euclidean_sorter_20CRv2c')
gc.collect()
print('Running ERA-I')
importlib.import_module('Euclidean_sorter_ERA-I')
gc.collect()
print('Running JRA55')
importlib.import_module('Euclidean_sorter_JRA')
gc.collect()
print('Running MERRA2')
importlib.import_module('Euclidean_sorter_MERRA')
gc.collect()
print('Running CERA20C')
importlib.import_module('Euclidian_sorter_CERA')
gc.collect()
print('Running ERA20C')
importlib.import_module('Euclidian_sorter_ERA20C')
gc.collect()
print('Running NCEP_NCAR2')
importlib.import_module('Euclidian_sorter_NCEP_NCAR2')
gc.collect()
print('Running ERA5')
importlib.import_module('Euclidian_sorter_ERA5')
gc.collect()


