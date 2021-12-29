"""
Take the user inputted location in either the tweet element or user location, and return an actual place

NOTE: Kartverket's API is not flexible enough, Bergen, norge is not recognized etc.
Asked jessica for help finding location csv, and looking into google developers
"""
import re
import os
import time
import requests
import json
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

data = pd.read_csv('data/second_rendition_data/second_rendition_output.csv',
                    usecols = ['username', 'text', 'loc', 'created_at', 'like_count', 'quote_count']
                    )

url = 'https://ws.geonorge.no/stedsnavn/v1/navn'

cty=[]

i = 0
for i, line in enumerate(data['loc']):
    #extracting the place names that make sense, in string form
    print(line)
    print('-------')
    payload = {'sok': str(line), 'fuzzy':True, 'treffPerSide':1} #NOTE fuzzy is true
    res = requests.get(url, params = payload) 
    res = res.json() 
    print(res) 
    #city = GeoText(str(line)).cities #fuck geotext, change this for manual search through kartverket
    #unsure how well this geotext functions. may be better alternatives somewhere
    if res['navn'] != []:
        
        city = res['navn'][0]
        longitude_latitude = city['representasjonspunkt'] #ESPG:4258 format / ETRS89
        long_lat = (longitude_latitude['øst'], longitude_latitude['nord'])
        print(long_lat)
    else:
        #pass
        cty.append('')
    if i == 5:
        pass
data['city']=cty
data = data[data['loc'] != 'New York, NY'] #don't want new yark 

#note, data_out is now all the data with geo locations
data_out = pd.DataFrame()
data_out = data[data.city != ''].copy()  


print(data_out)
data_out.to_csv('full_geodata_longlat_noforeign.csv')

