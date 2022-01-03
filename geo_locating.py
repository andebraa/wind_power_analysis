"""
Take the user inputted location in either the tweet element or user location, and return an actual place

NOTE: Kartverket's API is not flexible enough, Bergen, norge is not recognized etc.
Asked jessica for help finding location csv, and looking into google developers
"""
# NOTE: we assume the first recognizable place in user bio is the main place
import re
import os
import time
import requests
import json
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim

places = pd.read_csv('places_norway.csv', usecols = ['places'], index_col = False)
places = places.places.str[1:-1] #remove first and last element in all rows. i.e. removing quotes
def geolocate(user_input):
    elems = user_input.split() 
    for elem in elems: #assume the first place it recognizes is the main place
        #add capitalization and other features
        #TODO manually handle cases such as 'Jorden' and 'Norge'?
        #TODO handle edge cases where user_input messes with regex pattern search
        if places.str.contains(str(elem), case=False).any():
            print(places[places.str.contains(str(elem), case=False)])
            return True



data = pd.read_csv('data/second_rendition_data/second_rendition_output.csv',
                    usecols = ['username', 'text', 'loc', 'created_at', 'like_count', 'quote_count']
                    )

orig_len = len(data)
#extract rows that has a valid place name
cty=[]

i = 0
for i, line in enumerate(data['loc']):
    #extracting the place names that make sense, in string form
    print(line, i)
    print('-------')
    if geolocate(line):
        cty.append(line)
    else:
        cty.append('')

    #if i == 5:
        #break
data['city']=cty
data = data[data['loc'] != 'New York, NY'] #don't want new yark 

#note, data_out is now all the data with geo locations
data_out = pd.DataFrame()
data_out = data[data.city != ''].copy()  
alt_len = len(data_out) 
print('original length: ', orig_len)
print('len geodata: ', alt_len)
print('percentage: ', orig_len/alt_len)


print(data_out)
data_out.to_csv('second_rendition_test_geolocate.csv')

