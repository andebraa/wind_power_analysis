"""
Take the user inputted location in either the tweet element or user location, and return an actual place

NOTE: Kartverket's API is not flexible enough, Bergen, norge is not recognized etc.
Asked jessica for help finding location csv, and looking into google developers

Removed counties from places norway. Maybe add theese again?

NOTE: we assume the first recognizable place in user bio is the main place
"""
import re
import os
import sys
import time
import requests
import json
import numpy as np
import pandas as pd
from tqdm import trange
from geopy.geocoders import Nominatim
from difflib import SequenceMatcher

places_longlat = pd.read_csv('data/places_norway.csv', usecoils = ['places', 'longitude', 'latitude'], index_col = False)
places_longlat = places_longlat.places.str[1:-1] #remove first and last element in all rows. i.e. removing quotes
def geolocate(user_input):
    """
    Script that compares a user input (user selected place) and compares it to a list of 
    places in norway based on the following list from wikipedia. 
    https://no.wikipedia.org/wiki/Liste_over_Norges_st%C3%B8rste_tettsteder

    The counties are removed, as well as multiple variations of Oslo

    The first place in the user input that is a match is picked to be the location.
    i.e. it is assumed this is the main place of the user.

    If multiple places in the list are a match, the difflib function sequence matcher
    is used to pick the most similar one. 
    """
    #Some users don't split with comma, but rather spaces. this tries to handle this
    if ',' in user_input: 
        elems = user_input.split(',')
    else:
        elems = user_input.split()
    
    places = places_longlat['places']
    for elem in elems: #assume the first place it recognizes is the main place
        print(elem)
        #potential_place = places.str.contains(str(elem), case=False)
        #TODO: make this function return longitude and latitude and be done with it
        try:
            potential_place = places.str.contains(str(elem), case=False)
        except:
            print('lost case: ', elem)
            return False, '_'
        if potential_place.any():
            pot_places = places[potential_place].tolist() 
            if len(pot_places) > 1:
                #If multiple places fulfill the sequence criteria we use
                #SequenceMatcher to select the one which fits the best.
                ratios = []
                for i in range(len(pot_places)): #loop over all matches
                        ratios.append(SequenceMatcher(None, elem, pot_places[i]).ratio())
                best_match = np.argmax(ratios)
                return True, pot_places[best_match]
            else:
                return True, pot_places


data = pd.read_csv('data/second_rendition_data/second_rendition_output.csv',
                    usecols = ['username', 'text', 'loc', 'created_at', 'like_count', 'quote_count']
                    )

orig_len = len(data)
#extract rows that has a valid place name


print('length before removing "norge" and "Norway', len(data))
#we wish to remove the known larges occurences of locations, i.e. oslo, bergen etc
data = data[data['loc'] != 'Jorden']
data = data[data['loc'] != 'New York, NY'] #don't want new yark 

print('length after removing norge and norway', len(data))


cty=[]



i = 0
for i, line in enumerate(data['loc']):
    #extracting the place names that make sense, in string form
    print(line, i)
    try:
        bol, place = geolocate(line)
        cty.append(place)
    except:
        cty.append('')

    #if i == 5:
        #break
data['city']=cty

#note, data_out is now all the data with geo locations
data_out = pd.DataFrame()

data_out = data[data.city != ''].copy() #removing tweets without location
alt_len = len(data_out) 
print('original length: ', orig_len)
print('len geodata: ', alt_len)
print('percentage: ', alt_len/orig_len)

print('-----------')
print(len(data_out))
print(len(data))


data_out['indx'] = np.arange(len(data_out)) #manually make index collumn. fuck pandas
data_out['indx'] = data_out['indx'].astype(int)
data_out['latitude'] = ''
data_out['longitude'] = '' #initialixing empty latitude longitude collumn

print(data_out['latitude'])
#removing mentions of Oslo and Bergen, as there are a fuck ton of them which takes time
#working data is only used to place the non bergen oslo trondheim places on the map

#woorking_data = data_out.mask((data_out['loc'] != 'Oslo, Norge')|(data_out['loc'] != 'Oslo, Norway')|\
#                              (data_out['loc'] != 'Bergen, Norway') | (data_out['loc'] != 'Oslo') |\
#                              (data_out['loc'] != 'Bergen') | (data_out['loc'] != 'Trondheim, Norge') |\
#                              (data_out['loc'] != 'Trondheim, Norway')) 


working_data = data_out[data_out['loc'] != 'Oslo, Norge']
working_data = working_data[working_data['loc'] != 'Oslo, Norway']
working_data = working_data[working_data['loc'] != 'Bergen, Norway']
working_data = working_data[working_data['loc'] != 'Oslo']
working_data = working_data[working_data['loc'] != 'Bergen']
working_data = working_data[working_data['loc'] != 'Trondheim, Norge']
working_data = working_data[working_data['loc'] != 'Trondheim, Norway']
#this reduces from 44423 to 14152



#getting geografical data from nominatim. NOTE max 1 request per second
nom_instance = Nominatim(user_agent = 'GetLoc')

oslo_coords = nom_instance.geocode('Oslo, Norway') 
bergen_coords = nom_instance.geocode('Bergen, Norway')
trondheim_coords = nom_instance.geocode('Trondheim, Norway') 

data_out.loc[data_out['loc'] == 'Oslo, Norway', 'latitude'] = oslo_coords.latitude  
data_out.loc[data_out['loc'] == 'Oslo, Norway', 'longitude'] = oslo_coords.longitude
data_out.loc[data_out['loc'] == 'Oslo, Norge', 'latitude'] = oslo_coords.latitude
data_out.loc[data_out['loc'] == 'Oslo, Norge', 'longitude'] = oslo_coords.longitude
data_out.loc[data_out['loc'] == 'Oslo', 'latitude'] = oslo_coords.latitude
data_out.loc[data_out['loc'] == 'Oslo', 'longitude'] = oslo_coords.longitude
data_out.loc[data_out['loc'] == 'Bergen, Norway', 'latitude'] = bergen_coords.latitude
data_out.loc[data_out['loc'] == 'Bergen, Norway', 'longitude'] = bergen_coords.longitude
data_out.loc[data_out['loc'] == 'Bergen', 'latitude'] = bergen_coords.latitude
data_out.loc[data_out['loc'] == 'Bergen', 'longitude'] = bergen_coords.longitude
data_out.loc[data_out['loc'] == 'Trondheim, Norway', 'latitude'] = trondheim_coords.latitude
data_out.loc[data_out['loc'] == 'Trondheim, Norway', 'longitude'] = trondheim_coords.longitude
data_out.loc[data_out['loc'] == 'Trondheim, Norge', 'latitude'] = trondheim_coords.latitude
data_out.loc[data_out['loc'] == 'Trondheim, Norge', 'longitude'] = trondheim_coords.longitude

print(data_out['latitude'])

i = 0

fin = len(working_data['city'])

for i, row in working_data.iterrows():
    place = row['city']
    if isinstance(place, list):
        place = place[0] 
    print(place)
    print(i, fin)
    if place in places_longlat['places']
    if country[-5:] != 'Norge': #Shouldn't be any occurances of this
        print('-'*20)
        #data_out.iloc[data_out.index[workdata_indx], 'latitude'] = 'drop'
        data_out.loc[workdata_indx, 'latitude'] = 'drop'
        data_out.loc[workdata_indx, 'longitude'] = 'drop'
    else:

        print(workdata_indx)
        print(type(workdata_indx))
        data_out.loc[workdata_indx, 'latitude'] = getLoc.latitude
        data_out.loc[workdata_indx, 'longitude'] = getLoc.longitude
    time.sleep(1) 
    
    i += 1



#for i in range(len(working_data['city'])):
#    latitude.append(np.random.randint(1000))
#    longitude.append(np.random.randint(1000))
#print(len(latitude))
#print(data_out.latitude.value_counts())
data_out[~data_out.latitude.str.contains('drop')]
data_out[~data_out.longitude.str.contains('drop')]

#data_out.loc[data_out['latitude'] == '', 'latitude'] = latitude
#data_out.loc[data_out['longitude'] == '', 'longitude'] = longitude
#data_out = data_out[data_out['latitude'] != ''] 

print('size after nominatim: ', len(data_out))

del data_out['indx']

print(data_out)
data_out.to_csv('second_rendition_test_geolocate.csv')

