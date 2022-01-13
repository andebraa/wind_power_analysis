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

places_longlat = pd.read_csv('data/places_norway_longlat.csv', usecols = ['places','latitude', 'longitude'], index_col = False)

def ratio_finder(user_input, matches):
    """

    function that takes a list of strings, and applies sequence matcher
    on all possible matches and the user input

    args:
    user_input: str
        the inputed user location
    matches: list of str
        the list of possible matches from places_longlat

    returns:
    best_match: str
        place with hightest ratio

    """
    ratios = []
    for i in range(len(matches)):
        ratios.append(SequenceMatcher(None, user_input, matches[i]).ratio())
    
    best_indx = np.argmax(ratios)
    best_match = matches[best_indx] 
    return best_match, best_indx

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
    #if ' ' in user_input:
    elems = user_input.split()
    #elif ',' in user_input: 
    #    elems = user_input.split(',')
    #elif '/' in user_input:
    #    elems = user_input.split('/')
    #elif '-' in user_input:
    #    elems = user_input.split('-')
    #else:
    #    elems = user_input.split()
    
    #places = places_longlat['places']
    #if user has input_list of multiple places
    for elem in elems: #assume the first place it recognizes is the main place
        #potential_place = places.str.contains(str(elem), case=False)
        
        try:
            elem = elem.replace(',', '')
        except:
            print('pass')
            pass
        
        #try:
        potential_place_mask = places_longlat['places'].str.contains(str(elem), case=False)
        #except:
        #    print('lost case: ', elem) 
        #    return False, False, False, False
        print('pot_mask', potential_place_mask.any())
        if potential_place_mask.any():
            pot_places = places_longlat.loc[potential_place_mask].astype('string')
            pot_places.index = pd.RangeIndex(len(pot_places.index)) #reset index of pot_places 
            print('-'*10)
            print('pot_places: \n',pot_places)
            print('len pot places ', len(pot_places))            
           
            #if len(pot_places['places']) > 1: #more than one place matches 
                

            
            if len(pot_places) > 1:
                print('\n ******************************** \n ')
                #If multiple places fulfill the sequence criteria we use
                #SequenceMatcher to select the one which fits the best.
                best_match, best_indx = ratio_finder(user_input, pot_places['places'].tolist())
                print('after ratio finder')
                print(best_match, best_indx)
                best_long = pot_places.loc[best_indx, 'longitude']
                best_lat = pot_places.loc[best_indx, 'latitude']
                print(best_long)
                print(best_lat)
                return True, best_match, best_long, best_lat 
            else:
                return True, pot_places, places_longlat[potential_place_mask].longitude, \
                             places_longlat[potential_place_mask].latitude

        else:
            print('lost case: ', elem)
            return False, False, False, False

data = pd.read_csv('data/second_rendition_data/second_rendition_output.csv',
                    usecols = ['username', 'text', 'loc', 'created_at', 'like_count', 'quote_count']
                    )


#we wish to remove the known larges occurences of locations, i.e. oslo, bergen etc
data = data[data['loc'] != 'Jorden']
data = data[data['loc'] != 'New York, NY'] #don't want new yark 

print('length after removing norge and norway', len(data))

cty=[]
longitude = []
latitude = [] 
data['city'] = ''
data['latitude'] = ''
data['longitude'] = ''

drops = 0
for i, line_ in data.iterrows():
    #extracting the place names that make sense, in string form
    line = line_['loc']
    print(i, line)    
    #try:
    bol, place, _longitude, _latitude = geolocate(line)
    print('after call \n \n')
    print('bol: \n', bol)
    print('place: \n', place)
    print('long: \n', _longitude)
    print('lat: \n', _latitude)
    
    if bol:
        cty.append(place)
        longitude.append(_longitude)
        latitude.append(_latitude)

    else:
        cty.append('drop')
        longitude.append(0)
        latitude.append(0)
        print('drop; ', drops)
        drops += 1
data['loc'] = cty
data['latitude'] = latitude
data['longitude'] = longitude 

data = data.drop(data['loc'] == 'drop')
data = data.drop(data['longitude'] == 0)
data = data.drop(data['latitude'] == 0)


print('length after: ', len(data)) 
print('drops: ', drops)
data.to_csv('second_rendition_test_geolocate.csv')
"""
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
"""
