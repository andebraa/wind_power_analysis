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
            pass
        
        potential_place_mask = places_longlat['places'].str.contains(str(elem), case=False)
        if potential_place_mask.any():
            pot_places = places_longlat.loc[potential_place_mask].astype('string')
            pot_places.index = pd.RangeIndex(len(pot_places.index)) #reset index of pot_places 
            
            if len(pot_places) > 1:
                print('\n ******************************** \n ')
                #If multiple places fulfill the sequence criteria we use
                #SequenceMatcher to select the one which fits the best.
                best_match, best_indx = ratio_finder(user_input, pot_places['places'].tolist())
                best_long = pot_places.loc[best_indx, 'longitude']
                best_lat = pot_places.loc[best_indx, 'latitude']

                return True, best_match, float(best_long), float(best_lat) #best match is string 
            else:
                return True, pot_places.places.to_string(index=False), float(places_longlat[potential_place_mask].longitude), \
                             float(places_longlat[potential_place_mask].latitude)

        else:
            
            print('lost case: ', elem)
            print('user_input: ', user_input)
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
data['latitude'] = ''
data['longitude'] = ''

drops = 0
unexplained_drops = 0
j = 0
for i, line_ in data.iterrows():
    #extracting the place names that make sense, in string form
    line = line_['loc']
    try:
        bol, _place_, _longitude, _latitude = geolocate(line)
    except:
        unexplained_drops += 1
        data = data.drop(data.index[j])
        
        data.index = pd.RangeIndex(len(data.index)) #reset index of pot_places 
        continue 
    
    if bol:
        cty.append(_place_)
        longitude.append(_longitude)
        latitude.append(_latitude)
        j += 1

    else:
        cty.append('drop')
        longitude.append(0.0)
        latitude.append(0.0)
        print('drop; ', drops)
        drops += 1
        j += 1


print('drops: ', drops)
print('unexplained drops: ', unexplained_drops)

data['loc'] = cty
data['latitude'] = latitude
data['longitude'] = longitude 

print(data.head())
print('here \n')
print(data.tail())
data.to_csv('test_data.csv')
data = data.drop(data.index[data['loc'] == 'drop'])
data = data.drop(data.index[data['latitude'] == 0.0])
data = data.drop(data.index[data['longitude'] == 0.0])

print('length after: ', len(data)) 
data.to_csv('second_rendition_test_geolocate.csv')
















