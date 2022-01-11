"""
Script that loops over the places_norway list of places and call nominatim
for the longitude and latitude of these places. writes to list which
is then used to locate tweets in the dataset
"""
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import time 

places_data = pd.read_csv('places_norway.csv')
place_ = places_data.places.str[1:-1] #remove first and last element i.e. removing quotes

print(type(places_data))
#getting geografical data from nominatim. NOTE max 1 request per second
nom_instance = Nominatim(user_agent = 'GetLoc')

print(type(places_data))
places_data['longitude'] = ''
places_data['latitude'] = ''

print(type(places_data))
for i, place in place_.iteritems():
    place += ', Norway'
    print(place)
    try:
        GetLoc = nom_instance.geocode(place, timeout = 10)
        adress = GetLoc.address

        places_data.loc[i, 'longitude'] = GetLoc.longitude
        places_data.loc[i, 'latitude'] = GetLoc.latitude

    except:
        places_data.loc[i, 'longitude'] = 'drop'
        places_data.loc[i, 'latitude'] = 'drop'
        print('issues with', place) 

    time.sleep(1)

places_data[places_data.latitude.str.contains('drop', na=False)]
places_data[places_data.longitude.str.contains('drop', na=False)]
places_data['places'] = place_
places_data.to_csv('places_norway_longlat.csv')

