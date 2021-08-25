import numpy as np
import pandas as pd
import time
from geopy.geocoders import Nominatim
import re
from geotext import GeoText
#from geopy.geocoders import get_geocoders_for_service

#get_geocoder_for_service('nomatim') #nominatim
data = pd.read_csv('all_data_all_time_edited.csv',
                    usecols = ['username', 'text', 'loc', 'created_at', 'like_count', 'quote_count']
                    )

cty=[]
for i, line in enumerate(data['loc']):
    #extracting the place names that make sense, in string form
    
    city = GeoText(str(line)).cities
    #unsure how well this geotext functions. may be better alternatives somewhere
    if city:
        #r",*\s*'*(\w*)'*" captures ", 'Oslo'". If it's there. allowes regex to extract the multiple places.
        reg_extract = re.search(r"\['([\w\sæøåÆØÅ-]*)',*\s*'*([\w\sæøåÆØÅ-]*)'*,*\s*'*([\w\sæøåÆØÅ-]*)'*(?:,*\s*'*([\w\sæøåÆØÅ-]*)'*)*\]", str(city)) #regex is my passion 
        #TODO fix 'pic one' doesn't work seemingly 
        cty.append(reg_extract.group(1)) #We assume that the first written place is the main one  
    else:
        #pass
        cty.append('')
        

data['city']=cty
data = data[data['loc'] != 'New York, NY'] #don't want new yark 

#note, data_out is now all the data with geo locations
data_out = pd.DataFrame()
data_out = data[data.city != ''].copy()  

#removing mentions of Oslo and Bergen, as there are a fuck ton of them which takes time
working_data = data_out[data_out['loc'] != 'Oslo, Norge']
working_data = working_data[working_data['loc'] != 'Oslo, Norway']
working_data = working_data[working_data['loc'] != 'Bergen, Norway']
working_data = working_data[working_data['loc'] != 'Oslo']
working_data = working_data[working_data['loc'] != 'Bergen']
working_data = working_data[working_data['loc'] != 'Trondheim, Norge']
working_data = working_data[working_data['loc'] != 'Trondheim, Norway']
#this reduces from 44423 to 14152


data_out['latitude'] = ''
data_out['longitude'] = '' 



#getting geografical data from nominatim. NOTE max 1 request per second
nom_instance = Nominatim(user_agent = 'GetLoc')
latitude = []
longitude = []

oslo_coords = nom_instance.geocode('Oslo, Norway') 
bergen_coords = nom_instance.geocode('Bergen, Norway')
trondheim_coords = nom_instance.geocode('Trondheim, Norway') 

data_out.loc[data['loc'] == 'Oslo, Norway', 'latitude'] = oslo_coords.latitude  
data_out.loc[data['loc'] == 'Oslo, Norway', 'longitude'] = oslo_coords.longitude
data_out.loc[data['loc'] == 'Oslo', 'latitude'] = oslo_coords.latitude
data_out.loc[data['loc'] == 'Oslo', 'longitude'] = oslo_coords.longitude
data_out.loc[data['loc'] == 'Bergen, Norway', 'latitude'] = bergen_coords.latitude
data_out.loc[data['loc'] == 'Bergen, Norway', 'longitude'] = bergen_coords.longitude
data_out.loc[data['loc'] == 'Bergen', 'latitude'] = bergen_coords.latitude
data_out.loc[data['loc'] == 'Bergen', 'longitude'] = bergen_coords.longitude
data_out.loc[data['loc'] == 'Trondheim, Norway', 'latitude'] = trondheim_coords.latitude
data_out.loc[data['loc'] == 'Trondheim, Norway', 'longitude'] = trondheim_coords.longitude
data_out.loc[data['loc'] == 'Trondheim, Norge', 'latitude'] = trondheim_coords.latitude
data_out.loc[data['loc'] == 'Trondheim, Norge', 'longitude'] = trondheim_coords.longitude

#works till here
i = 0
fin = len(working_data['city'])
for line in working_data['city']:
    print(line)
    print(i, fin)
    getLoc = nom_instance.geocode(line)
    try:
        print(getLoc.address) 
    except:
        print('nah')
    latitude.append(getLoc.latitude)
    longitude.append(getLoc.longitude)
    time.sleep(1) 
    i += 1

data_out.loc[data_out['latitude'] == '', 'latitude'] = latitude
data_out.loc[data_out['longitude'] == '', 'longitude'] = longitude


data_out.to_csv('full_geodata.csv')

