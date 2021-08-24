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
        
        cty.append(reg_extract.group(0)) #We assume that the first written place is the main one  
    else:
        #pass
        cty.append('')
        

data['city']=cty
data = data[data['loc'] != 'New York, NY'] #don't want new yark 

#note, data_out is now all the data with geo locations
data_out = pd.DataFrame()
data_out = data[data.city != ''] 

#removing mentions of Oslo and Bergen, as there are a fuck ton of them which takes time
working_data = data_out[data_out['loc'] != 'Oslo, Norge']
working_data = working_data[working_data['loc'] != 'Oslo, Norway']
working_data = working_data[working_data['loc'] != 'Bergen, Norway']
working_data = working_data[working_data['loc'] != 'Oslo']
working_data = working_data[working_data['loc'] != 'Bergen']
working_data = working_data[working_data['loc'] != 'Trondheim, Norge']
working_data = working_data[working_data['loc'] != 'Trondheim, Norway']
#this reduces from 44423 to 14152

print('mode')
print(working_data['loc'].mode())
print(len(working_data['loc']))
print(working_data['loc'])
raise AssertionError('twat')





#getting geografical data from nominatim. NOTE max 1 request per second
Nom_instance = Nominatim(user_agent = 'GetLoc')
latitude = []
longitude = []

for line in data_out['city']:
    print(len(line))
    print(type(line))
    if len(line) > 1:
        line = line[0] #is user has written multiple places, we just use the first one.
    getLoc = Nom_instance.geocode(line)
    try:
        print(getLoc.address) 
    except:
        print('nah')
    latitude.append(getLoc.latitude)
    longitude.append(getLoc.longitude)
    time.sleep(1) 
     

data_out['latitude'] = latitude
data_out['longitude'] = longitude

data_out.to_csv('full_geodata.csv')

