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

data_out = pd.DataFrame()


cty=[]
for i, line in enumerate(data['loc']):
    #extracting the place names that make sense, in string form
    
    city = GeoText(str(line)).cities
    #unsure how well this geotext functions. may be better alternatives somewhere
    if city:
        print(city)
        #r",*\s*'*(\w*)'*" captures ", 'Oslo'". If it's there. allowes regex to extract the multiple places.
        reg_extract = re.search(r"\['([\w\s]*)',*\s*'*([\w\s]*)'*,*\s*'*([\w\s]*)'*(?:,*\s*'*([\w\s]*)'*)*\]", str(city)) #regex is my passion 
        #TODO fix scandinavian letters. Also doesn't seem to handle repeats of the last capture group
        #TODO add - support. doesn't manage Mitry-Mory (in france)
        print(reg_extract)
        if len(reg_extract.group()) > 1:
            places = list(reg_extract.group())
            if 'Oslo' in places:
                places.remove('Oslo')
        
        print(reg_extract)
        print(reg_extract.group())
        cty.append(reg_extract.group(0)) #after removing Oslo if there are multiple, we simply pick the first one 
    else:
        #pass
        cty.append('')
        

data['city']=cty

#note, data_out is now all the data with geo locations

data_out = data[data.city != ''] 
print(data_out)

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

