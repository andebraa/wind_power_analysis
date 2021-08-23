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
        
        cty.append(str(city))
    else:
        #pass
        cty.append('')
        

data['city']=cty

#note, data_out is now all the data with geo locations

data_out = data[data.city != ''] 


#getting geografical data from nominatim. NOTE max 1 request per second
Nom_instance = Nominatim(user_agent = 'GetLoc')
latitude = []
longitude = []
for line in data_out[2:]:
    print(line)
    getLoc = Nom_instance.geocode(line['city'])
    print(getLoc.adress) 
    latitude.append(getLoc.latitude)
    longitude.append(getLoc.longitude)
    time.sleep(2) 

