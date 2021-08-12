import numpy as np
import pandas as pd
import geopandas
import time
from geotext import GeoText
#from geopy.geocoders import get_geocoders_for_service

#get_geocoder_for_service('nomatim') #nominatim
data = pd.read_csv('all_data_all_time_edited.csv',
                    usecols = ['username', 'text', 'loc', 'created_at', 'like_count', 'quote_count']
                    )
print(data['loc'])
#idata_out = geopandas.tools.geocode(data.loc)

data_out = pd.DataFrame()

#print(geopandas.tools.geocode('Oslo, Norway'))
#data_out.append(geopandas.tools.geocode(data['loc']))

cty=[]
for i, line in enumerate(data['loc']):
    #print(GeoText(str(line)).cities)
    #data_out.append(GeoText(str(line)).cities)
    city = GeoText(str(line)).cities
    if city:
        
        cty.append(str(city))
        #data.iloc[i]['city'] = pd.Series(city)
    else:
        cty.append('')
        #data.iloc[i]['city'] = pd.Series('') 

print(len(data))
#print(data['city'])
print(data.iloc[0])
data['city']=cty
print(data)
print(data['city'])
print(len(data['city']))

non_empts = 0
for i, line in enumerate(data['city']):
    
    elem = line.split(' ')
    print(elem)
    if elem[1]:
        non_empts += 1 
        data_out.append(data.iloc[i])
    #print(geopandas.tools.geocode(line, 
    #                                provider = 'nominatim'),
    #                                user_agent = 'my_request'
    #                                )
    #time.sleep(1) #nominatim has a 1 request per second limit
print(data_out)
