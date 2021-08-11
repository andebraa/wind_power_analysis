import numpy as np
import pandas as pd
import geopandas 
from geotext import GeoText

data = pd.read_csv('all_data_all_time_edited.csv',
                    usecols = ['username', 'text', 'loc', 'created_at', 'like_count', 'quote_count']
                    )
print(data['loc'])
#data_out = geopandas.tools.geocode(data.loc)

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
