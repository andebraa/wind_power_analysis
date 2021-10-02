import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import geopandas


norway = geopandas.read_file('kommuner_komprimert.json')

data = pd.read_csv('small_datafile.csv',
                  usecols = ['username','text','loc','created_at',
                             'like_count','quote_count','city','latitude',
                             'longitude']
                  )

data['labels'] = np.random.randint(0,2,size = (len(data))) 
print(data.head())


gdf = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.longitude, data.latitude))

fig, ax = plt.subplots() 
norway.plot(ax = ax, alpha = 0.4)

gdf[gdf['labels'] == 0].plot(ax = ax, markersize = 20, color = 'red', marker = 'o', label = 'neg')
gdf[gdf['labels'] == 1].plot(ax = ax, markersize = 20, color = 'blue', marker = '^', label = 'pos') 
 


""" #natural earth lowres attempt
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
norway = world.log[world['name'] == 'Norway'] 

ax = world[world.continent == 'Europe'].plot()
gdf.plot(ax=ax, color = 'red')
"""

plt.show()


