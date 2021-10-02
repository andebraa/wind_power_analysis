import pandas as pd 
import matplotlib.pyplot as plt 
import geopandas

<<<<<<< HEAD

norway = geopandas.read_file('kommuner_komprimert.json')
=======
norway = geopandas.read_file('kommuner_komprimert.json')


>>>>>>> ec1da845b99d57894b82eeb0340942477fd335a3

data = pd.read_csv('small_datafile.csv',
                  usecols = ['username','text','loc','created_at',
                             'like_count','quote_count','city','latitude',
                             'longitude']
                  )

<<<<<<< HEAD
norway.plot()
plt.plot(data['latitude'], data['longitude'])
=======
gdf = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.longitude, data.latitude))

print(gdf.head())

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
norway = world.log[world['name'] == 'Norway'] 

ax = world[world.continent == 'Europe'].plot()
gdf.plot(ax=ax, color = 'red')
>>>>>>> ec1da845b99d57894b82eeb0340942477fd335a3
plt.show()


