import pandas as pd 
import matplotlib.pyplot as plt
import geopandas

## from url
url  = 'https://raw.githubusercontent.com/robhop/fylker-og-kommuner-2020/master/Kommuner-small.json'
norway = pd.read_json(url)

#save github to local
#norway.to_json('github_nowray_kommuner.json')

data = pd.read_csv('full_geodata_longlat_noforeign_anonymous.csv', usecols = ['text', 'latitude', 'longitude']) 
## from local file
#norway = geopandas.read_file('kommuner_komprimert.json')

#data['labels'] = np.random.randint(0,2,size = (len(data)))

print(data.head())

fig, ax = plt.subplots()

gdf = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.longitude, data.latitude))
gdf.plot(ax = ax, markersize = 20, color = 'red', marker = 'o', label = 'occurance')


norway.plot(ax = ax, alpha = 0.4)

plt.show()
