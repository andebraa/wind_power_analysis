import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import geopandas
import re

norway = geopandas.read_file('kommuner_komprimert.json')

data = pd.read_csv('final_dataset.csv',
                  usecols = ['username','text','loc','created_at',
                             'like_count','quote_count','city','latitude',
                             'longitude', 'labels']
                  )

print(data.head())

delete_rows = []
max_label = np.zeros(len(data))
for i, row in data.iterrows():
    print(i, len(data))
    print(row)
    print(row['labels'])
    try:    
        search = re.search(r"\[\s*((?:-){0,1}\d.\d+(?:e-\d+)*)\s+((?:-){0,1}\d.\d+(?:e-\d+)*)\s*\]", row['labels']).groups() 
        labels = (float(search[0]), float(search[1]) )
        max_label[i] = np.max(labels) 
    except:
        delete_rows.append(i) #rows with nan labels are saved for later deletion

data.drop(data.index[delete_rows]) # delete rows with nan indexes


plt.hist(max_label, label="mean: {}".format(np.mean(max_label)))
plt.savefig('fig/uncertainty_hist.png')

gdf = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.longitude, data.latitude))


fig, ax = plt.subplots() 
norway.plot(ax = ax, alpha = 0.4)



gdf[gdf['labels'] == 0].plot(ax = ax, markersize = 20, color = 'red', marker = 'o', label = 'neg')
gdf[gdf['labels'] == 1].plot(ax = ax, markersize = 20, color = 'blue', marker = '^', label = 'pos') 
 

plt.show()


