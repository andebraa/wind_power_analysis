"""
Script for plotting uncertainty hist (histogram of max values of labels) 
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import geopandas
import re
import math

plt.style.use('ggplot')

norway = geopandas.read_file('kommuner_komprimert.json')

data = pd.read_csv('final_dataset.csv',
                  usecols = ['username','text','loc','created_at',
                             'like_count','quote_count','city','latitude',
                             'longitude', 'labels']
                  )

print(data.head())

delete_rows = []
max_label = np.zeros(len(data))
bool_label = np.zeros(len(data))
for i, row in data.iterrows():
    try:    
        search = re.search(r"\[\s*((?:-){0,1}\d.\d+(?:e-\d+)*)\s+((?:-){0,1}\d.\d+(?:e-\d+)*)\s*\]", row['labels']).groups() 
        labels = (float(search[0]), float(search[1]) )
        max_label[i] = np.max(labels)
                    #note; logits are [0, 1], i think|

        if labels[0] < labels[1]:
            bool_label[i] = 1 
        
    except:
        print("delete row {}".format(i))
        delete_rows.append(i) #rows with nan labels are saved for later deletion

data.drop(data.index[delete_rows]) # delete rows with nan indexes

def plot_uncertainty_hist():
    w = 3
    bins = math.ceil((np.max(labels) - np.min(labels))/w)
    plt.hist(max_label)#, bins = bins)#, label="mean: {}".format(np.mean(max_label)))
    plt.legend()
    plt.title('Distribution of values returned by NorBert')
    plt.savefig('fig/uncertainty_hist.png')



def norway_plot():
    gdf = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.longitude, data.latitude))


    fig, ax = plt.subplots() 
    norway.plot(ax = ax)#, alpha = 0.4)



    #gdf[bool_label['labels']].plot(ax = ax, markersize = 20, color = 'red', marker = 'o', label = 'neg')
    gdf[bool_label['labels']].plot(ax = ax, markersize = 20, color = 'blue', marker = '^', label = 'pos') 
 

    plt.savefig('fig/geo_distribution.png')

if __name__ == '__main__':
    norway_plot()
