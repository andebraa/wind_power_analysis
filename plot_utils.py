"""
Script for plotting uncertainty hist (histogram of max values of labels) 
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import geopandas
import re
import math

#plt.style.use('ggplot')


def plot_uncertainty_hist(data: pd.DataFrame) -> None:

    bool_label = data['label'].values
    print(len(bool_label[bool_label == 1]))
    print(len(bool_label[bool_label == 0]))

    logits = data[['logits0', 'logits1']]

    #max_logits = logits.max(axis=1) 
    #min_logits = logits.min(axis=1) 

    binwidth = 0.005
    #plt.hist(max_logits, bins=np.arange(min(max_logits), max(max_logits) + binwidth, binwidth), 
    #         alpha = 0.5, label = 'highest values')
    #plt.hist(min_logits, bins=np.arange(min(max_logits), max(max_logits) + binwidth, binwidth), 
    #         alpha = 0.5, label = 'lowest values')
    logits0 = data['logits0'] 
    logits1 = data['logits1'] 
    plt.hist(logits0, bins=np.arange(min(logits0), max(logits0) + binwidth, binwidth), 
             alpha = 0.5, label = 'negative logits')
    plt.hist(logits1, bins=np.arange(min(logits1), max(logits1) + binwidth, binwidth), 
             alpha = 0.5, label = 'positive logits')
    
    plt.legend()
    plt.title('Distribution of positive and negative logits')
    plt.show()
    #plt.savefig('fig/uncertainty_hist.png')



def norway_plot():
    gdf = geopandas.GeoDataFrame(data, geometry=geopandas.points_from_xy(data.longitude, data.latitude))


    fig, ax = plt.subplots() 
    norway.plot(ax = ax)#, alpha = 0.4)

    #dispersion = np.random.normal( scale = 10 ,size = (len(gdf['labels']), len(gdf['labels'])))
    for i,row in gdf.iterrows():
        print(row['latitude'])
        print('add')
        row['latitude'] += np.random.normal(scale=100) 
        row['longitude'] += np.random.normal(scale=100)
        print(row['latitude'])

    gdf[gdf['labels'] == 0].plot(ax = ax, markersize = 20, color = 'red', marker = 'o', label = 'neg')
    #print(gdf.iloc[bool_label])
    gdf[gdf['labels']==1].plot(ax = ax, markersize = 20, color = 'blue', marker = '^', label = 'pos') 
    plt.legend()
    plt.title('distribution of positive and negative sentiment across the country')
    plt.savefig('fig/geo_distribution.png')


def district_bar_plot():

    print(data.head())

def logits_to_bool():
    '''
    First rendition has only labels as (logits0, logits1). to convert 
    these to bool use this script (legacy)
    '''
    delete_rows = []
    max_label = np.zeros(len(data))
    bool_label = np.zeros(len(data))
    label_diff = np.zeros(len(data))
    for i, row in data.iterrows():
        try:    
            search = re.search(r"\[\s*((?:-){0,1}\d.\d+(?:e-\d+)*)\s+((?:-){0,1}\d.\d+(?:e-\d+)*)\s*\]", row['labels']).groups() 
            labels = (float(search[0]), float(search[1]) )
            max_label[i] = np.max(labels)
            label_diff[i] = np.abs(labels[0] - labels[1]) 
            if labels[0] < labels[1]:
                bool_label[i] = int(1) 
            
        except:
            print("delete row {}".format(i))
            delete_rows.append(i) #rows with nan labels are saved for later deletion

    data.drop(data.index[delete_rows]) # delete rows with nan indexes
    data['labels'] = bool_label

    data.to_csv('final_dataset_bolllabel.csv', index = False) 

    print(np.sum(bool_label)) 
    print(len(bool_label))

if __name__ == '__main__':
    norway = geopandas.read_file('kommuner_komprimert.json')
    path = '/home/andebraa/wind_power_analysis/data/second_rendition_data/second_rendition_geolocated_noemoji_anonymous_predict.csv'


    data = pd.read_csv(path,
                      usecols = ['username','text','loc','created_at',
                                 'like_count','quote_count','latitude',
                                 'longitude', 'label', 'logits0', 'logits1']
                      )

    print(data.head())
    #plot_uncertainty_diff()
    plot_uncertainty_hist(data)
    #norway_plot()
    #pass
