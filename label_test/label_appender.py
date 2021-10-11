#script for taking labels csv and adding it to a dataset 
#NOT FUNCTIONAL
import re
import numpy as np
import pandas as pd 

dataset = pd.read_csv('../data/full_geodata_longlat_noforeign.csv', 
                      usecols = ['username','text','loc','created_at','like_count','quote_count','city','latitude','longitude']) 

labels = pd.read_csv('labels_output_test.csv') 
print(type(labels))
print(type(labels.iloc[0]))
labels_float = []
for i, row in labels.iterrows():
    search = re.search(r"\[\s*((?:-){0,1}\d.\d+(?:e-\d+)*)\s+((?:-){0,1}\d.\d+(?:e-\d+)*)\s*\]", row).groups()
    labels_float.append(search) 

print(labels_float)

#getting boolean label values
delete_rows = []
bool_label = np.zeros(len(df_twitter_sentiment))

for i, row in labels.iterrows():
    try:
        search = re.search(r"\[\s*((?:-){0,1}\d.\d+(?:e-\d+)*)\s+((?:-){0,1}\d.\d+(?:e-\d+)*)\s*\]", row['labels']).groups()
        labels = (float(search[0]), float(search[1]) )
        if labels[0] < labels[1]:
            bool_label[i] = 1

    except:
            print("delete row {}".format(i))
            delete_rows.append(i) #rows with nan labels are saved for later deletion

            df_twitter_sentiment.drop(df_twitter.index[delete_rows]) # delete rows with nan indexes



#adding boolean label to dataframe
df_twitter_sentiment['label'] = bool_label
