import pandas as pd 
import numpy as np

data = pd.read_csv('anotized_combined_200.csv', usecols =['text', 'label', 'labels']) 

label = np.zeros(len(data)) 

outdata = pd.DataFrame()

for i, row in data.iterrows():
    if isinstance(row['label'], float):
        label[i] = row['label'] 
    elif isinstance( row['labels'], float):
        label[i] = row['labels'] 

outdata['text'] = data['text'] 
outdata['labels'] = label 

outdata = outdata.astype({'labels':'int'})

outdata.to_csv('anotized_200_x.csv', index=False)
