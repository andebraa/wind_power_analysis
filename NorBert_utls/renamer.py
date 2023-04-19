import pandas as pd 

filename = 'anotized_data_100_2.csv'
data = pd.read_csv(filename, usecols = ['label', 'text']) 

if 'label' in data.columns:
    data = data.rename(columns = {'label': 'labels'})
    print('label now labels') 
    data.to_csv(filename, index = False) 
else:
    print('labels is labels')
