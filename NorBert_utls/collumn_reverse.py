import pandas as pd 

filename = 'anotized_data_200_5.csv'

data = pd.read_csv(filename, usecols = ['text', 'labels']) 

reverse_data = data[data.columns[::-1]] 

print(reverse_data.head())

reverse_data['labels'] = reverse_data['labels'].astype('int')

reverse_data.to_csv('reverse_'+filename, index = False)
