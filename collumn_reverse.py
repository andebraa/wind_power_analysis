import pandas as pd 

data = pd.read_csv('anotized_data.csv', usecols = ['text', 'label']) 

reverse_data = data[data.columns[::-1]] 
reverse_data = reverse_data.rename(columns = {'label': 'labels'})

print(reverse_data.head())

reverse_data.to_csv('reverse_anotized_data.csv', index = False)
