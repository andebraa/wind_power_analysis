
import pandas as pd

data = pd.read_csv('anotized_data_100_2.csv', usecols = ['labels', 'text']) 

data = data.rename(columns = {'labels': 'label'}) 

data.to_csv('anotized_data_100_2.csv', index=False)
