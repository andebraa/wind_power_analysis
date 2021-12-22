"""
Short script that takes the last 10000 tweets, adds a number indicating 
number of days since first datapoint, and writes this to a new csv

"""
import pandas as pd
import numpy as np

data = pd.read_csv('data/full_geodata_longlat_noforeign_anonymous.csv',
                   usecols = ['username', 'text', 'created_at'],
                   index_col = False)



part = data.tail(10000) 
part.loc[:,'created_at'] = part.loc[:,'created_at'].apply(pd.to_datetime) 

start_date = min(part['created_at'])
end_date = max(part['created_at'])



days = np.zeros(len(part))

days_dict = {}

for i, elem in part.iterrows():
    curr_day = elem['created_at']
    if curr_day not in days_dict:
        num_days = (curr_day - start_date).days
        days_dict[curr_day]= num_days

j = 0
for i,elem in part.iterrows():
    days[j] = days_dict[elem['created_at']]
    j +=1
    
part['days'] = days

print(part.head())
print(part.tail())

part.to_csv('natalia_slice_first_rendition_anon.csv')
