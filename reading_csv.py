"""
Script for reading csv files made by lese_txt.py. The order of csv collumns is very ridgid.
NOTE: assumes week 1 is the start week for all datasets
"""
import re
import json
import csv
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np

test_hist = pd.read_csv('twitterdata.csv', parse_dates=True)

#reading, parsing and sorting time elements from twitter data
tiems = test_hist['created_at']
tiems= pd.to_datetime(tiems, errors='coerce', format = "%Y-%m-%dT%H:%M:%S.%fZ")
tiems_sorted = tiems.sort_values()

week_nums = pd.date_range(tiems_sorted.iloc[0], tiems_sorted.iloc[-1], freq='W-MON')
#year_nums = pd.date_range(tiems_sorted.iloc[0].year, tiems_sorted.iloc[-1].year, freq='YS')
year_nums = np.arange(tiems_sorted.iloc[0].year, tiems_sorted.iloc[-1].year +1)

print('twat')
print(year_nums)

year_indx_dict = {} 
for i, elem in enumerate(year_nums): #make a dictionary containing year and corresponding index 
    print(i)
    year_indx_dict[elem] = i
print(year_indx_dict)

start_year = tiems_sorted.iloc[0].year
end_year = tiems_sorted.iloc[-1].year
print(start_year)
print(end_year)

#test_hist['created_at'].groupby(test_hist["created_at"].dt.week).count()#.plot(kind="bar")

#bins = int(np.ceil(np.log(n)+1))
occurences = np.zeros(len(week_nums)+1)

print('twat')
for i, elem in enumerate(tiems_sorted):
    print(int(elem.week + (year_indx_dict[elem.year]*52)) -1)

    occurences[int(elem.week + (year_indx_dict[elem.year]*52)) -1]  += 1 
            #the number of year times 52 ensures indexing goes beyond 52 for the subsequent years

print(occurences)
print(week_nums)
print(len(occurences))
print(len(week_nums))
plt.plot(week_nums, occurences[:-1])
plt.ylabel('number of tweets')
plt.title('frequency of tweets with geodata in the last election cycle')

plt.savefig('first_elec_cycle.jpg', bbox_inches = 'tight', pad_inches = 0.001) #0.1 is default when bbox is tight

