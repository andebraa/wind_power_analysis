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

test_hist = pd.read_csv('all_data_all_time_edited.csv', parse_dates=True)

#reading, parsing and sorting time elements from twitter data
tiems = test_hist['created_at']
tiems= pd.to_datetime(tiems, errors='coerce', format = "%Y-%m-%dT%H:%M:%S.%fZ")
tiems_sorted = tiems.sort_values()


week_nums = pd.date_range(tiems_sorted.iloc[0], tiems_sorted.iloc[-1], freq='W-MON')
#year_nums = pd.date_range(tiems_sorted.iloc[0].year, tiems_sorted.iloc[-1].year, freq='YS')
year_nums = np.arange(tiems_sorted.iloc[0].year, tiems_sorted.iloc[-1].year +1)


print(tiems_sorted.iloc[-1])
print(tiems_sorted.iloc[0])

        
year_indx_dict = {} 
for i, elem in enumerate(year_nums): #make a dictionary containing year and corresponding index 
    year_indx_dict[elem] = i

start_year = tiems_sorted.iloc[0].year
end_year = tiems_sorted.iloc[-1].year

#test_hist['created_at'].groupby(test_hist["created_at"].dt.week).count()#.plot(kind="bar")

#bins = int(np.ceil(np.log(n)+1))
occurences = np.zeros(len(week_nums)+1)
print(len(week_nums))

print('twat')
for i, elem in enumerate(tiems_sorted):
    #NOTE: datetime.date(elem.year,29,12).isocalendar()[1]) is to exctract the total number of weeks in a given year. 
    #this is because 2020 had 53 weeks and this fucked my code
    #print(int(elem.week + (year_indx_dict[elem.year]*datetime.date(elem.year,12,29).isocalendar()[1]) -1))
    #print(elem.year, elem.week)
    #print(elem)
    doy = elem.day_of_year
    dow = elem.day_of_week
    woy = ((10 + doy - dow) //7) -1 #https://en.wikipedia.org/wiki/ISO_week_date#Differences_to_other_calendars
    #print('woy')
    #print(woy)
    occurences[int(woy + (year_indx_dict[elem.year]*datetime.date(elem.year,12,29).isocalendar()[1]) )]  += 1 
            #the number of year times 52 ensures indexing goes beyond 52 for the subsequent years
            # calculating woy is due to ISO calendar not ending the year at first of january
print(occurences)
print(week_nums)
print(len(occurences)) 
print(len(week_nums))
fig, ax = plt.subplots()

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(8)

ax.plot(week_nums, occurences[:-1])
plt.ylabel('number of tweets')
plt.title('frequency of tweets with geodata in the last election cycle')

plt.savefig('tweets_per_week_2006_and_up.jpg', bbox_inches = 'tight', pad_inches = 0.1) #0.1 is default when bbox is tight




