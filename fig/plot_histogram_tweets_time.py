"""
Script for reading csv files made by lese_txt.py. The order of csv collumns is very ridgid.
"""
import re
import json
import csv
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np

test_hist = pd.read_csv('../data/third_rendition_data/third_rendition_output.csv', parse_dates=True)

#test_hist = test_hist[test_hist['labels'] == 1] 
#reading, parsing and sorting time elements from twitter data
tiems = test_hist['created_at']
tiems= pd.to_datetime(tiems, errors='coerce', format = "%Y-%m-%dT%H:%M:%S.%fZ")
tiems_sorted = tiems.sort_values()


print(tiems_sorted)
week_nums = pd.date_range(tiems_sorted.iloc[0], tiems_sorted.iloc[-1], freq='W-MON')
year_nums = np.arange(tiems_sorted.iloc[0].year, tiems_sorted.iloc[-1].year +1)
print(week_nums)

print(tiems_sorted.iloc[-1])
print(tiems_sorted.iloc[0].week)

        
year_indx_dict = {} 
for i, elem in enumerate(year_nums): #make a dictionary containing year and corresponding index 
    year_indx_dict[elem] = i

start_year = tiems_sorted.iloc[0].year
end_year = tiems_sorted.iloc[-1].year
long_years = [2009, 2015, 2020]

occurences = np.zeros((len(year_indx_dict), 53))
print(np.shape(occurences))
long_weeks = 0

#TODO:  Check week num and year in week_nums array and elem to find matching index. 
for i, elem in enumerate(tiems_sorted):
    """
    NOTE: datetime.date(elem.year,29,12).isocalendar()[1]) is to exctract the total number of weeks in a given year. 
    this is because 2020 had 53 weeks and this fucked my code
    """
    #print(int(elem.week + (year_indx_dict[elem.year]*datetime.date(elem.year,12,29).isocalendar()[1]) -1))
    
    #bool_list = week_nums.isin(np.array([date]).astype('datetime64[ns]'))
    #print(bool_list)
    
    #print(datetime.date(elem.year,12,29).isocalendar()[1])
    #doy = elem.day_of_year
    #dow = elem.day_of_week
    #woy = ((10 + doy - dow) //7) -1 #https://en.wikipedia.org/wiki/ISO_week_date#Differences_to_other_calendars
    if elem.week == 53 and elem.year in long_years:
        occurences[year_indx_dict[elem.year], elem.week -2] += 1 
        print(elem)
        long_weeks += 1
        pass 

        
    #print(elem.year, elem.week-1)  
    occurences[year_indx_dict[elem.year], elem.week-1] += 1 
    #occurences[int(woy + (year_indx_dict[elem.year]*datetime.date(elem.year,12,29).isocalendar()[1]) )]  += 1 
            #the number of year times 52 ensures indexing goes beyond 52 for the subsequent years
            # calculating woy is due to ISO calendar not ending the year at first of january

flat = occurences.flatten()
print(long_weeks)
fig, ax = plt.subplots()
first_tweet_week = tiems_sorted.iloc[0].week -1 #-1 for index
flat = flat[first_tweet_week:]
#print(flat)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(8)
#print(week_nums)
#print(flat)
print(occurences) 
print(occurences[year_indx_dict[2015], 52])
print(occurences[year_indx_dict[2009], 52])
print(occurences[year_indx_dict[2020], 52])

print('zeros')
print(flat[154])
print(flat)


ax.plot(week_nums, flat[:len(week_nums)])
plt.ylabel('number of tweets')
plt.title('tweets per week')

plt.savefig('tweet_per_week_third_rendition_output.jpg', bbox_inches = 'tight', pad_inches = 0.1, dpi = 300) #0.1 is default when bbox is tight





