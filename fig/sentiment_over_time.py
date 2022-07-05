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
plt.style.use('seaborn')

test_hist = pd.read_csv('../data/third_rendition_data/third_rendition_geolocated_anonymous_posneutral_predict.csv', parse_dates=True)

pos_tweets = test_hist[test_hist['label'] == 1]
neg_tweets = test_hist[test_hist['label'] == 0] 

tiems0 = neg_tweets['created_at']
tiems0= pd.to_datetime(tiems0, errors='coerce', format = "%Y-%m-%dT%H:%M:%S.%fZ")
tiems_sorted0 = tiems0.sort_values()

tiems1 = pos_tweets['created_at']
tiems1= pd.to_datetime(tiems1, errors='coerce', format = "%Y-%m-%dT%H:%M:%S.%fZ")
tiems_sorted1 = tiems1.sort_values()

week_nums1 = pd.date_range(tiems_sorted1.iloc[0], tiems_sorted1.iloc[-1], freq='W-MON')
year_nums1 = np.arange(tiems_sorted1.iloc[0].year, tiems_sorted1.iloc[-1].year +1)
        
week_nums0 = pd.date_range(tiems_sorted0.iloc[0], tiems_sorted0.iloc[-1], freq='W-MON')
year_nums0 = np.arange(tiems_sorted0.iloc[0].year, tiems_sorted0.iloc[-1].year +1)

year_indx_dict1 = {} 
year_indx_dict0 = {} 
for i, elem in enumerate(year_nums1): #make a dictionary containing year and corresponding index 
    year_indx_dict1[elem] = i

for i, elem in enumerate(year_nums0): #make a dictionary containing year and corresponding index 
    year_indx_dict0[elem] = i

start_year0 = tiems_sorted0.iloc[0].year
end_year0 = tiems_sorted0.iloc[-1].year

start_year1 = tiems_sorted1.iloc[0].year
end_year1 = tiems_sorted1.iloc[-1].year

long_years = [2009, 2015, 2020]

occurences0 = np.zeros((len(year_indx_dict0), 53))
occurences1 = np.zeros((len(year_indx_dict1), 53))
long_weeks0 = 0
long_weeks1 = 0

for i, elem in enumerate(tiems_sorted0):
    """
    NOTE: datetime.date(elem.year,29,12).isocalendar()[1]) is to exctract the total number of weeks in a given year. 
    this is because 2020 had 53 weeks and this fucked my code
    """
    if elem.week == 53 and elem.year in long_years:
        occurences0[year_indx_dict0[elem.year], elem.week -2] += 1 
        long_weeks0 += 1
        pass 

        
    occurences0[year_indx_dict0[elem.year], elem.week-1] += 1 
    #occurences[int(woy + (year_indx_dict[elem.year]*datetime.date(elem.year,12,29).isocalendar()[1]) )]  += 1 
            #the number of year times 52 ensures indexing goes beyond 52 for the subsequent years
            # calculating woy is due to ISO calendar not ending the year at first of january

for i, elem in enumerate(tiems_sorted1):
    if elem.week == 53 and elem.year in long_years:
        occurences1[year_indx_dict1[elem.year], elem.week -2] += 1 
        long_weeks1 += 1
        pass 

        
    occurences1[year_indx_dict1[elem.year], elem.week-1] += 1 


flat0 = occurences0.flatten()
flat1 = occurences1.flatten()
fig, ax = plt.subplots()

first_tweet_week0 = tiems_sorted0.iloc[0].week -1 #-1 for index
first_tweet_week1 = tiems_sorted1.iloc[0].week -1 #-1 for index
flat0 = flat0[first_tweet_week0:]
flat1 = flat1[first_tweet_week1:]

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(8)
N = 5
print(np.shape(flat0))
flat0 = np.convolve(flat0, np.ones(N)/N, mode = 'valid')
flat1 = np.convolve(flat1, np.ones(N)/N, mode = 'valid')
print(np.shape(flat0))

ax.plot(week_nums0, flat0[:len(week_nums0)], label='negative', color='yellow')
ax.plot(week_nums1, flat1[:len(week_nums1)], label='positive', color='blue')
plt.ylabel('number of tweets')
plt.title('tweets per week')
plt.legend()
plt.savefig('tweet_per_week_third_rendition_geolocated_predict.png', format='png', bbox_inches = 'tight', pad_inches = 0.1, dpi = 300) #0.1 is default when bbox is tight





