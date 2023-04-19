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

def make_json(data, save=False):
    '''
    script for making a dictionary containing users and their tweets
    '''
    user_dict = {}

    for i, elem in data.iterrows():
        assert elem['username'] != '' and not pd.isnull(elem['username'])
        if elem['username'] not in user_dict:
            user_dict[elem['username']] = [[elem['text'],
                                            elem['loc'],
                                            elem['created_at'],
                                            elem['like_count'],
                                            elem['quote_count'],
                                            elem['latitude'],
                                            elem['longitude'],
                                            elem['label'],
                                            elem['logits0'],
                                            elem['logits1']]]
        else:
            #nnamed: 0,id,username,text,loc,created_at,like_count,quote_count,latitude,longitude,label,logits0,logits1
            user_dict[elem['username']].append([elem['text'],
                                                elem['loc'],
                                                elem['created_at'],
                                                elem['like_count'],
                                                elem['quote_count'],
                                                elem['latitude'],
                                                elem['longitude'],
                                                elem['label'],
                                                elem['logits0'],
                                                elem['logits1']])
    if save:
        with open('fourth_rendition_geolocated_anonymous_usernamedict.json' ,'w') as fp:
            json.dump(user_dict, fp)

    return user_dict

def make_user_freq_list(data):
    user_dict = make_json(data)

    times = pd.to_datetime(data['created_at'])
    times_sorted = times.sort_values()
    start_date = times_sorted.iloc[0]
    end_date = times_sorted.iloc[-1]

    num_days = (end_date - start_date).days
    num_weeks = int(num_days/7) #assuming this works. probarbly won't

    date_range = pd.date_range(start_date, end_date, freq = 'w')


    num_users = len(user_dict)
    user_freq_dict = {}
    user_freq = np.zeros((num_users, 2))
    for i,user in enumerate(user_dict):
        print(i, user)
        user_freq_dict[user] = len(user_dict[user])
        user_freq[i, 1] = user #NOTE asumes the uname is int
        user_freq[i,0] = len(user_dict[user])


    sort_userfreq = user_freq[user_freq[:,0].argsort()]
    return sort_userfreq, user_dict

test_hist = pd.read_csv('../data/fourth_rendition_data/fourth_rendition_geolocated_id_posneutral_anonymous_predict.csv', parse_dates=True)
print(type(test_hist))
userfreq, user_dict = make_user_freq_list(test_hist)
print(len(user_dict))
stop

#reading, parsing and sorting time elements from twitter data
tiems = test_hist['created_at']
tiems= pd.to_datetime(tiems, errors='coerce', format = "%Y-%m-%dT%H:%M:%S.%fZ")
tiems_sorted = tiems.sort_values()


week_nums = pd.date_range(tiems_sorted.iloc[0], tiems_sorted.iloc[-1], freq='W-MON')
year_nums = np.arange(tiems_sorted.iloc[0].year, tiems_sorted.iloc[-1].year +1)

        
year_indx_dict = {} 
for i, elem in enumerate(year_nums): #make a dictionary containing year and corresponding index 
    year_indx_dict[elem] = i

start_year = tiems_sorted.iloc[0].year
end_year = tiems_sorted.iloc[-1].year
long_years = [2009, 2015, 2020]

occurences = np.zeros((len(year_indx_dict), 53))
long_weeks = 0

#TODO:  Check week num and year in week_nums array and elem to find matching index. 
for i, elem in enumerate(tiems_sorted):
    """
    NOTE: datetime.date(elem.year,29,12).isocalendar()[1]) is to exctract the total number of weeks in a given year. 
    this is because 2020 had 53 weeks and this fucked my code
    """
    
    #bool_list = week_nums.isin(np.array([date]).astype('datetime64[ns]'))
    
    #doy = elem.day_of_year
    #dow = elem.day_of_week
    #woy = ((10 + doy - dow) //7) -1 #https://en.wikipedia.org/wiki/ISO_week_date#Differences_to_other_calendars
    if elem.week == 53 and elem.year in long_years:
        occurences[year_indx_dict[elem.year], elem.week -2] += 1 
        long_weeks += 1
        pass 

        
    occurences[year_indx_dict[elem.year], elem.week-1] += 1 
    #occurences[int(woy + (year_indx_dict[elem.year]*datetime.date(elem.year,12,29).isocalendar()[1]) )]  += 1 
            #the number of year times 52 ensures indexing goes beyond 52 for the subsequent years
            # calculating woy is due to ISO calendar not ending the year at first of january

def remove_most_active(occurence_arr, orig_data, percentile, userfreq, year_index_dict):
    '''
    fetches most active users from make user freq list, loops thorugh
    these and removes occurences from the weeks of the year in which they tweet.
    occurence_ar (np array): array containing number of occurences per week
    orig_data (pandas dataframe): the original dataframe
    '''
    occurence_trim = occurence_arr.copy()
    most_active, user_dict = make_user_freq_list(orig_data)
    top_percent = most_active[int(len(most_active)*percentile):,:]
    for uname, values in user_dict.items():
        if uname in top_percent[:,1]:
            for tweet in values:
                created_at = pd.to_datetime(tweet[2])
                occurence_trim[year_index_dict[created_at.year], created_at.week-1] -= 1
    return occurence_trim


occurences_trim_99 = remove_most_active(occurences, test_hist, 0.99, userfreq, year_indx_dict) 
occurences_trim_95 = remove_most_active(occurences, test_hist, 0.95, userfreq, year_indx_dict) 
occurences_trim_90 = remove_most_active(occurences, test_hist, 0.90, userfreq, year_indx_dict) 
occurences_trim_85 = remove_most_active(occurences, test_hist, 0.85, userfreq, year_indx_dict) 
    
flat_trim_99 = occurences_trim_99.flatten()
flat_trim_95 = occurences_trim_95.flatten()
flat_trim_90 = occurences_trim_90.flatten()
flat_trim_85 = occurences_trim_85.flatten()

flat = occurences.flatten()
fig, ax = plt.subplots()
#ax = ax.ravel()

first_tweet_week = tiems_sorted.iloc[0].week -1 #-1 for index
flat = flat[first_tweet_week:]
flat_trim_99 = flat_trim_99[first_tweet_week:]
flat_trim_95 = flat_trim_95[first_tweet_week:]
flat_trim_90 = flat_trim_90[first_tweet_week:]
flat_trim_85 = flat_trim_85[first_tweet_week:]

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(8)


ax.plot(week_nums, flat[:len(week_nums)], label='full data', alpha=0.7, lw=1)
ax.plot(week_nums, flat_trim_95[:len(week_nums)], label='95%', alpha = 0.6, lw=0.8)
ax.plot(week_nums, flat_trim_90[:len(week_nums)], label='90%', alpha = 0.6, lw=0.8)
ax.plot(week_nums, flat_trim_85[:len(week_nums)], label='85%', alpha = 0.6, lw=0.8)
ax.plot(week_nums, flat_trim_99[:len(week_nums)], label='99%', alpha = 0.6, lw=0.8)
plt.legend()
plt.ylabel('number of tweets')
plt.title('tweets per week')

plt.savefig('tweet_per_week_fourth_rendition_geolocated_trimmed2.png', format='png', bbox_inches = 'tight', pad_inches = 0.1, dpi = 300) #0.1 is default when bbox is tigh





