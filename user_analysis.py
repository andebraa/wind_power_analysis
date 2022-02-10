"""
script that analyses the users and the frequency of tweets etc
"""
import json
import calmap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from datetime import datetime, timedelta

data = pd.read_csv('data/second_rendition_data/second_rendition_geolocated_noemoji_anonymous.csv',
                   parse_dates = True)
#username,text,loc,created_at,like_count,quote_count,city,latitude,longitude
user_dict = {}

def make_json(data, save=False):

    for i, elem in data.iterrows():
        assert elem['username'] != '' and not np.isnan(elem['username'])
        if elem['username'] not in user_dict:
            user_dict[elem['username']] = [[elem['text'],
                                            elem['loc'],
                                            elem['created_at'],
                                            elem['like_count'],
                                            elem['quote_count'],
                                            elem['latitude'],
                                            elem['longitude']]]
        else:
            user_dict[elem['username']].append([elem['text'],
                                                elem['loc'],
                                                elem['created_at'],
                                                elem['like_count'],
                                                elem['quote_count'],
                                                elem['latitude'],
                                                elem['longitude']])
    if save:
        with open('second_rendition_geolocated_noemoji_anonymous_usernamedict.json' ,'w') as fp:
            json.dump(user_dict, fp)

    return user_dict



def plot_user_freq(data, percentage = 0.90):
    """
    make a github green dot type plot over users and tweets on each axis.
    """
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
        user_freq_dict[user] = len(user_dict[user])
        user_freq[i, 1] = user #NOTE asumes the uname is int
        user_freq[i,0] = len(user_dict[user])


    sort_userfreq = user_freq[user_freq[:,0].argsort()]
    top_percent = sort_userfreq[int(num_users*percentage):,:]


    """ 
    plt.subplot(3,1,1)
    plt.plot(user_freq[:,0])
    plt.subplot(3,1,2)
    plt.yscale('log')
    plt.plot(sort_userfreq[:,0])
    print(sort_userfreq)
    plt.subplot(3,1,3)
    plt.yscale('log')
    plt.plot(top_percent)
    plt.show()
    """

    image = np.zeros((num_weeks+1, len(top_percent)))

    week_dict = {}

    #for i in range(num_weeks):
    #    ts = start_date + timedelta(days=7*i)
    #    week_dict[ts.year, ts.weekofyear] = i
    #print(week_dict)
    for i, date_i in enumerate(start_date + timedelta(n*7) for n in range(num_weeks+1)):
        week_dict[date_i.isocalendar()[:-1]] = i #year and week of date
    for i, user in enumerate(top_percent[:,1]):
        for j, tweet in enumerate(user_dict[user]):
            image[week_dict[pd.to_datetime(tweet[2]).isocalendar()[:-1]],i] +=1

    """
    for i, user in enumerate(user_dict):
        if user not in top_80[:,1]:
            continue
        for j, tweet in enumerate(user_dict[user]):
             
            #.isocalendar returns (year, week, day) and we only want year, week
            image[week_dict[pd.to_datetime(tweet[2]).isocalendar()[:-1]],i] +=1
    """
    fig, ax = plt.subplots()

    cax = ax.imshow(np.transpose(image), cmap = cm.seismic)
    ax.set_title(f'Tweets per week of top {percentage} users over time')
    ax.set_xticks(np.arange(len(week_dict)))
    ax.set_xticklabels(week_dict.keys())
    cbar = fig.colorbar(cax)
    #plt.imshow(np.transpose(image))
    plt.show()


if __name__ == '__main__':
    #make_json(data, save=True)
    plot_user_freq(data, percentage = 0.97)
    #TODO: 
