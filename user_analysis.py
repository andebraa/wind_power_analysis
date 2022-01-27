
"""
script that analyses the users and the frequency of tweets etc
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

data = pd.read_csv('data/second_rendition_data/second_rendition_geolocated_noemoji_anonymous.csv',
                   parse_dates = True)
#username,text,loc,created_at,like_count,quote_count,city,latitude,longitude
user_dict = {}

def make_json(data, save=False):

    for i, elem in data.iterrows():
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


def plot_user_freq(data):
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


    users = len(user_dict)
    user_freq = np.zeros((users,2))
    user_freq_names = []
    for i,user in enumerate(user_dict):
        user_freq[i,0] = len(user_dict[user])
        user_freq_names.append(user)


    user_freq[:,1] = np.arange(users).astype('int') #make a sort of index collumn
    sort_userfreq = user_freq[user_freq[:,0].argsort()]
    top_80 = sort_userfreq[int(users*0.8):,0]

    top_80_users = [user_freq_names[int(i)] for i in sort_userfreq[int(users*0.8):,0]]
    print(top_80_users)
    tweet_threshold = sort_userfreq[int(users*0.8):,1] #the amount of tweets the 80% has 


    """
    plt.subplot(3,1,1)
    plt.plot(user_freq[:,0])
    plt.subplot(3,1,2)
    plt.yscale('log')
    plt.plot(sort_userfreq[:,0])
    print(sort_userfreq)
    plt.subplot(3,1,3)
    plt.yscale('log')
    plt.plot(top_80)
    plt.show()
    
    """
    image = np.zeros((num_weeks+1, top_80_users))

    week_dict = {}
    for i, date_i in enumerate(start_date + timedelta(n*7) for n in range(num_weeks+1)):
        week_dict[date_i.isocalendar()[:-1]] = i #year and week of date

    for i, user in enumerate(user_dict):
        if len(user_dict[user] > user_threshold):
            continue
        for j, tweet in enumerate(user_dict[user]):
            #.isocalendar returns (year, week, day) and we only want year, week
            image[week_dict[pd.to_datetime(tweet[2]).isocalendar()[:-1]],i] +=1

    plt.imshow(np.transpose(image))
    plt.show()


if __name__ == '__main__':
    make_json(data, save=True)
    plot_user_freq(data)


