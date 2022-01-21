"""
script that analyses the users and the frequency of tweets etc
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/second_rendition_data/second_rendition_geolocated_noemoji_anonymous.csv')
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

    users = len(user_dict)
    user_freq = np.zeros((users,2)) 
    for i,user in enumerate(user_dict):
        user_freq[i,0] = len(user_dict[user])
    user_freq[:,1] = np.arange(users) #make a sort of index collumn
    print(user_freq)
    sort_userfreq = user_freq[user_freq[:,0].argsort()] 
    plt.plot(sort_userfreq[:,0], np.arange(len(sort_userfreq[:,1])))
    plt.savefig('test_sortuserfreq.png')
    print(sort_userfreq)
    top_80 = sort_userfreq[:int(users*0.8), :int(users*0.8)]

    


if __name__ == '__main__':
    plot_user_freq(data)

