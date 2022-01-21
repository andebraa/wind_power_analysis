"""
script that analyses the users and the frequency of tweets etc
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data/second_rendition_data/second_rendition_geolocated.csv')
#username,text,loc,created_at,like_count,quote_count,city,latitude,longitude
user_dict = {} 

def make_json(data, save=False):

    for i, elem in data.iterrows():
        if elem['username'] not in user_dict.keys:
            user_dict[elem['username'].to_string()] = [[elem['text'], 
                                                       elem['loc'],
                                                       elem['created_at'],
                                                       elem['like_count'],
                                                       elem['quote_count'],
                                                       elem['city'],
                                                       elem['latitude'],
                                                       elem['longitude']]]
        else:
            user_dict[elem['username']].append([elem['text'],
                                                       elem['loc'],
                                                       elem['created_at'],
                                                       elem['like_count'],
                                                       elem['quote_count'],
                                                       elem['city'],
                                                       elem['latitude'],
                                                       elem['longitude']])
    if save:
        with open('second_rendition_geolocated_usernamedict.json' ,'w') as fp:
            json.dump(user_dict, fp) 

    return user_dict 

if __name__ == '__main__':
    make_json(data, save=True)

