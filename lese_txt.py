"""
Script that takes the txt formated output data of search.py and converts it to a csv.
later this is read by the script reading_csv.py

If the tweet element doesn't have element['place']['name'], then it is attempted
to access the element['user']['location']. 
i.e. If the tweet element doesn't have geo location, then we look at the geo location
of the user instead. This is manual input and does not have to be a real place. This
is handled in geo_locating. 
"""

import re
import json
import csv
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np

file = open('data/fourth_rendition_data/all_data_all_time.txt')
tekst = file.read()


tweets = tekst.split('\n\n')
tweets.pop()
print(len(tweets)) 
header = ['id','username', 'text', 'loc', 'created_at', 'like_count', 'quote_count']
elements = 0
no_tweetinfo = 0
no_geodata = 0
tot_tweets = 0
with open('data/fourth_rendition_data/fourth_rendition_output_id.csv', 'w+', encoding='UTF8', newline='') as file_2:
    #print('open')
    writer = csv.writer(file_2)
    writer.writerow(header)
    times = []
    for tweet in tweets[5:]:
        success = False
        tweet_info = []
        element = json.loads(tweet)
        elements += 1 
        times.append(datetime.datetime.strptime(element['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ").time()) 
        #tweet_info.extend((element['user']['username'], element['text'],  element['place']['name'], element['created_at'], element['public_metrics']['like_count'], element['public_metrics']['quote_count']))

        if 'place' not in element.keys():
            no_tweetinfo += 1
            try:
                tweet_info.extend((element['id'], element['user']['username'], element['text'], element['user']['location'], element['created_at'], element['public_metrics']['like_count'], element['public_metrics']['quote_count']))
                writer.writerow(tweet_info)
                tot_tweets +=1
            except:
                no_geodata +=1
        else:
            tweet_info.extend((element['id'], element['user']['username'], element['text'], element['place']['name'], element['created_at'], element['public_metrics']['like_count'], element['public_metrics']['quote_count']))
            writer.writerow(tweet_info)
            tot_tweets += 1


file.close()
print('no_tweetinfo:')
print(no_tweetinfo)
print('no_geodata')
print(no_geodata)
print('elements:')
print(elements)
print('tot_tweets')
print(tot_tweets)
file_2.close()
