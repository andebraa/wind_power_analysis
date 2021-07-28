import re
import json
import csv
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np

file = open('all_data_all_time.txt')
tekst = file.read()


tweets = tekst.split('\n\n')
tweets.pop()
print(len(tweets)) 
header = ['username', 'text', 'language', 'loc', 'created_at']
elements = 0
no_tweetinfo = 0
no_geodata = 0
lost_tweets = 0 
with open('all_data_all_time_edited.csv', 'w+', encoding='UTF8', newline='') as file_2:
    writer = csv.writer(file_2)
    writer.writerow(header)
    times = []
    for tweet in tweets[5:]:
        success = False
        tweet_info = []
        element = json.loads(tweet)
        elements += 1 
        times.append(datetime.datetime.strptime(element['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ").time())
    
        try:
            tweet_info.extend((element['user']['username'], element['text'], element['lang'], element['place']['name'], element['created_at']))
            writer.writerow(tweet_info)
            print(element['lang'])
            success = True
        except:
            no_tweetinfo += 1
            pass
        if success != True:
            try:
                """
                Handling missing tweet location info by adding user location info instead. 
                """
                tweet_info.extend((element['user']['username'], element['text'], element['lang'], element['user']['location'], element['created_at']))
                writer.writerow(tweet_info)
            except:
                no_geodata += 1
                pass 
        if success != True: #still haven't found any data, this tweet is a loss   
            lost_tweets += 1 
            pass


file.close()
print('no_tweetinfo:')
print(no_tweetinfo)
print('no_geodata')
print(no_geodata)
print('lost_tweets:')
print(lost_tweets)
print('elements:')
print(elements)
file_2.close()
