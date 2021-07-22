import re
import json
import csv
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np

file = open('full_query_election.txt')
tekst = file.read()

tweets = tekst.split('\n\n')
tweets.pop()

skipped = 0 #number of skipped entries, which have no element[place][name] 
header = ['username', 'text', 'language', 'loc', 'created_at']
elements = 0 
with open('twitterdata.csv', 'w+', encoding='UTF8', newline='') as file_2:
    writer = csv.writer(file_2)
    writer.writerow(header)
    times = []
    for tweet in tweets[5:]:
        tweet_info = []
        element = json.loads(tweet)
        elements += 1 
        times.append(datetime.datetime.strptime(element['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ").time())

        try:
            tweet_info.extend((element['user']['username'], element['text'], element['lang'], element['place']['name'], element['created_at']))
            writer.writerow(tweet_info)
            print(element['lang'])
        except:
            """
            Handling missing tweet location info by adding user location info instead. 
            """
            print(tweet)
            tweet_info.extend((element['user']['username'], element['text'], element['lang'], element['user']['location'], element['created_at']))
            writer.writerow(tweet_info)
            skipped += 1 


file.close()
print( skipped)
print(elements)
file_2.close()
