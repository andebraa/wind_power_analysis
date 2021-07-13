import re
import json
import csv
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np

file = open('vindkraft_ny.txt')
tekst = file.read()

tweets = tekst.split('\n\n')
tweets.pop()


header = ['author_id', 'text', 'language', 'loc', 'created_at']
with open('twitterdata.csv', 'w+', encoding='UTF8', newline='') as file_2:
    writer = csv.writer(file_2)
    writer.writerow(header)
    times = []
    for tweet in tweets[5:]:
        tweet_info = []
        element = json.loads(tweet)

        times.append(datetime.datetime.strptime(element['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ").time())

        #child = element['place']
        #print(type(child))
        #print(child['name'])
        #print('hello')
        tweet_info.extend((element['author_id'], element['text'], element['lang'], element['place']['name'], element['created_at']))
        writer.writerow(tweet_info)



file.close()
file_2.close()
