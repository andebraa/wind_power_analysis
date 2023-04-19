"""
Script for removing emojis from text.
first attempts to remove most emojis with short regex RE_EMOJI, then
reads a list of known emojis from emoji_table.txt and compares.

NOTE: doesn't catch all cases for some reason.
"""
import re
import os
import pandas as pd
import numpy as np
import requests
import emoji

def strip_emoji(text):
    count = emoji.emoji_count(text)
    return re.sub(emoji.get_emoji_regexp(), r'', text), count  

data = pd.read_csv('second_rendition_data/second_rendition_geolocated.csv') 

for i, line in data.iterrows():
   data.iloc[[i]]['text'] = strip_emoji(data['text'].to_string())  
print(data['text'])
data.to_csv('noemoji_test.csv')
stop

#RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)

file_path = os.path.dirname(os.path.abspath(__file__))
emoji_key = pd.read_csv(file_path  + '/emoji_table.txt', usecols=['emoji'])#, encoding='utf-8', index_col=0)
print(emoji_key.head())
print(emoji_key.tail())

emoji_list = emoji_key['emoji'].to_list()

RE_EMOJI = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE)

#def strip_emoji(text):
#    res = RE_EMOJI.sub(r'', text)
#    return res

emoji_lines = 0
for i, line in data.iterrows():
    prev = line['text']
    line['text'] = strip_emoji(line['text'])
    for j in emoji_list:
        if str(j) in str(line['text']):
            line['text'].strip(j) 

    if line['text'] != prev:
        emoji_lines += 1
        print(prev)
        print(line['text'])
        print('\n \n')
    

data.to_csv('test_no_emoji.csv')
