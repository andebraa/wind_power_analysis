import numpy as np
import io
import pandas as pd
import os
import torch
import tensorflow as tf
import sys, os
import re
import emoji

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    sys.stdout = sys.__stdout__

def strip_emoji(text):
    print(emoji.emoji_count(text))
    new_text = re.sub(emoji.get_emoji_regexp(), r"", text)
    return new_text

def cleaning(z):
       #y = re.sub(r"(?:\@|http?\://|https?\://|www)\S+|\#|\&\S+", "", z) # remove hyperlinks, @..., hashtags, symbols starting with &...
       y = re.sub(r'https?:\/\/\S*', '', z, flags=re.MULTILINE) # remove hyperlinks
       y = re.sub('@[^\s]+','',y)      # remove @ and usernames everywhere in the tweet
       y = re.sub('@',' ', y)          # remove stand-alone separately
       y = re.sub('\#[\w\_]+',' ', y)  # remove hashtags
       y = re.sub('\#',' ', y)          # remove stand-alone hashtag separately
       y = re.sub('http',' ', y)       # remove http separately if RTs are kept
       y = re.sub('https',' ', y)      # remove http separately if RTs are kept
       y = strip_emoji(y)              # remove emojis
       y = re.compile('RT').sub('', y) # remove RT everywhere in the tweet
       y = " ".join(y.split())         # put sentence parts back together without /n separator
       return(y)

dfRT = pd.read_csv('annotaion_3000_012label.csv')
dfRT = dfRT.loc[1:3001, ['text', 'label']]

neutral=dfRT.loc[dfRT['label'] == '0']
positive=dfRT.loc[dfRT['label'] == '1']
negative=dfRT.loc[dfRT['label'] == '-1']

dfRT['text'] = dfRT['text'].map(cleaning)

###replacing labels
dfRT['label'].mask(dfRT['label'] == '-1', 1, inplace=True)
dfRT['label'].mask(dfRT['label'] == '0', 2, inplace=True)
dfRT['label'].mask(dfRT['label'] == '1', 3, inplace=True)

dfRT.to_csv('3000cleaned.csv')
