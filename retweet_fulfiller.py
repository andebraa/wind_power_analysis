import re
import numpy as np
import pandas as pd

def fulfill_retweets(filename):
    """
    twitter retweet elements start with RT, and have only a limited number of letters.
    they then end abruptly with .... this scripts looks for the original of a retweet
    and fills inn all tweets that is a retweet of this original tweet.
    This is so that we still get that opinion, with the retweetee geodata.
    NOTE we assume that a retweet (not a quote retweet) is an agreement with the original
    sentiment.
    """
    data = pd.read_csv(filename) 
    
    #make two different datasets with and without retweets  
   
    retweets_mask = data['text'].str.startswith('RT')
    

    retweets = data[retweets_mask]
    unfulfilled_mask = retweets['text'].str.endswith('…')

    unfulfilled_retweets = retweets[unfulfilled_mask] 
    unfulfilled_retweets['text'].drop_duplicates(keep=False) 

    non_retweets = data[~retweets_mask]


    itera= 0
    for i, row in unfulfilled_retweets.iterrows():
        #RT (.*(?:\n.*){0,1})(?:$|…)
        string = row['text'][3:-1]

        print(string) 
        print(row['text'])
        print('-'*10)
        if itera == 10:
            break
        itera += 1

if __name__ == '__main__':
    fulfill_retweets('final_dataset.csv')
