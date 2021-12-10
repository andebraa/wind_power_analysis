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
    data['indx'] = np.arange(len(data))
    
    #make two different datasets with and without retweets  
   
    #retweets_mask = data['text'].str.startswith('RT @')
    #unfulfilled_mask = data['text'].str.endswith('...') #is it ... or … ?
    #unfulfilled_retweets = data[retweets_mask | unfulfilled_mask] # tweets starts with RT and ends with ...
    unfulfilled_mask = data['text'].str.match(f'RT @.+\.\.\.$')
    unfulfilled_retweets= data[unfulfilled_mask]

    non_retweets = data[ ~unfulfilled_mask]
    non_retweets['text'].drop_duplicates(keep=False) 


    itera= 0
    for i, row in unfulfilled_retweets.iterrows():
         
        print('-'*10)
        print(type(row['text']))
        #extract unfulfilled tweet. match this with the fulfilled one, replace
        #stripped_string = row.iloc['text'].str.extract(f'RT @(?:\w{1,15})\b(?::){0,1}(.+)\.\.\.') #finds username, captures everything after.
        
        stripped_string = re.search(f'RT @(?:\w{1,15})\b(?::){0,1}(.+)\.\.\.', row['text'])
        print(stripped_string)

        print(non_retweets['text'])        
        match_mask = non_retweets['text'].str.match(stripped_string) 
        print('here')
        match_case = non_retweets[match_mask]
        data.iloc[row['indx']]['text'] = match_case
        print(data.iloc[row['indx']]['text'])
        print(print(match_case))

        if itera == 2:
            break
        itera += 1

if __name__ == '__main__':
    fulfill_retweets('data/first_rendition_data/final_dataset.csv')
