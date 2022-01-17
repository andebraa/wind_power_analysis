import re
import numpy as np
import pandas as pd
#note, following supresses possibly vital warnings
pd.options.mode.chained_assignment = None  # default='warn'


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
   
    unfulfilled_mask = data['text'].str.match(f'RT @.+\.\.\.$')
    unfulfilled_retweets= data[unfulfilled_mask]

    non_retweets = data[ ~unfulfilled_mask]
    non_retweets['text'].drop_duplicates(keep=False) 


    itera= 0
    for i, row in unfulfilled_retweets.iterrows():

        print('original retweet: ',row['text'])
        #extract unfulfilled tweet. match this with the fulfilled one, replace
        #stripped string is everything but the username and dots
        res  = re.findall(r'RT @(\w{1,15})\b(?::){0,1}(.+)\.\.\.', row['text'])
        uname = res[0][0]
        stripped_string = res[0][1]
        print('-------------------')
        print('stripped string :\n', stripped_string)
        print(uname)
        
        #Some original tweets might not occur in the dataset (user might not have geo tag, or 
        # not directly mention wind energy). if so we skip to the next one

        user_tweets = data.loc[data['username']==uname] 
        print(user_tweets)
        if user_tweets.empty:
            continue #skips to next iteration of loo#skips to next iteration of loopp
        stop
        #finding the original tweet based on the stripped retweet
        for _i, _row in non_retweets.iterrows():
            #print('\nRT ', row['text'])
            #print('\noriginal',_row['text'])
            preamble = 5 + len(uname) +1 #RT @<uname>:
            #print(row['text'][preamble:preamble+len(stripped_string)])
            #print(_row['text'][preamble:preamble+len(stripped_string)])

            #print(row['text'][preamble:preamble+len(stripped_string)] == _row['text'][preamble:preamble+len(stripped_string)])
            if _row.str.contains(stripped_string, regex=False).any():
                print('twat')
            if re.findall(re.escape(stripped_string), _row['text']):
                print('match')
        match_case = non_retweets[non_retweets['text'].str.contains(re.escape(stripped_string), na=False)==True]
        match_mask = non_retweets['text'].str.contains(stripped_string)
        print(non_retweets[match_mask])
        #match_case = non_retweets[match_mask]
        print('original tweet: \n', match_case)
        #setting the text row in dataset to be the original tweet
        data.iloc[row['indx']]['text'] = match_case

        print('--------w----------')
#        print(data.iloc[row['indx']]['text'])

        if itera == 5:
            return 0
        itera += 1
    data.to_csv('no_unfulfilled_retweets.csv')

if __name__ == '__main__':
    fulfill_retweets('second_rendition_data/second_rendition_geolocated.csv')
    #fulfill_retweets('data/first_rendition_data/final_dataset.csv')
