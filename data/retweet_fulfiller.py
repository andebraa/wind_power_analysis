import re
import time 
import numpy as np
import pandas as pd
#note, following supresses possibly vital warnings
pd.options.mode.chained_assignment = None  # default='warn'


def fulfill_retweets(filename, outname, drop_unfulfilled = False):
    """
    twitter retweet elements start with RT, and have only a limited number of letters.
    they then end abruptly with .... this scripts looks for the original of a retweet
    and fills inn all tweets that is a retweet of this original tweet.
    This is so that we still get that opinion, with the retweetee geodata.
    NOTE we assume that a retweet (not a quote retweet) is an agreement with the original
    sentiment.

    args:
        filename(string): name of input file
        outname(string): name of ouput file
        drop_unfulfilled(bool): whether to drop unfulfilled retweets with no orignial
                                in the dataset
    """
    data = pd.read_csv(filename, index_col = False) 
    data['indx'] = np.arange(len(data))
   
    print(data.columns)
    orig_len = len(data)
    #make two different datasets with and without retweets  
   
    #unfulfilled_mask = data['text'].str.match(f'RT @.+\.\.\.$')
    unfulfilled_mask = data['text'].str.match(r'RT @(?:\w{1,15})\b(?::){0,1} (?:(?:.|\n)+)(?:\.\.\.|…)') # add limitation in number of tweet characters?
    unfulfilled_retweets= data[unfulfilled_mask] #14000 tweets

    non_retweets = data[ ~unfulfilled_mask] # 40000 tweets
    non_retweets['text'].drop_duplicates(keep=False) 

    data_drops = [] #list of row indexes to drop
    drops = 0
    for i, unfulfilled_row in unfulfilled_retweets.iterrows():
   
        tweet_found = False
        match_case = False
        #print('\n \n original retweet: ',i,row['text'])
        #extract unfulfilled tweet. match this with the fulfilled one, replace
        #stripped string is everything but the username and dots
        res  = re.findall(r'RT @(\w{1,15})\b(?::){0,1} ((?:.|\n)+)(?:\.\.\.|…)', unfulfilled_row['text'])
        uname = res[0][0]
        stripped_string = res[0][1].strip()
        
        #Some original tweets might not occur in the dataset (user might not have geo tag,). 
        #if so we skip to the next one

        #user_tweets = data.loc[data['username']==uname] 
        user_tweets = non_retweets.loc[non_retweets['username']==uname] 
            
        if user_tweets.empty:
            #print('empty')
            drops += 1
            if drop_unfulfilled:
                data_drops.append(unfulfilled_row['indx'])
            continue #skips to next iteration of loo#skips to next iteration of loopp
        #finding the original tweet based on the stripped retweet
   
        #print('stripped string ', stripped_string)
        for _i, _row in user_tweets.iterrows():
            #print('_ row ', _row['text'])
            if stripped_string in _row['text']:
                
                #print('match __________________________________________________________________')
                match_case = _row['text']
                tweet_found = True
                break


        
        #print('match, ', match_case)
        if tweet_found:
            #print('tweet found')
            print('orig tweet', data.iloc[unfulfilled_row['indx']]['text'])
            print('match case', match_case)
            data.iloc[unfulfilled_row['indx']]['text'] = match_case
        elif not tweet_found:
            drops +=1
            #print('no match?')
            if drop_unfulfilled:
                data_drops.append(unfulfilled_row['indx'])
        #print('--------w----------')
#        print(data.iloc[row['indx']]['text'])
    #print(drops)
    #print(len(data))
    for indx in data_drops:
        data.drop(indx)
    data.drop(columns = 'indx')
    data.to_csv('no_unfulfilled_retweets_dropped.csv')

def test_no_unfilled_retweet():
    filename = 'second_rendition_data/second_rendition_geolocated.csv'
    outname = 'no_unfulfilled_retweets_dropped.csv'
    #fulfill_retweets(filename, outname, True) 

    data = pd.read_csv(outname)
    wrong_data = pd.DataFrame()
    for i, row in data.iterrows():
        res = re.findall(r'RT @(?:\w{1,15})\b(?::){0,1} (?:(?:.|\n)+)(?:\.\.\.|…)', row['text'])
        if res:
            print(row['text'])
            wrong_data.append(row)
    wrong_data.to_csv('test_wrong_data.csv')



if __name__ == '__main__':
    #fulfill_retweets('second_rendition_data/second_rendition_geolocated.csv', 
    #                  'no_unfulfilled_retweets_dropped.csv',  
    #                  drop_unfulfilled = True)
    #test_no_unfilled_retweet()
    fulfill_retweets('annotation_5000_012label_wli.csv', 'annotation_5000_012label_wli_fulfilled.csv', True)
