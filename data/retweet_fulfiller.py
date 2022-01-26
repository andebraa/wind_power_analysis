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
   
        tweet_found = False
        print('\n \n original retweet: ',row['text'])
        print('index ',i)
        #extract unfulfilled tweet. match this with the fulfilled one, replace
        #stripped string is everything but the username and dots
        res  = re.findall(r'RT @(\w{1,15})\b(?::){0,1}(.+)\.\.\.', row['text'])
        uname = res[0][0]
        stripped_string = res[0][1]
        
        curr_line_reg = '.*(?:' + stripped_string + ').*'
        
        match_case = 0
        #Some original tweets might not occur in the dataset (user might not have geo tag,). 
        #if so we skip to the next one

        user_tweets = non_retweets.loc[non_retweets['username']==uname] 
        if user_tweets.empty:
            print('empty')
            continue #skips to next iteration of loo#skips to next iteration of loopp
        print(user_tweets) 
        #finding the original tweet based on the stripped retweet
    
        for _i, _row in user_tweets.iterrows():
            preamble = 5 + len(uname) +1 #RT @<uname>:
            print('_row: ', _row['text'])
            test = _row['text'][preamble:len(stripped_string)] 
            print('test ', test)
            
            print('stripped string ', stripped_string)
            #if _row.str.contains(stripped_string, regex=False).any():
            #    print('yeah, here---------------------------------------------------------------')
           
            print(_row.str.contains(curr_line_reg, regex=True).any())
            if _row.str.contains(curr_line_reg, regex=True).any():
                print('yeah, here***************************************************************')
            
            if re.match(curr_line_reg, _row['text']):
                print('here')
                match_case = _row['text']
                print('MATCH ',match_case)
                tweet_found = True
                break
            
            
            
        #match_case = user_tweets[user_tweets['text'].str.contains(stripped_string)]
        #print(match_case)
        """
        orig_tweet = user_tweets['text'][user_tweets['text'].str.contains(stripped_string)].to_string(index=False)

        print('orig_tweet', orig_tweet)
        match_mask = user_tweets['text'].str.contains(stripped_string)
        if match_mask.any():
            print('sucess 2')
        print(match_mask)
        stop
        #match_case = non_retweets[match_mask]
        print('original tweet: \n', match_case)
        #setting the text row in dataset to be the original tweet
        """
        
        print('match, ', match_case)
        if tweet_found:
            print('tweet found')
            data.iloc[row['indx']]['text'] = match_case

        print('--------w----------')
#        print(data.iloc[row['indx']]['text'])

        if itera >= 5:
            STOP
        itera += 1
    data.to_csv('no_unfulfilled_retweets.csv')

if __name__ == '__main__':
    fulfill_retweets('second_rendition_data/second_rendition_geolocated.csv')
    #fulfill_retweets('data/first_rendition_data/final_dataset.csv')
