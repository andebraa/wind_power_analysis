def flatten_tweets(tweets):
    """ Flattens out tweet dictionaries so relevant JSON is
        in a top-level dictionary. """

    tweets_list = []

    # Iterate through each tweet
    for tweet_obj in tweets:

        ''' User info'''
        # Store the user screen name in 'user-screen_name'
        tweet_obj['user-screen_name'] = tweet_obj['user']['screen_name']

        # Store the user location
        tweet_obj['user-location'] = tweet_obj['user']['location']

        ''' Text info'''
        # Check if this is a 140+ character tweet
        if 'extended_tweet' in tweet_obj:
            # Store the extended tweet text in 'extended_tweet-full_text'
            tweet_obj['extended_tweet-full_text'] = \
                                    tweet_obj['extended_tweet']['full_text']

        if 'retweeted_status' in tweet_obj:
            # Store the retweet user screen name in
            # 'retweeted_status-user-screen_name'
            tweet_obj['retweeted_status-user-screen_name'] = \
                        tweet_obj['retweeted_status']['user']['screen_name']

            # Store the retweet text in 'retweeted_status-text'
            tweet_obj['retweeted_status-text'] = \
                                        tweet_obj['retweeted_status']['text']

            if 'extended_tweet' in tweet_obj['retweeted_status']:
                # Store the extended retweet text in
                #'retweeted_status-extended_tweet-full_text'
                tweet_obj['retweeted_status-extended_tweet-full_text'] = \
                tweet_obj['retweeted_status']['extended_tweet']['full_text']

        if 'quoted_status' in tweet_obj:
            # Store the retweet user screen name in
            #'retweeted_status-user-screen_name'
            tweet_obj['quoted_status-user-screen_name'] = \
                            tweet_obj['quoted_status']['user']['screen_name']

            # Store the retweet text in 'retweeted_status-text'
            tweet_obj['quoted_status-text'] = \
                                            tweet_obj['quoted_status']['text']

            if 'extended_tweet' in tweet_obj['quoted_status']:
                # Store the extended retweet text in
                #'retweeted_status-extended_tweet-full_text'
                tweet_obj['quoted_status-extended_tweet-full_text'] = \
                    tweet_obj['quoted_status']['extended_tweet']['full_text']

        ''' Place info'''
        if 'place' in tweet_obj:
            # Store the country code in 'place-country_code'
            try:
                tweet_obj['place-country'] = \
                                            tweet_obj['place']['country']

                tweet_obj['place-country_code'] = \
                                            tweet_obj['place']['country_code']

                tweet_obj['location-coordinates'] = \
                            tweet_obj['place']['bounding_box']['coordinates']
            except: pass

        tweets_list.append(tweet_obj)

    return tweets_list


def select_text(tweets):
    ''' Assigns the main text to only one column depending
        on whether the tweet is a RT/quote or not'''

    tweets_list = []

    # Iterate through each tweet
    for tweet_obj in tweets:

        if 'retweeted_status-extended_tweet-full_text' in tweet_obj:
            tweet_obj['text'] = \
                        tweet_obj['retweeted_status-extended_tweet-full_text']

        elif 'retweeted_status-text' in tweet_obj:
            tweet_obj['text'] = tweet_obj['retweeted_status-text']

        elif 'extended_tweet-full_text' in tweet_obj:
                    tweet_obj['text'] = tweet_obj['extended_tweet-full_text']

        tweets_list.append(tweet_obj)

    return tweets_list


import pandas as pd
import json

tweets_data = []
for line in open('havvind_2.json','r'):
    tweets_data.append(json.loads(line))


# flatten tweets
tweets = flatten_tweets(tweets_data)

# select text
tweets = select_text(tweets)
columns = ['text', 'lang', 'user-location', 'place-country',
           'place-country_code', 'location-coordinates',
           'user-screen_name']

# Create a DataFrame from `tweets`
df_tweets = pd.DataFrame(tweets)#, columns=columns)
# replaces NaNs by Nones
df_tweets.where(pd.notnull(df_tweets), None, inplace=True)

print(df_tweets.head)
