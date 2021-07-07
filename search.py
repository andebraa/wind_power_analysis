#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json

#The place object is always present when a Tweet is geo-tagged, while the coordinates object is only present (non-null) when the Tweet is assigned an exact location
#ach point is an array in the form of [longitude, latitude].
config = {
  "bearer_token": "AAAAAAAAAAAAAAAAAAAAAJMfPAEAAAAA3rFe4%2Fj%2Fwd%2Bsiff%2FOcMVw0RtnZo%3DuPS4o36B7uMWh4Dy9qdfUJLTkHmwW0JtUTAAPxaFZ1Blfpv38l",
  "params": {
    "start_time": "2020-02-28T00:00:00Z",
    "end_time": "2020-12-01T00:00:00Z",
    "query": "havvind -is:retweet has:geo",
    "max_results": 20, #it seems like you also have to change the other two places where max_results are listed below
    "tweet_fields": "context_annotations,created_at",
    "user_fields": "created_at",
    "place_fields": "contained_within,country,country_code,full_name,geo,id,name,place_type",
    "expansions": "author_id,geo.place_id"
  },
  "write_path": "havvind_2.txt"
}


def lookup(id, list):
    return next(item for item in list if item['id'] == id)


def get_formatted_tweets(json_response):
    list_of_tweets = []
    has_expansion_data = False
    data = json_response['data']
    if 'includes' in json_response:
        includes = json_response['includes']
        has_expansion_data = True
    for tweet_info in data:
        if has_expansion_data:
            if 'users' in includes:
                tweet_info['user'] = lookup(tweet_info['author_id'], includes['users'])
            if 'places' in includes:
                if 'geo' in tweet_info:
                    tweet_info['place'] = lookup(tweet_info['geo']['place_id'], includes['places'])
        list_of_tweets.append(tweet_info)
    return list_of_tweets


def validate_config():
    if 'bearer_token' not in config or config['bearer_token'] == "":
        raise Exception("Bearer token is missing from the config file")
    if 'params' in config and ('query' not in config['params'] or config['params']['query'] == ""):
        raise Exception("Please make sure to provide a valid search query")
    if 'write_path' not in config:
        raise Exception("Please specify the output path where you want the Tweets to be written to")


# Function to write Tweet to new line in a file
def write_to_file(file_name, tweets):
    with open(file_name, 'a+') as filehandle:
        for tweet in tweets:
            filehandle.write('%s\n' %tweet)


def search_tweets(next_token=None):
    tweet_params = {'max_results': 20}  #THIS IS PER REQUEST! YOU CAN'T HAVE MORE THAN THIS PER REQUEST

    if 'params' in config:
        params = config['params']

        if 'tweet_fields' in params:
            tweet_params['tweet.fields'] = params['tweet_fields']
        if 'user_fields' in params:
            tweet_params['user.fields'] = params['user_fields']
        if 'place_fields' in params:
            tweet_params['place.fields'] = params['place_fields']
        if 'expansions' in params:
            tweet_params['expansions'] = params['expansions']
        if 'start_time' in params:
            tweet_params['start_time'] = params['start_time']
        if 'end_time' in params:
            tweet_params['end_time'] = params['end_time']
        if 'query' in params:
            tweet_params['query'] = params['query']

    if next_token is not None:
        tweet_params['next_token'] = next_token

    headers = {"Authorization": "Bearer {}".format(config['bearer_token'])}

    response = requests.request("GET", 'https://api.twitter.com/2/tweets/search/all', headers=headers,
                                params=tweet_params)
    #in webinar he called this requests.get

    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )

    return response.json()


if __name__ == '__main__':

    validate_config()

    count = 0
    next_token = None

    if 'params' in config and 'max_results' in config['params']:
        max_results = config['params']['max_results']
    else:
        max_results = 20

    while count < max_results:

        json_response = search_tweets(next_token)
        tweets = get_formatted_tweets(json_response)
        write_to_file(config['write_path'], tweets)
        result_count = json_response['meta']['result_count']
        count += result_count

        if 'meta' in json_response and 'next_token' in json_response['meta']:
            next_token = json_response['meta']['next_token']
        else:
            break


# In[ ]:
