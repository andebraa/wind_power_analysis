#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import json
import numpy as np
import datetime

start_time = "2022-10-17T00:00:00z"
end___time = "2022-10-23T00:00:00z"
print(start_time)
print(end___time)

config = {
  "bearer_token": "" #ADD BEARER TOKEN
  "params": {
    "start_time": start_time,
    "end_time": end___time,
    "query": "(havvind OR vindkraft OR vindmølle OR vindmøller OR vindmøllene OR vindturbiner OR vindenergi) lang:no",
    "max_results": 500, #it seems like you also have to change the other two places where max_results are listed below
    "tweet_fields": "geo,created_at,public_metrics",
    "user_fields": "location",
    "place_fields": "country,full_name,geo,name",
    "expansions": "author_id,geo.place_id"
  },
  "write_path": "all_data_all_time.txt"
}


def lookup(id, list):
    return next(item for item in list if item['id'] == id)


def get_formatted_tweets(json_response):
    list_of_tweets = []
    has_expansion_data = False
    data = json_response['data']
    #print(json_response)  
    skipped = 0
    if 'includes' in json_response:
        includes = json_response['includes']
        has_expansion_data = True
    for tweet_info in data:
        if has_expansion_data:
            if 'users' in includes:
                tweet_info['user'] = lookup(tweet_info['author_id'], includes['users'])
            if 'places' in includes:
                if 'geo' in tweet_info:
                    try:
                        tweet_info['place'] = lookup(tweet_info['geo']['place_id'], includes['places'])
                    except:
                        skipped += 1 
                        pass
        list_of_tweets.append(tweet_info)
    print('len of list of tweets ',len(list_of_tweets))
    print('skipped: ', skipped)
    return list_of_tweets


def validate_config():
    if 'bearer_token' not in config or config['bearer_token'] == "":
        raise Exception("Bearer token is missing from the config file")
    if 'params' in config and ('query' not in config['params'] or config['params']['query'] == ""):
        raise Exception("Please make sure to provide a valid search query")
    if 'write_path' not in config:
        raise Exception("Please specify the output path where you want the Tweets to be written to")

def connect_to_endpoint(url, headers, params):
    """
    https://stackoverflow.com/questions/65109472/how-do-i-loop-my-python-code-for-twitter-api-v2-recent-search
    """
    print('calling twitter')
    response = requests.request("GET", url, headers=headers, params=params)

    # Twitter returns (in the header of the request object) how many
    # requests you have left. Lets use this to our advantage
    remaining_requests = int(response.headers["x-rate-limit-remaining"])
    print('remaining_requests', remaining_requests)
    # If that number is one, we get the reset-time
    #   and wait until then, plus 15 seconds (your welcome Twitter).
    # The regular 429 exception is caught below as well,
    #   however, we want to program defensively, where possible.
    if remaining_requests == 1:
        buffer_wait_time = 15
        resume_time = datetime.fromtimestamp( int(response.headers["x-rate-limit-reset"]) + buffer_wait_time )
        print(f"Waiting on Twitter.\n\tResume Time: {resume_time}")
        pause_until(resume_time)  ## Link to this code in above answer

    # We still may get some weird errors from Twitter.
    # We only care about the time dependent errors (i.e. errors
    #   that Twitter wants us to wait for).
    # Most of these errors can be solved simply by waiting
    #   a little while and pinging Twitter again - so that's what we do.
    if response.status_code != 200:

        # Too many requests error
        if response.status_code == 429:
            buffer_wait_time = 15
            resume_time = datetime.fromtimestamp( int(response.headers["x-rate-limit-reset"]) + buffer_wait_time )
            print(f"Waiting on Twitter.\n\tResume Time: {resume_time}")
            pause_until(resume_time)  ## Link to this code in above answer

        # Twitter internal server error
        elif response.status_code == 500:
            # Twitter needs a break, so we wait 30 seconds
            resume_time = datetime.now().timestamp() + 30
            print(f"Waiting on Twitter.\n\tResume Time: {resume_time}")
            pause_until(resume_time)  ## Link to this code in above answer

        # Twitter service unavailable error
        elif response.status_code == 503:
            # Twitter needs a break, so we wait 30 seconds
            resume_time = datetime.now().timestamp() + 30
            print(f"Waiting on Twitter.\n\tResume Time: {resume_time}")
            pause_until(resume_time)  ## Link to this code in above answer

        # If we get this far, we've done something wrong and should exit
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )

    # Each time we get a 200 response, lets exit the function and return the response.json
    if response.ok:
        return response



# Function to write Tweet to new line in a file
def write_to_file(file_name, tweets):
    with open(file_name, 'a+') as filehandle:
        for tweet in tweets:
            filehandle.write('%s\n\n' % json.dumps(tweet))


def search_tweets(next_token=None):
    """
    
    """

    tweet_params = {'max_results': 500}  #THIS IS PER REQUEST! YOU CAN'T HAVE MORE THAN THIS PER REQUEST

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

    response = connect_to_endpoint('https://api.twitter.com/2/tweets/search/all', headers=headers,
                                params=tweet_params)

    #response = requests.request("GET", 'https://api.twitter.com/2/tweets/search/all', headers=headers,
                                #params=tweet_params)
    #in webinar he called this requests.get

    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )

    return response.json()



def loop_request():
    validate_config()

    count = 0
    next_token = None

    first_of_month= "-01T00:00:00Z" # for looping over time invertvals [2020-01-01, 2020-02-01...] etc
    year = "2020-"
    months = np.arange(1,12, dtype='int') #not include 12 as we have +1 in config
    months = ["{:02d}".format(item) for item in months] #adding leading zeros


    if 'params' in config and 'max_results' in config['params']:
        max_results = config['params']['max_results']
    else:
        max_results = 500

    for i,month in enumerate(months):
        config['params']['start_time'] = year + str(months[i]) + first_of_month
        config['params']['end_time'] = year+ str(months[i+1]) + first_of_month

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

if __name__ == '__main__':
    validate_config()

    count = 0
    next_token = None

    if 'params' in config and 'max_results' in config['params']:
        max_results = config['params']['max_results']
    else:
        max_results = 500

    while count < max_results:

        json_response = search_tweets(next_token)
        if "'result_count': 0" in json_response:
            pass #this means there are no results
        tweets = get_formatted_tweets(json_response)
        write_to_file(config['write_path'], tweets)
        result_count = json_response['meta']['result_count']
        count += result_count

        if 'meta' in json_response and 'next_token' in json_response['meta']:
            next_token = json_response['meta']['next_token']
        else:
            break
print('result count: ',result_count)
