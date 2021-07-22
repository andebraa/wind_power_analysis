"""
script for assigning users in dataset a random unique identifier.
Takes a finished csv, where 'username' is the user name.

"""
import numpy as np
import pandas as pd

data = pd.read_csv('twitterdata.csv', parse_dates = True)# , usecols = [''])
ID = {} #dictionary to translate ID and usernames
tweet_occurances = {} #keep track of number of tweets per user

def generate_ID(ID):
    """
    should recursively loop through random numbers until a unique value is found.
    NOTE; don't use with a too large dataset, as it will bottom out at 10000
    """
    new_ID = np.random.randint(0, 10000)
    if new_ID not in ID:
        return new_ID
    else:
        generate_ID(ID)

for i, elem in enumerate(data['username']):
    if elem in ID.values():
        #this nasty line extracts the key given a value. apparently it's the only way
        #might have issues if lists get too long
        new_ID = list(ID.keys())[list(ID.values()).index(elem)]
        tweet_occurances[elem] += 1 #another tweet from user elem
    else:
        new_ID = generate_ID(ID)
        tweet_occurances[elem] = 1 #first tweet by user elem 
    ID[new_ID] = elem
    data['username'][i] = new_ID

print(len(ID))
num_of_tweets = np.arange(0,np.max(data.values()) 
for i in range(len(tweet_occurances)):
    num_of_tweets[tweet_occurances[i].values()] += 1 
print(num_of_tweets)
print(tweet_occurances)
#TODO use tweet_occurances, which lists the amounts of time each user tweeted to make
# a histogram of number of tweets on the x axis, and amount of users which have 
# tweeted given amount 
data.to_csv('anon_twitterdata.csv')
