"""
script for assigning users in dataset a random unique identifier.
Takes a finished csv, where 'username' is the user name.

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

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
print(tweet_occurances.values())
print(tweet_occurances)
max_number_of_tweets = max(tweet_occurances.values())
num_of_tweets = np.arange(1, max_number_of_tweets+2) 
for elem in tweet_occurances.values():
    num_of_tweets[elem] += 1 
print(num_of_tweets)
print(tweet_occurances)

fig, ax = plt.subplots()

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(8)

ax.hist(tweet_occurances.values(), bins = np.arange(1, max_number_of_tweets))
plt.ylabel('number of users')
plt.xlabel('number of tweets')
plt.title('Frequency of tweets by users. 2017 -2021')

plt.savefig('first_elec_tweetfrequency.jpg', bbox_inches = 'tight', pad_inches = 0.0001) #0.1 is default when bbox is tight




data.to_csv('anon_twitterdata.csv')
