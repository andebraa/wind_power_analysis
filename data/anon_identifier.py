"""
script for assigning users in dataset a random unique identifier.
Takes a finished csv, where 'username' is the user name.

"""
import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle

data = pd.read_csv('second_rendition_data/second_rendition_geolocated.csv', parse_dates = True)# , usecols = [''])
ID = {} #dictionary to translate ID and usernames
tweet_occurances = {} #keep track of number of tweets per user

def generate_ID(ID):
    """
    should recursively loop through random numbers until a unique value is found.
    NOTE; don't use with a too large dataset, as it will bottom out at 100000
    """
    new_ID = np.random.randint(0, 100000)
    if new_ID not in ID:
        return new_ID
    else:
        generate_ID(ID)

for i, elem in enumerate(data['username']):
    #If the element (username) has already been given a anon value, 
    if elem in ID.values():
        #this nasty line extracts the key given a value. apparently it's the only way
        #might have issues if lists get too long
        new_ID = list(ID.keys())[list(ID.values()).index(elem)]
        tweet_occurances[elem] += 1 #another tweet from user elem
    else:
        new_ID = generate_ID(ID)
        tweet_occurances[elem] = 1 #first tweet by user elem 
    ID[new_ID] = elem
    data.loc[i, 'username'] = new_ID
    #data['username'][i] = new_ID

print("number of unique users: {}".format(len(ID)))

max_number_of_tweets = max(tweet_occurances.values())
num_of_tweets = np.arange(1, max_number_of_tweets+2) 
for elem in tweet_occurances.values():
    num_of_tweets[elem] += 1 


print("highest number of tweets by single user:{}".format(max_number_of_tweets)) 
fig, ax = plt.subplots()

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(8)


#logbins=np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

max_key = max(tweet_occurances, key=tweet_occurances.get)
print(max_key)
print(tweet_occurances[max_key]) 

ax.hist(tweet_occurances.values() , bins = np.linspace(1, max_number_of_tweets-2, 50))
plt.yscale('log')
plt.xscale('linear')
plt.ylabel('number of users')
plt.xlabel('number of tweets')
plt.title('Frequency of tweets by users. 2007 -2021')

plt.savefig('second_rendition_geolocated.jpg', bbox_inches = 'tight', pad_inches = 0.1) #0.1 is default when bbox is tight

outfile_ID = open('second_rendition_geolocated_translator.json', 'w')
json.dump(ID, outfile_ID) #writing the translation dictionary to file

data.to_csv('second_rendition_geolocated_anonymous.csv', index=False)
