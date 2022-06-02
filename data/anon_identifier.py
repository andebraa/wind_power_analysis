"""
script for assigning users in dataset a random unique identifier.
Takes a finished csv, where 'username' is the user name.

"""
import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import (TextArea, AnnotationBbox)

data = pd.read_csv('third_rendition_data/third_rendition_geolocated.csv', parse_dates = True)# , usecols = [''])
ID = {} #dictionary to translate ID and usernames
tweet_occurances = {} #keep track of number of tweets per user


def generate_ID(ID):
    """
    should recursively loop through random numbers until a unique value is found.
    NOTE; don't use with a too large dataset, as it will bottom out at 100000
    """
    new_ID = np.random.randint(0, 1000000)
    if new_ID not in ID:
        return new_ID
    else:
        return generate_ID(ID)

def anonymizer():
    for i, elem in enumerate(data['username']):
        #If the element (username) has already been given a anon value, 
        if elem in ID.values():
            #this nasty line extracts the key given a value. apparently it's the only way
            #might have issues if lists get too long
            new_ID = list(ID.keys())[list(ID.values()).index(elem)]
            tweet_occurances[elem] += 1 #another tweet from user elem
        else:
            new_ID = generate_ID(ID)
            assert not np.isnan(new_ID)
            tweet_occurances[elem] = 1 #first tweet by user elem 
        ID[new_ID] = elem
        data.loc[i, 'username'] = new_ID
        #data['username'][i] = new_ID

    print("number of unique users: {}".format(len(ID)))

    max_number_of_tweets = max(tweet_occurances.values())
    num_of_tweets = np.arange(1, max_number_of_tweets+2) 
    for elem in tweet_occurances.values():
        num_of_tweets[elem] += 1 

    sorted_tweet_occurances = dict(sorted(tweet_occurances.items(), key = lambda item: item[1]))
    print("highest number of tweets by single user:{}".format(max_number_of_tweets)) 
    fig, ax = plt.subplots()

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(8)

    
    #logbins=np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    

    max_key = max(tweet_occurances, key=tweet_occurances.get)
    print(max_key)
    print('tweet occurances')
    print(list(sorted_tweet_occurances)[-1])
    print(list(sorted_tweet_occurances)[-2])
    print(list(sorted_tweet_occurances)[-3])
    print(tweet_occurances[list(sorted_tweet_occurances)[-1]])

    p1, t1 = (list(sorted_tweet_occurances)[-1], tweet_occurances[list(sorted_tweet_occurances)[-1]])
    #making a marker for the highest tweets user
    xy1 = (t1, 2)
    ax.plot(xy1[0], xy1[1])
    
    offsetbox1 = TextArea(p1)
    ab1 = AnnotationBbox(offsetbox1, xy1,
                        xybox = (2500, 4000),
                        #box_alignment = (5, 1),
                        arrowprops = dict(arrowstyle='->'))
                
    
    p2, t2 = (list(sorted_tweet_occurances)[-2], tweet_occurances[list(sorted_tweet_occurances)[-2]])
    #making a marker for the highest tweets user
    xy2 = (t2, 2)
    ax.plot(xy2[0], xy2[1])
    offsetbox2 = TextArea(p2)
    ab2 = AnnotationBbox(offsetbox2, xy2,
                        xybox = (2000, 3500),
                        #box_alignment = (5, 1),
                        arrowprops = dict(arrowstyle='->'))

    p3, t3 = (list(sorted_tweet_occurances)[-3], tweet_occurances[list(sorted_tweet_occurances)[-3]])
    #making a marker for the highest tweets user
    xy3 = (t3, 2)
    ax.plot(xy3[0], xy3[1])
    offsetbox3 = TextArea(p3)
    ab3 = AnnotationBbox(offsetbox3, xy3,
                        xybox = (1500, 4000),
                        #box_alignment = (5, 1),
                        arrowprops = dict(arrowstyle='->'))

    print(tweet_occurances.values())
    ax.hist(tweet_occurances.values() , bins = np.linspace(1, max_number_of_tweets+2, 500))
    plt.yscale('log')
    plt.xscale('linear')
    plt.ylabel('number of users')
    plt.xlabel('number of tweets')
    plt.title('Frequency of tweets by users. 2007 -2021')

    ax.add_artist(ab1)
    ax.add_artist(ab2)
    ax.add_artist(ab3)
    #plt.show()
    plt.savefig('third_rendition_data/third_rendition_geolocated_user_tweetfreq.png',dpi = 300, format='png',  bbox_inches = 'tight', pad_inches = 0.1) #0.1 is default when bbox is tight

    with open('third_rendition_data/third_rendition_geolocated_translator.json', 'w') as outfile_ID:
        json.dump(ID, outfile_ID) #writing the translation dictionary to file

    #data.to_csv('third_rendition_data/third_rendition_geolocated_anonymous.csv', index=False)

def identifyer():
    df = pd.read_csv('third_rendition_data/third_rendition_geolocated_anonymous_posneutral_predict.csv')
    with open('third_rendition_data/third_rendition_geolocated_translator.json') as fp:
        trans_dict = json.loads(fp.read())

    
    for i, elem in df.iterrows():
        elem = elem['username']
        
        
        df.loc[i, 'username'] = trans_dict[str(elem)]
    print(df)
    df.to_csv('third_rendition_data/third_rendition_geolocated_posneutral_predict.csv')
        
if __name__ == '__main__':
    #identifyer()
    anonymizer()
