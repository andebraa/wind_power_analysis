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
plt.style.use('seaborn')

data = pd.read_csv('fourth_rendition_data/fourth_rendition_geolocated_id.csv', parse_dates = True)# , usecols = [''])
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

    p1, t1 = (list(sorted_tweet_occurances)[-1], tweet_occurances[list(sorted_tweet_occurances)[-1]])
    #making a marker for the highest tweets user
    xy1 = (t1, 2)
    #ax.plot(xy1[0], xy1[1])
    
    offsetbox1 = TextArea(p1)
    ab1 = AnnotationBbox(offsetbox1, xy1,
                        xybox = (2500, 4000),
                        #box_alignment = (5, 1),
                        arrowprops = dict(arrowstyle='->'))
                
    
    p2, t2 = (list(sorted_tweet_occurances)[-2], tweet_occurances[list(sorted_tweet_occurances)[-2]])
    #making a marker for the highest tweets user
    xy2 = (t2, 2)
    #ax.plot(xy2[0], xy2[1])
    offsetbox2 = TextArea(p2)
    ab2 = AnnotationBbox(offsetbox2, xy2,
                        xybox = (2000, 1500),
                        #box_alignment = (5, 1),
                        arrowprops = dict(arrowstyle='->'))

    p3, t3 = (list(sorted_tweet_occurances)[-3], tweet_occurances[list(sorted_tweet_occurances)[-3]])
    #making a marker for the highest tweets user
    xy3 = (t3, 1)
    #ax.plot(xy3[0], xy3[1])
    offsetbox3 = TextArea(p3)
    ab3 = AnnotationBbox(offsetbox3, xy3,
                        xybox = (1600, 4000),
                        #box_alignment = (5, 1),
                        arrowprops = dict(arrowstyle='->'))

    p4, t4 = (list(sorted_tweet_occurances)[-4], tweet_occurances[list(sorted_tweet_occurances)[-4]])
    #making a marker for the highest tweets user
    xy4 = (t4, 2)
    #ax.plot(xy4[0], xy4[1])
    offsetbox4 = TextArea(p4)
    ab4 = AnnotationBbox(offsetbox4, xy4,
                        xybox = (1200, 1500),
                        #box_alignment = (5, 1),
                        arrowprops = dict(arrowstyle='->'))

    p5, t5 = (list(sorted_tweet_occurances)[-5], tweet_occurances[list(sorted_tweet_occurances)[-5]])
    #making a marker for the highest tweets user
    xy5 = (t5, 2)
    #ax.plot(xy5[0], xy5[1])
    offsetbox5 = TextArea(p5)
    ab5 = AnnotationBbox(offsetbox5, xy5,
                        xybox = (1000, 4000),
                        #box_alignment = (5, 1),
                        arrowprops = dict(arrowstyle='->'))
    

    p6, t6 = (list(sorted_tweet_occurances)[-6], tweet_occurances[list(sorted_tweet_occurances)[-6]])
    #making a marker for the highest tweets user
    xy6 = (t6, 2)
    #ax.plot(xy6[0], xy6[1])
    offsetbox6 = TextArea(p6)
    ab6 = AnnotationBbox(offsetbox6, xy6,
                        xybox = (500, 1500),
                        #box_alignment = (5, 1),
                        arrowprops = dict(arrowstyle='->'))

    ax.hist(tweet_occurances.values() , bins = np.linspace(1, max_number_of_tweets+2, 350)) 
    plt.yscale('log')
    plt.xscale('linear')
    plt.ylabel('number of users')
    plt.xlabel('number of tweets')
    plt.title('Frequency of tweets by users. 2007 -2021')

    #ax.add_artist(ab1)
    #ax.add_artist(ab2)
    #ax.add_artist(ab3)
    #ax.add_artist(ab4)
    #ax.add_artist(ab5)
    #ax.add_artist(ab6)
    #plt.show()
    plt.savefig('fourth_rendition_data/fourth_rendition_geolocated_user_tweetfreq.png',dpi = 300, format='png',  bbox_inches = 'tight', pad_inches = 0.1) #0.1 is default when bbox is tight

    with open('fourth_rendition_data/forth_rendition_geolocated_id_translator.json', 'w') as outfile_ID:
        json.dump(ID, outfile_ID) #writing the translation dictionary to file

    data.to_csv('fourth_rendition_data/fourth_rendition_geolocated_id_anonymous.csv', index=False)

def identifyer():
    '''
    translate anon usernames back to plain text usernames
    '''
    df = pd.read_csv('fourth_rendition_data/fourth_rendition_geolocated_id_posneutral_anonymous_predict.csv')
    with open('fourth_rendition_data/forth_rendition_geolocated_id_translator.json') as fp:
        trans_dict = json.loads(fp.read())

    
    for i, elem in df.iterrows():
        elem = elem['username']
        
        
        df.loc[i, 'username'] = trans_dict[str(elem)]
    print(df)
    df.to_csv('fourth_rendition_data/fourth_rendition_geolocated_id_posneutral_predict.csv')
        
if __name__ == '__main__':
    identifyer()
    #anonymizer()

