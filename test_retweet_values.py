import numpy as np
import pandas as pd 

def all_same(items):
    return all(x == items[0] for x in items) 


def test_identical_retweets():
    """
    script for testing if retweets that are identical actually have identical labels
    """
    data = pd.read_csv('final_dataset.csv',
                       usecols = [
                           'username','text','loc','created_at','like_count',
                           'quote_count','city','latitude','longitude','labels']
                       )
    retweets_mask = data['text'].str.startswith('RT') 
    
    retweets = data[retweets_mask] 
    non_retweets = data[~retweets_mask] 
    

    duplicates = {} #{tweet_text: [list of labels of all occurances, i.e. [[1.949, 0.434],[0.555, 0.666]]
    for i, row in data[retweets_mask].iterrows():
        if row['text'] not in duplicates:
            duplicates[row['text']] = [row['labels']] #if item not already in duplicates, add it 
        else:
            duplicates[row['text']].append(row['labels'])

    for key, val in duplicates.items():
        if len(val) > 1:
            assert all_same(val) 


    return 0 

if __name__ == '__main__':
    test_identical_retweets()

