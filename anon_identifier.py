"""
script for assigning users in dataset a random unique identifier.
Takes a finished csv, where 'author_id' is the user name.

"""
import numpy as np
import pandas as pd

data = pd.read_csv('twitterdata.csv', parse_dates = True)# , usecols = [''])
ID = {} #dictionary to translate ID and usernames

def generate_ID(ID):
    """
    should recursively loop through random numbers until a unique value is found.
    NOTE; don't use with a too large dataset, as it will bottom out at 10000
    """
    new_ID = np.random.randint(0, 10000)
    if new_ID nor in ID:
        return new_ID
    else:
        generate_ID(ID)

for i, elem in enumerate(data['author_id']):
    new_ID = generate_ID(ID)
    ID[new_ID] = elem
    data['author_id'][i] = new_ID
