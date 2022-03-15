import pandas as pd


def remove_category():
    data = pd.read_csv('annotaion_3000_012label.csv')
    data_out = data.loc[data['label'] != 1] 
    data_out.loc[data_out['label'] == 2, 'label'] = 1 # set label 2 to 1, so we have 0,1
    data_out.to_csv('annotaion_3000_01label_noneutral.csv', index = False)

def rename_category():
    data = pd.read_csv('annotaion_3000_012label.csv')
    data.loc[data['label'] == 1, 'label'] = 2 # set label 2 to 1, so we have 0,1
    data.loc[data['label'] == 2, 'label'] = 1 #positive now 1
    data.to_csv('annotaion_3000_01label_comb_posneutral_0neg_1pos.csv', index = False)
    

def rename_column():

    data = pd.read_csv('anotized_data_100_2.csv', usecols = ['labels', 'text'])

    data = data.rename(columns = {'labels': 'label'})

    data.to_csv('anotized_data_100_2.csv', index=False)



def slice_data():
    """
    Short script that takes the last 10000 tweets, adds a number indicating
    number of days since first datapoint, and writes this to a new csv
    """
    import pandas as pd
    import numpy as np

    data = pd.read_csv('data/full_geodata_longlat_noforeign_anonymous.csv',
                       usecols = ['username', 'text', 'created_at'],
                       index_col = False)



    part = data.tail(10000)
    part.loc[:,'created_at'] = part.loc[:,'created_at'].apply(pd.to_datetime)

    start_date = min(part['created_at'])
    end_date = max(part['created_at'])



    days = np.zeros(len(part))

    days_dict = {}

    for i, elem in part.iterrows():
        curr_day = elem['created_at']
        if curr_day not in days_dict:
            num_days = (curr_day - start_date).days
            days_dict[curr_day]= num_days

    j = 0
    for i,elem in part.iterrows():
        days[j] = days_dict[elem['created_at']]
        j +=1

    part['days'] = days

    print(part.head())
    print(part.tail())

    part.to_csv('natalia_slice_first_rendition_anon.csv')


def column_remover():

    """
    Pandas often adds an index collumn, which it then can interperet as an unnamed collumn
    if it is changed and then written. datasets then end up with multiple unnamed collumns.
    This temporary script removed the first few collumnss
    """

    import pandas as pd
    import re
    infile = input('infile: ')

    data = pd.read_csv(infile)

    print(data.columns)
    col = int(input('index of collumn to remove '))

    data = data.drop(columns=col)
    data.to_csv(infile, index = False)
    print(data)

if __name__ == '__main__':
    rename_category()
