import pandas as pd 
import numpy as np

data = pd.read_csv('full_geodata_longlat_noforeign.csv',
                    usecols = ['text'])

tweet_indx = np.random.randint(0, len(data), size = 200) 


trainingdata = data.iloc[tweet_indx, :] 


pos_mask = np.zeros(len(trainingdata))

for i, text in enumerate(trainingdata['text']):
    print('-'*20, i)
    print(text)
    try:
    
        arg = int(input('1: postivie, 0: negative  ')) 
    except:
        print('argument must be int.')
        arg = int(input('1: postivie, 0: negative  '))
    pos_mask[i] = int(arg)

trainingdata['label'] = pos_mask.astype(int)

trainingdata.to_csv('anotized_data_200_5.csv', index=False)


