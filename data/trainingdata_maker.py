"""
Script for anotizing trainingdata.
Currently has three categories, 2 for neutral, 1 for positive and 0 for negative.
change the num_tweets variable to alter the number of tweets to annotate.

"""
import pandas as pd 
import numpy as np

data = pd.read_csv('fourth_rendition_data/fourth_rendition_output_id.csv',
                    usecols = ['text'])

#data = pd.read_csv('second_rendition_predicted_logitsorted_midbottom300.csv',
#                    usecols = ['text'])

output_file = '100_thirdrendition_anotated_fourthrendition4.csv'
num_tweets = 100

tweet_indx = np.random.randint(0, len(data), size = num_tweets) 
trainingdata = data.iloc[tweet_indx, :] 


label = np.zeros(len(trainingdata))

def label_prompt():
    
    labels = [0,1,2]
    try:
    
        arg = int(input('1: neutral/informative, 2: postivie, 0: negative  ')) 
        assert int(arg) in labels
    except:
        print('argument must be int, and either 0, 1 or 2')
        arg = int(input('1: neutral/informative, 2: postivie, 0: negative  '))

    return arg

for i, text in enumerate(trainingdata['text']):
    print('-'*20+'\n', i)
    print(text)
    

    arg = label_prompt()
        


    label[i] = int(arg)


trainingdata['label'] = label.astype(int)
trainingdata.to_csv(output_file, index=False)


