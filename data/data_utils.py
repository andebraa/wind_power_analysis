import nltk 
import string
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

def append_csv():
    #note; wli: weak logits included
    df1 = pd.read_csv('annotation_5000_012label_wli.csv', usecols = ['text', 'label'], index_col = False)
    df2 = pd.read_csv('300_secondrendition_anotated_worse_logitssorted_midbottom.csv', usecols = ['text', 'label'], index_col = False)
    df3 = pd.read_csv('200_thirdrendition_anotated4.csv', usecols = ['text', 'label'], index_col = False)
    df4 = pd.read_csv('200_thirdrendition_anotated5.csv', usecols = ['text', 'label'], index_col = False)
    df5 = pd.read_csv('100_thirdrendition_anotated8.csv', usecols = ['text', 'label'], index_col = False)

    df_out = pd.concat([df1, df2, df3, df4, df5])
    df_out = df_out.reset_index()
    df_out.to_csv('annotation_5800_012label_600wli.csv')

def remove_category():
    data = pd.read_csv('annotation_5800_012label_600wli.csv')
    data_out = data.loc[data['label'] != 1] 
    data_out.loc[data_out['label'] == 2, 'label'] = 1 # set label 2 to 1, so we have 0,1
    data_out.to_csv('annotaion_5000_01label_noneutral_wli.csv', index = False)

def rename_category():
    data = pd.read_csv('annotation_5800_012label_600wli.csv')
    #data.loc[data['label'] == 1, 'label'] = 2 #neutral now positive
    #data.loc[data['label'] == 2, 'label'] = 1 # set label 2 to 1, so we have 0,1
    #data.to_csv('annotaion_5800_01label_comb_posneutral_0neg_1pos_600iwl.csv', index = False)
    
    data.loc[data['label'] == 1, 'label'] = 0 # set label 1 to 0
    data.loc[data['label'] == 2, 'label'] = 1 # set label 2 to 1, so we have 0,1
    data.to_csv('annotaion_5800_01label_comb_negneutral_0neg_1pos_600iwl.csv', index = False)
    

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

def skewed_data():
    '''
    looks at the percentage of negative and positive tweets in combined datasets
    '''
    #data_negneut = pd.read_csv('annotaion_3000_01label_comb_negneutral_0neg_1pos.csv') 
    #data_posneut = pd.read_csv('annotaion_3000_01label_comb_posneutral_0neg_1pos.csv') 
    data_posneut = pd.read_csv('annotation_5800_012label_600wli.csv') 
    print(len(data_posneut.loc[data_posneut['label']==0, 'label']))
    print(len(data_posneut.loc[data_posneut['label']==1, 'label']))
    print(len(data_posneut.loc[data_posneut['label']==2, 'label']))

    stop
    neg0 = len(data_negneut.loc[data_negneut['label'] ==0, 'label'])
    neg1 = len(data_negneut.loc[data_negneut['label'] ==1, 'label'])
    neg_tot = len(data_negneut)

    pos0 = len(data_posneut.loc[data_posneut['label'] ==0, 'label'])
    pos1 = len(data_posneut.loc[data_posneut['label'] ==1, 'label'])
    pos_tot = len(data_posneut)

    print(f'neg0: {neg0/neg_tot}, neg1: {neg1/neg_tot}')
    print(f'pos0: {pos0/pos_tot}, pos1: {pos1/pos_tot}')
    
def weak_performer_extractor():
    df = pd.read_csv('second_rendition_data/second_rendition_geolocated_noemoji_anonymous_predict.csv')
    sortorder = np.zeros((len(df), 2))
    sortorder[:,0] = np.arange(len(df))
    for i, line in df.iterrows():
        sortorder[i,1] = np.abs(line['logits0'] - line['logits1'])
    print(sortorder)
    sortorder = sortorder[sortorder[:,1].argsort()]
    df = df.iloc[sortorder[:,0]]
    bottom_600 = df.head(n=600)
    midbottom_300 = bottom_600.head(n=300)
    midbottom_300.to_csv('second_rendition_predicted_logitsorted_midbottom300.csv')
    print(df)

def unskew():
    df = pd.read_csv('annotation_5800_012label_600wli.csv')
    print('len df', len(df))
    #unfulfilled_mask = data['text'].str.match(r'RT @(?:\w{1,15})\b(?::){0,1} (?:(?:.|\n)+)(?:\.\.\.|???)')
    rt_mask = df['text'].str.match(r'RT.*')
    df = df[~rt_mask]
    df.to_csv('annotation_5800_012label_600wli_noRT.csv')
    print('len df noRT', len(df))
    
    zero = len(df[df['label'] == 0].index)
    one = len(df[df['label'] == 1].index)
    two = len(df[df['label'] == 2].index)
   
    limit = np.min(np.array((zero, one, two))) 
    print(limit)
    
    df1 = df[df['label'] ==0][:limit]
    df2 = df[df['label']==1][:limit]
    df3 = df[df['label']==2][:limit]
    
    unskewed = pd.concat([df1,df2,df3])

    unskewed = unskewed.sample(frac=1).reset_index(drop=True)
    
    unskewed.to_csv('annotation_1845_5800unskewed_wli.csv')


def data_clean():
    #https://www.kaggle.com/code/ragnisah/text-data-cleaning-tweets-analysis/notebook
    df = pd.read_csv('third_rendition_data/third_rendition_geolocated_output_anonymous.csv')
    df = df.dropna()

    def remove_punct(text):
        text  = "".join([char for char in text if char not in string.punctuation])
        text = re.sub('[0-9]+', '', text)
        return text
    df['text_punct'] = df['text'].apply(lambda x: remove_punct(x))
    print(df.head(10))

    stop_words = set(stopwords.words('norwegian')) 
    print(stop_words)

    def remove_stopwords(text):
        text = [word for word in text if word not in stopword]
        return text
    
    df['Tweet_nonstop'] = df['Tweet_tokenized'].apply(lambda x: remove_stopwords(x))
    df.head(10)
    

def temp_pos_neg_extractor():
    df = pd.read_csv('third_rendition_data/third_rendition_geolocated_anonymous_posneutral_predict.csv')
    print(df.columns)
    df_neg = df.loc[df['label'] ==0, 'text']
    df_pos = df.loc[df['label'] ==1, 'text']

    print(df_neg)
    df_neg.to_csv('third_rendition_data/third_rendition_geolocated_negative_text.csv')
    df_pos.to_csv('third_rendition_data/third_rendition_geolocated_positive_neutral_text.csv')
if __name__ == '__main__':
    #remove_category()
    #rename_category()
    skewed_data()
    #weak_performer_extractor()
    #append_csv()
    #unskew()
    #data_clean()
    #temp_pos_neg_extractor()
