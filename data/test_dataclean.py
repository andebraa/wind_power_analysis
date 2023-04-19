
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
pd.options.mode.chained_assignment = None #default='warn'
def stopword_remove(tweet):
    #start with removing stopwords
    stopword = nltk.corpus.stopwords.words('norwegian')
    searchquery = ['havvind','vindkraft','vindmølle','vindmøller','vindmøllene','vindturbiner','vindenergi']
    for word in searchquery:
        stopword.append(word)
    tweet_words = tweet.split(' ')
    for tweet_word in tweet_words:
        if tweet_word in stopword:
            tweet_words.remove(tweet_word)
            
    tweet = ' '.join(tweet_words)
    #remove searchwords
    havvind_regex = '(?:^|(?<= ))(havvind|vindkraft|vindmølle|vindmøller|vindmøllene|vindturbiner|vindenergi])(?:$|(?= ))'
    while re.search(havvind_regex, tweet, re.I| re.M):
        tweet = re.sub(havvind_regex, '', tweet, re.I)
    return tweet
        

def wordcloud_call(labels, sentences):

    truemask = labels==1
    falsemask = labels==0

    truetweets = sentences[truemask]
    falsetweets = sentences[falsemask]
    fig, ax = plt.subplots(3,1, figsize = (30,30))

    tweet_pos = ' '.join(elem for elem in truetweets)
    tweet_neg = ' '.join(elem for elem in falsetweets)
    tweet_all = ' '.join(elem for elem in sentences)


    wordcloud_all = WordCloud(max_font_size = 50, max_words = 100, background_color = 'white').generate(tweet_all)
    wordcloud_neg = WordCloud(max_font_size = 50, max_words = 100, background_color = 'white').generate(tweet_neg)
    wordcloud_pos = WordCloud(max_font_size = 50, max_words = 100, background_color = 'white').generate(tweet_pos)

    ax[0].imshow(wordcloud_all, interpolation = 'bilinear')
    ax[0].set_title('All tweets')
    ax[0].axis('off')

    ax[1].imshow(wordcloud_pos, interpolation = 'bilinear')
    ax[1].set_title('positive tweets')
    ax[1].axis('off')

    ax[2].imshow(wordcloud_neg, interpolation = 'bilinear')
    ax[2].set_title('negative tweets')
    ax[2].axis('off')

    plt.show()
def preprocess(sentences):
    '''
    remove usernames, stopword, urls and ": " from retweets
    NOTE! will alter short input to 'drop', make sure to remove theese after 
    function call like this:
    data_clean = data.drop(data.index[data['text'] =='drop'])
    '''





    for i in range(0,len(sentences)):
        print(i)
        print('before process: ')
        print(sentences[i])
        sentences[i] = re.sub('RT ', ' ', sentences[i]) # the RT in retweets
        sentences[i] = re.sub('@[^\s]+',' ',sentences[i]) #all sernames
        sentences[i] = re.sub('https:\/\/t.co\/(?:[a-zA-Z])+(\s+)',' ',sentences[i]) #https://t.co/w+
        sentences[i] = re.sub('&[^\s]+',' ',sentences[i]) #&[*all non whitespace*] ?
        sentences[i] = re.sub('https?://\S+',' ',sentences[i]) #urls
        sentences[i] = ' '.join(sentences[i].split()) #insert space
        if len(sentences[i]) <5: #if input is really short, i.e. it was all usernames etc
            sentences[i] = 'drop' #REMEMBER to remove drop input after function call
            continue #skip to next iteration in for loop
        if sentences[i][0] == ':':
            if sentences[i][1] == ' ':
                sentences[i] = sentences[i][2:] # is sentence starts with ': ', remove it (retweets?)
            else:
              sentences[i] = sentences[i][1:]
        sentences[i] = sentences[i].split()
        sentences[i] = [s.strip() for s in sentences[i]]
        for j, word in enumerate(sentences[i]):
            sentences[i][j] = ''.join(e for e in sentences[i][j] if e.isalnum())
        sentences[i] = ' '.join(sentences[i])
        sentences[i] = stopword_remove(sentences[i])
        print('after all is done: ')
        print(sentences[i])
        breakpoint()
    return sentences

data = pd.read_csv('anotation_data/annotaion_5900_01label_comb_posneutral_0neg_1pos_300iwl.csv', usecols = ['text', 'label'], index_col = None, sep = ',')

#data = data.head(10)
data_clean = data.copy()

data_clean['text'] = preprocess(data['text'])
data_clean = data.drop(data.index[data['text'] =='drop'])
print(data_clean)

wordcloud_call(data_clean['label'], data_clean['text'])

