import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from top2vec import Top2Vec

data = pd.read_csv('anotation_data/annotation_5900_posneutral_0neg_1pos_300wli_cleaned.csv',
                   usecols = ['text'])
data_list = data.values.tolist()
data_list = list(chain.from_iterable(data_list))#unpack nested list of strings"
model = Top2Vec(data_list, speed = 'deep-learn')

num_topics = model.get_num_topics()
topic_words, words_scores, topic_nums = model.get_topics(num_topics)


topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["vindkraft"], num_topics=num_topics)

for topic in topic_nums:
    model.generate_topic_wordcloud(topic)
plt.savefig('../fig/top2vec_wordcloud.png')
