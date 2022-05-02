import pandas as pd
import matplotlib.pyplot as plt 
from wordcloud import WordCloud

df = pd.read_csv('third_rendition_data/third_rendition_geolocated_output_anonymous.csv')
df = df.dropna()
anot_df = pd.read_csv('annotation_5800_012label_600wli.csv')

df_pos = anot_df[anot_df['label'] == 2]
df_neg = anot_df[anot_df['label'] == 0] 
df_neut = anot_df[anot_df['label'] == 1] 

tweet_pos = ' '.join(elem for elem in df_pos.text)
tweet_neg = ' '.join(elem for elem in df_neg.text)
tweet_neut = ' '.join(elem for elem in df_neut.text)
tweet_all = " ".join(review for review in df.text)

wordcloud_all = WordCloud(max_font_size = 50, max_words = 100, background_color = 'white').generate(tweet_all)
wordcloud_neg = WordCloud(max_font_size = 50, max_words = 100, background_color = 'white').generate(tweet_neg)
wordcloud_neut = WordCloud(max_font_size = 50, max_words = 100, background_color = 'white').generate(tweet_neut)
wordcloud_pos = WordCloud(max_font_size = 50, max_words = 100, background_color = 'white').generate(tweet_pos)

fig, ax = plt.subplots(4,1, figsize = (30,30))

ax[0].imshow(wordcloud_all, interpolation = 'bilinear')
ax[0].set_title('All tweets')
ax[0].axis('off')

ax[1].imshow(wordcloud_pos, interpolation = 'bilinear')
ax[1].set_title('positive tweets')
ax[1].axis('off')

ax[2].imshow(wordcloud_neut, interpolation = 'bilinear')
ax[2].set_title('neutral tweets')
ax[2].axis('off')

ax[3].imshow(wordcloud_neg, interpolation = 'bilinear')
ax[3].set_title('negative tweets')
ax[3].axis('off')

plt.show()
