import re
import pandas as pd
import numpy as np

data = pd.read_csv('second_rendition_data/second_rendition_geolocated.csv') 


#RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)

file_path = os.path.dirname(os.path.abspath(__file__))
emoji_key = pd.read_csv(file_path + '/data/' + 'emoji_table.txt', encoding='utf-8', index_col=0)

emoji_dict = emoji_key['count'].to_dict()

dictionary = emoji_dict
emoji_list = emoji_dict.keys()

RE_EMOJI = re.compile(
    "["
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251" 
    "]+",
    flags=re.UNICODE)

def strip_emoji(text):
    return RE_EMOJI.sub(r'', text)

for i, line in data.iterrows():
    prev = line['text']
    line['text'] = strip_emoji(line['text'])
    if line['text'] != prev:
        print(prev)
        print(line['text'])
        print('\n \n')

data.to_csv('test_no_emoji.csv')
