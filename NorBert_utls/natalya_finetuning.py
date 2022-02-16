
import os 
import io
import torch
import pandas as pd
import numpy as np
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, random_split

import torch
# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

df = pd.read_csv('~/annotaion_3000_012label.csv',delimiter=",", usecols=['text', 'label'])
df = df.reset_index(drop=True)
#df = df.loc[1:1934,['text', 'label']]

# shuffle to get CV 
df_train, df_test = sklearn.model_selection.train_test_split(df, train_size=round(df.shape[0]*0.8), test_size=round(df.shape[0]*0.2),shuffle=True)
df_train.shape

sentences = df.text.values
labels    = df.label.values

print(sentences)
print(labels)

# Load the BERT tokenizer.
print('Loading AutoTokenizer...')
tokenizer = AutoTokenizer.from_pretrained('ltgoslo/norbert2', do_lower_case=False)

max_len = 0

# For every sentence...
for sent in sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, add_special_tokens=True)

    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))

print('Max sentence length: ', max_len)
