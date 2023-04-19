import pandas as pd
import numpy as np
import tensorflow as tf
import json
import math
import os
from transformers import BertTokenizer, AutoConfig, TFAutoModelForSequenceClassification, optimization_tf, AutoTokenizer
from torch.utils.data import random_split

#@markdown Set the main model that the training should start from
model_name = 'NbAiLab/nb-bert-large' #@param ["NbAiLab/nb-bert-base", "NbAiLab/nb-bert-large", "bert-base-multilingual-cased"]
#@markdown ---
#@markdown Set training parameters
batch_size =  16#@param {type: "integer"}
init_lr = 2e-5 #@param {type: "number"}
end_lr = 0  #@param {type: "number"}
warmup_proportion = 0.1#@param {type: "number"}
num_epochs =   5#@param {type: "integer"}

#You might increase this for bert-base
max_seq_length = 128

infile = f'annotaion_5900_01label_comb_posneutral_0neg_1pos_300iwl_cleaned_walphanumerical'
df = pd.read_csv('~/wind_power_analysis/data/anotation_data/'+infile+'.csv', 
                 sep=',', usecols=['text', 'label'], index_col=None)


df = df.iloc[1:]
df = df.dropna()
print('' in df)
print('twt')
train_data = df.sample(frac = 0.9, random_state=195)
test_data = df.drop(train_data.index)
dev_data = train_data.sample(frac=0.3)
train_data = train_data.drop(dev_data.index)
train_data.dropna()
test_data.dropna()
dev_data.dropna()

#train_data = pd.read_csv(
#dev_data = pd.read_csv(
#test_data = pd.read_csv(
print(train_data)
print(test_data)
print(dev_data)


# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Turn text into tokens
train_encodings = tokenizer(
    train_data['text'].tolist(), truncation=True, padding=True, max_length=max_seq_length
)
dev_encodings = tokenizer(
    dev_data["text"].tolist(), truncation=True, padding=True, max_length=max_seq_length
)
test_encodings = tokenizer(
    test_data["text"].tolist(), truncation=True, padding=True, max_length=max_seq_length
)

# Create a tensorflow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), 
                                                    train_data["label"].tolist()
                                                    )).shuffle(1000).batch(batch_size)
dev_dataset = tf.data.Dataset.from_tensor_slices((
    dict(dev_encodings), dev_data["label"].tolist()
                                                )).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings), test_data["label"].tolist()
)).batch(batch_size)


print(f'The dataset is imported.\n\nThe training dataset has {len(train_data)} items.\nThe development dataset has {len(dev_data)} items. \nThe test dataset has {len(test_data)} items')
steps = math.ceil(len(train_data)/batch_size)
num_warmup_steps = round(steps*warmup_proportion*num_epochs)
print(f'You are planning to train for a total of {steps} steps * {num_epochs} epochs = {num_epochs*steps} steps. Warmup is {num_warmup_steps}, {round(100*num_warmup_steps/(steps*num_epochs))}%. We recommend at least 10%.')

# Estimate the number of training steps
train_steps_per_epoch = int(len(train_dataset)/batch_size)
num_train_steps = train_steps_per_epoch * num_epochs

# Initialise a Model for Sequence Classification with 2 labels
config = AutoConfig.from_pretrained(model_name, num_labels=2)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, config=config)

# Creating a scheduler gives us a bit more control
optimizer, lr_schedule = optimization_tf.create_optimizer(
    init_lr=init_lr,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps
)
# Compile the model
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy']) # can also use any keras loss fn
print(train_dataset)
print(dev_dataset)
print(num_epochs)
print(batch_size)

# Start training
history = model.fit(x=train_dataset, validation_data=dev_dataset, epochs=num_epochs, batch_size=batch_size)

print(f'\nThe training has finished training after {num_epochs} epochs.')


from sklearn.metrics import classification_report

print("Evaluate test set")
y_pred = model.predict(test_dataset)
y_pred_bool = y_pred["logits"].argmax(-1)
print(classification_report(test_data["label"], y_pred_bool, digits=4))
