# -*- coding: utf-8 -*-
'''
based on Serena Y Kims code for sentiment in USA.
https://github.com/SerenaYKim/Solar-Sentiment-BERT
'''

import re
import time
import torch
import random
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, AutoTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaConfig, RobertaForSequenceClassification, AdamW, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score

# Function to calculate the accuracy of our predictions vs labels
def _removeNonAscii(s): return "".join(i for i in s if ord(i)<128)
def stemmingWords(sentence,dictionary):
    return " ".join([dictionary.get(w,w) for w in sentence.split()])
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def roberta_sentiment(lr = 1e-5, batch_size = 16, epochs = 10, plot = False, predict = False):
    infile = 'annotaion_3000_01label_comb_posneutral_0neg_1pos'
    df = pd.read_csv('~/wind_power_analysis/data/'+infile+'.csv', 
                     sep=',', usecols=['text', 'label'], index_col=None)

    #----------------------------------------------------------------------------------------
    df = df.iloc[1:]
    train = df.sample(frac = 0.9, random_state=195)
    test = df.drop(train.index)

    # rm_list = ',:.";|()$1234567890-@^#!?$=%~&+*/\[]{}'
    rm_list = ''
    sentences = train.text.values
    labels = train.label.values
    """
    for i in range(0,len(sentences)):
      sentences[i] = re.sub('@[^\s]+',' ',sentences[i])
      sentences[i] = re.sub('&[^\s]+',' ',sentences[i])
      sentences[i] = re.sub('https?://\S+',' ',sentences[i])
      sentences[i] = _removeNonAscii(sentences[i])
      for j in rm_list:
        sentences[i] = sentences[i].replace(j,' ')
      sentences[i] = ' '.join(sentences[i].split())
      if sentences[i][0] == ':':
        if sentences[i][1] == ' ':
          sentences[i] = sentences[i][2:]
        else:
          sentences[i] = sentences[i][1:]
    """

    max_length = 300

    # Load the BERT tokenizer.
    tokenizer = AutoTokenizer.from_pretrained('ltgoslo/norbert2', do_lower_case=False)

    max_len = 0

    for i in range(0,len(sentences)):
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sentences[i], add_special_tokens=True)

        # Update the maximum sentence length.
        if (len(input_ids)>128):
            max_len = max(max_len, len(input_ids))

    print('max len ',max_len)
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    labels = labels.astype(int)
    labels = labels.astype(int)
    input_ids = []
    attention_masks = []


    for sent in sentences:
      
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,          # Pad & truncate all sentences.
                            truncation = True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Create a 90-10 train-validation split.
    # Calculate the number of samples to include in each set.
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))


    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order.
    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    model = BertForSequenceClassification.from_pretrained(
        "ltgoslo/norbert2", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    model.cuda()

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 42069

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    total_t0 = time.time()

    validation_loss = [] 
    training_loss = []
    validation_accuracy = [] 
    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            ll = model(b_input_ids,
                         token_type_ids=None,
                         attention_mask=b_input_mask,
                         labels=b_labels)

            total_train_loss += ll.loss.item()

            ll.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if (step + 1) % 2 == 0:
              optimizer.step()
            # Update the learning rate.
              scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)
        training_loss.append(avg_train_loss)
        
        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                ll  = model(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)

                loss = ll.loss
                logits = ll.logits
            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)


        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        validation_accuracy.append(avg_val_accuracy) # plotting
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        validation_loss.append(avg_val_loss) # plotting
        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')

    # A hack to force the column headers to wrap.
    #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])

    ################################################################################################################################
    ################################################ TEST ##########################################################################

    df = test
    print('Number of test sentences: {:,}\n'.format(df.shape[0]))
    print(df.head())
    # Create sentence and label lists
    sentences = df.text.values
    labels = df.label.values

    """
    for i in range(0,len(sentences)):
      sentences[i] = re.sub('@[^\s]+',' ',sentences[i])
      sentences[i] = re.sub('&[^\s]+',' ',sentences[i])
      sentences[i] = re.sub('https?://\S+',' ',sentences[i])
      sentences[i] = _removeNonAscii(sentences[i])
      for j in rm_list:
        sentences[i] = sentences[i].replace(j,' ')
      sentences[i] = ' '.join(sentences[i].split())
      if sentences[i][0] == ':':
        if sentences[i][1] == ' ':
          sentences[i] = sentences[i][2:]
        else:
          sentences[i] = sentences[i][1:]
    """

    labels = labels.astype(int)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 128,           # Pad & truncate all sentences.
                            truncation = True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)


    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    pd.options.display.max_colwidth = 400

    # Prediction on test set
    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions , true_labels = [], []
    total_eval_accuracy=0
    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        print(batch)    
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)
        # Store predictions and true labels
        predictions.append(np.argmax(logits, axis=1).flatten())
        true_labels.append(label_ids)

    predictions = [item for sublist in predictions for item in sublist]
    true_labels= [item for sublist in true_labels for item in sublist]
    f1 = f1_score(y_true=true_labels, y_pred=predictions)
    avg_val_accuracy = total_eval_accuracy / len(prediction_dataloader)
    print(predictions)
    print(true_labels)
    print("  Accuracy: {0:.3f}".format(avg_val_accuracy))
    print("  F1: {0:.3f}".format(f1))
    print('    DONE.')

    epochs = np.arange(epochs)
    if plot:
        plt.plot(epochs, validation_accuracy, label = 'val acc')
        plt.plot(epochs, training_loss, label = 'training loss')
        plt.plot(epochs, validation_loss, label = 'val loss')
        plt.xticks(epochs)
        plt.xlabel('epochs')
        plt.legend()
        plt.title(f'batch length {batch_size}, final f1 {f1:.2f}, test accuracy {avg_val_accuracy:.2f}')
        plt.savefig(infile+'.png')
        plt.show()
    
    return epochs, validation_accuracy, training_loss, validation_loss

        # ========================================
        #               Prediction
        # ========================================

    if predict:
        from itertools import chain
        import os

        for filename in os.listdir('...'):
          print("predicting file :", filename, '\n')
          with open('/.../'+filename) as f:
            df = pd.read_csv('/.../'+filename, sep=',', encoding='latin-1', names=['phrase', 'phraseid', 'state'])
            df = df[1:]
            df.insert(0, "label", [k for k in range(0,len(df))], True)
            print('Number of test sentences: {:,}\n'.format(df.shape[0]))
            print(df.head())
            # Create sentence and label lists
            df['phraseid'] = df['phraseid'].astype(int)
            sentences = df.phrase.values
            phraseids = df.phraseid.values
            labels = df.label.values
            for i in range(0,len(sentences)):
              sentences[i] = re.sub('@[^\s]+',' ',sentences[i])
              sentences[i] = re.sub('&[^\s]+',' ',sentences[i])
              sentences[i] = re.sub('https?://\S+',' ',sentences[i])
              sentences[i] = _removeNonAscii(sentences[i])
              for j in rm_list:
                sentences[i] = sentences[i].replace(j,' ')
              sentences[i] = ' '.join(sentences[i].split())
              if sentences[i][0] == ':':
                if sentences[i][1] == ' ':
                  sentences[i] = sentences[i][2:]
                else:
                  sentences[i] = sentences[i][1:]

              # Tokenize all of the sentences and map the tokens to thier word IDs.
            input_ids = []
            attention_masks = []

        # For every sentence...
            for sent in sentences:
                encoded_dict = tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = 128,           # Pad & truncate all sentences.
                                truncation = True,
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                           )

            # Add the encoded sentence to the list.
                input_ids.append(encoded_dict['input_ids'])

            # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(encoded_dict['attention_mask'])

        # Convert the lists into tensors.
            input_ids = torch.cat(input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)
            # print(labels, phraseids)
            labels = torch.tensor(labels)
            phraseids = torch.tensor(phraseids)
        # Set the batch size.
            batch_size = 16

        # Create the DataLoader.
            prediction_data = TensorDataset(input_ids, attention_masks, labels, phraseids)
            prediction_sampler = SequentialSampler(prediction_data)
            prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        # Prediction on test set

            print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

        # Put model in evaluation mode
            model.eval()

        # Tracking variables
            predictions , true_labels, phrase_id1 = [], [], []
        # Predict
            for batch in prediction_dataloader:
          # Add batch to GPU
              batch = tuple(t.to(device) for t in batch)

          # Unpack the inputs from our dataloader
              b_input_ids, b_input_mask, b_labels, b_phraseids = batch

          # Telling the model not to compute or store gradients, saving memory and
          # speeding up prediction
              with torch.no_grad():
              # Forward pass, calculate logit predictions
                  outputs = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask)

              logits = outputs[0]

          # Move logits and labels to CPU
              logits = logits.detach().cpu().numpy()
              label_ids = b_labels.to('cpu').numpy()
              phrase_ids = b_phraseids.to('cpu').numpy()

              pred_flat = np.argmax(logits, axis=1).flatten()
              labels_flat = label_ids.flatten()
              phrase_ids_flat = phrase_ids.flatten()
          # Store predictions and true labels
              predictions.append(pred_flat)
              true_labels.append(labels_flat)
              phrase_id1.append(phrase_ids_flat)

            predictions = list(chain.from_iterable(predictions))
            phrase_id1 = list(chain.from_iterable(phrase_id1))
            true_labels = list(chain.from_iterable(true_labels))
            df1 = pd.DataFrame(list(zip(sentences, predictions, phrase_id1)), columns=['text', 'label'])
            df1.to_csv('/.../'+filename)
        print('    DONE.')
def gridsearch():

    lrs = np.linspace(1e-8, 0.01, 8)
    batch_sizes = [16, 32]


    fig, ax = plt.subplots(4,4, figsize = (15,10)) 
    fig.tight_layout(pad = 3)
    ax = ax.ravel()
    _figure = 0
    for i, batch_size in enumerate(batch_sizes):
        print(batch_size)
        for j, lr in enumerate(lrs):
            epochs, validation_accuracy, training_loss, validation_loss = roberta_sentiment(lr=lr, 
                                                                                            batch_size = batch_size,
                                                                                            epochs = 15) 
           
            ax[_figure].plot(epochs, validation_accuracy, label = 'val acc')
            ax[_figure].plot(epochs, training_loss, label = 'training loss')
            ax[_figure].plot(epochs, validation_loss, label = 'val loss')
            ax[_figure].set_xticks(epochs)
            ax[_figure].set_xlabel('epochs')
            ax[_figure].legend(fontsize = 'x-small')
            ax[_figure].set_title(f'batch size {batch_size}, lr {lr:.3f}')
            
            _figure += 1

    plt.savefig('roberta_gridsearch.png', dpi = 500) 

if __name__ == '__main__':
        

    roberta_sentiment(lr = 1e-5, batch_size = 16, epochs = 15, plot=True, predict = False)

