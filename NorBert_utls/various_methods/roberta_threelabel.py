# -*- coding: utf-8 -*-
'''
based on Serena Y Kims code for sentiment in USA.
https://github.com/SerenaYKim/Solar-Sentiment-BERT
'''

import os
import re
import time
import nltk
import torch
import random
import datetime
import numpy as np
import pandas as pd
import seaborn as sn 
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.metrics import f1_score, confusion_matrix 
from transformers import BertForSequenceClassification, AutoTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaConfig, RobertaForSequenceClassification, AdamW, get_cosine_schedule_with_warmup
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

class EarlyStopper:
    #https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        print('early stopper')
        if validation_loss < self.min_validation_loss:
            print('validation loss < min_validation loss')
            print(self.min_validation_loss)
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            print('validation loss > (min val loss + delta)')
            print('counter: ', self.counter)
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def roberta_threelabel(lr = 1e-5, batch_size = 16, epochs = 10, plot = False, predict = False):
    infile = 'annotation_5900_012label_300wli_cleaned'
    df = pd.read_csv('~/wind_power_analysis/data/anotation_data/'+infile+'.csv', 
                     sep=',', usecols=['text', 'label'], index_col=None)

    #----------------------------------------------------------------------------------------
    df = df.iloc[1:]
    df = df.dropna()
    train = df.sample(frac = 0.9, random_state=195)
    test = df.drop(train.index)

    # rm_list = ',:.";|()$1234567890-@^#!?$=%~&+*/\[]{}'
    rm_list = ''
    sentences = train.text.values
    labels = train.label.values
    print(df.isnull().values.any())
    df.dropna()
    print(df.isnull().values.any())

    sentences = train['text'].values.tolist()
    print(type(sentences))
    print(type(sentences[0]))

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
    print(np.shape(input_ids))
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

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
        num_labels = 3, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    model.cuda()

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
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

    early_stopper = EarlyStopper(patience=1, min_delta=0)
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

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
        #early stopping
        if early_stopper.early_stop(avg_val_loss):
            print('early stop')
            break

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

    labels = labels.astype(int)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 300,           # Pad & truncate all sentences.
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
    f1 = f1_score(y_true=true_labels, y_pred=predictions, average=None)
    print(f1)
    print(type(f1))
    avg_val_accuracy = total_eval_accuracy / len(prediction_dataloader)
    print("  Accuracy: {0:.3f}".format(avg_val_accuracy))
    print(f"epochs {epochs}")
    print(f"lr {lr}")
    print(f"batch_size {batch_size}")
    print(f"  F1: {f1}")
    print('    DONE.')
    
    breakpoint()
    f10 = f1[0]
    f11 = f1[1]
    f12 = f1[2]
    print(f10)
    print(type(f10))
    #store training and prediction results
    runs_folder = f'../runs/epochs{epochs}_batch_size{batch_size}_lr{lr}_threelabel_f10{f10:.2f}_f11{f11:.2f}_f12{f12:.2f}_acc{avg_val_accuracy:.2f}'
    folder = False
    folder_number = 1
    while not folder:#add a _1 to folder if it exists
        try:
            os.mkdir(runs_folder)
            folder = True
        except:
            runs_folder = f'../runs/epochs{epochs}_batch_size{batch_size}_lr{lr}_threelabel_f1{f1:.2f}_acc{avg_val_accuracy:.2f}'
            runs_folder = runs_folder + f'_{folder_number}'
            folder_number += 1
    model.save_pretrained(runs_folder +f'/model')
    df_stats.to_csv(runs_folder+f'/training_stats_epochs{epochs}_batch_size{batch_size}_lr{lr}_threelabel.csv')
    header = f'epochs{epochs}_batch_size{batch_size}_lr{lr}_threelabel_f1{f1}_acc{avg_val_accuracy}'
    with open(runs_folder+'/config.txt', 'w') as fp:
        fp.write(header +'\n')
        fp.write(f'infile: {infile}')



    epochs_arr = np.arange(len(validation_accuracy))
    if plot:
        epochs_list = np.arange(epochs)
        plt.plot(epochs_list, validation_accuracy, label = 'val acc')
        plt.plot(epochs_list, training_loss, label = 'training loss')
        plt.plot(epochs_list, validation_loss, label = 'val loss')
        plt.xticks(epochs_list)
        plt.xlabel('epochs')
        plt.legend()
        plt.title(f'batch length {batch_size}, final f1; neg {f1[0]:.2f}, neut {f1[1]:.2f}, pos {f1[2]:.2f}, test accuracy {avg_val_accuracy:.2f}')
        plt.savefig(infile+'.png')
        plt.show()
        conf_matrix = True
        
        if conf_matrix:
            confusion = confusion_matrix(true_labels, predictions)
            plt.figure(figsize = (11,11))
            df_cm = pd.DataFrame(confusion, index = [i for i in ('negative', 'neutral', 'positive')],
                  columns = [i for i in ('negative', 'neutral', 'positive')])
            sn.heatmap(df_cm, annot = True, fmt = 'g')
            print(epochs)
            print(f1)
            print(batch_size)
            plt.title(f'final f1; neg {f1[0]:.2f}, neut {f1[1]:.2f}, pos {f1[2]:.2f}, test accuracy {avg_val_accuracy:.2f}')
            plt.savefig(infile + '_confusion.png')
        # ========================================
        #               Prediction
        # ========================================

    if predict:
        predict_dataset = '/../data/fourth_rendition_data/'
        filename = 'fourth_rendition_geolocated_id_anonymous_cleaned.csv'
        with open(predict_dataset+filename) as f:
            df = pd.read_csv(predict_dataset+filename, sep=',', usecols=['id',
                                                            'username',
                                                            'text',
                                                            'loc',
                                                            'created_at',
                                                            'like_count',
                                                            'quote_count',
                                                            'latitude',
                                                            'longitude'])

            print('Number of test sentences: {:,}\n'.format(df.shape[0]))
            print(df.head())
            # Create sentence and label lists
            df = df.dropna()
            sentences = df.text.values

            # Tokenize all of the sentences and map the tokens to thier word IDs.
            input_ids = []
            attention_masks = []

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
            batch_size = 16

            prediction_data = TensorDataset(input_ids, attention_masks, labels, phraseids)
            prediction_sampler = SequentialSampler(prediction_data)
            prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


            print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

            model.eval()

            predictions , true_labels, phrase_id1 = [], [], []
            for batch in prediction_dataloader:
                batch = tuple(t.to(device) for t in batch)

                b_input_ids, b_input_mask = batch

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up prediction
                with torch.no_grad():
                # Forward pass, calculate logit predictions
                    outputs = model(b_input_ids, token_type_ids=None,
                                    attention_mask=b_input_mask)
                print(outputs)
                print(np.shape(outputs))
                logits = outputs[0]

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                    
                print(logits)
                print(np.shape(logits))
                pred_flat = np.argmax(logits, axis=1).flatten()
                # Store predictions and true labels
                logits_list0.append(logits[:,0].flatten())
                logits_list1.append(logits[:,1].flatten())
                logits_list2.append(logits[:,2].flatten())

                predictions.append(pred_flat)

            predictions = list(chain.from_iterable(predictions))
            true_labels = list(chain.from_iterable(true_labels))
            logits0 = list(chain.from_iterable(logits_list0))
            logits1 = list(chain.from_iterable(logits_list1))
            logits2 = list(chain.from_iterable(logits_list2))
            df1 = pd.DataFrame(list(zip(sentences, predictions, logits0, logits1, logits2)), columns=['text', 'label', 'logits0', 'logits1', 'logits2'])
            out_filename = f'fourth_rendition_geolocated_id_threelabel_predict.csv'
            df['label'] = df1.label.copy()
            df['logits0'] = df1.logits0.copy()
            df['logits1'] = df1.logits1.copy()
            df['logits2'] = df1.logits2.copy()
            df.to_csv(predict_dataset+out_filename)

    return epochs_arr, validation_accuracy, training_loss, validation_loss, f1

def gridsearch():

    lr_start = 1e-6
    lr_stop = 8e-6
    lr_step = 8
    lrs = np.linspace(lr_start, lr_stop, lr_step)
    batch_sizes = [32,64]


    fig, ax = plt.subplots(4,4, figsize = (15,10))
    fig.tight_layout(pad = 3)
    ax = ax.ravel()
    _figure = 0
    for i, batch_size in enumerate(batch_sizes):
        print(batch_size)
        for j, lr in enumerate(lrs):
            epochs, validation_accuracy, training_loss, validation_loss, f1 = roberta_sentiment(lr=lr,
                                                                                            batch_size = batch_size,
                                                                                            epochs = 20)

            ax[_figure].plot(epochs, validation_accuracy, label = 'val acc')
            ax[_figure].plot(epochs, training_loss, label = 'training loss')
            ax[_figure].plot(epochs, validation_loss, label = 'val loss')
            ax[_figure].set_xticks(epochs)
            ax[_figure].set_xlabel('epochs')
            ax[_figure].legend(fontsize = 'x-small')
            ax[_figure].set_title(f'batch size {batch_size}, lr {lr:.3f}, f1{f1:.2f}')

            _figure += 1
    plt.savefig(f'../fig/roberta_gridsearch_batch{batch_sizes}_ls{lr_start}_{lr_stop}_{lr_step}.png', dpi = 500)

if __name__ == '__main__':
    roberta_threelabel(lr = 1e-6, batch_size = 64, epochs = 20, plot=False, predict = False)
