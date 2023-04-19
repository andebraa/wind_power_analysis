import seaborn as sn
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import f1_score, confusion_matrix
from transformers import BertForSequenceClassification, AutoTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import RobertaConfig, RobertaForSequenceClassification, AdamW, get_cosine_schedule_with_warmup
from itertools import chain
def apply_model(fp):

    # ========================================
    #               Prediction
    # ========================================

    model = BertForSequenceClassification.from_pretrained(
        fp, # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )


    predict_dataset = '/home/ubuntu/wind_power_analysis/data/fourth_rendition_data/'
    #second_rendition_geolocated_anonymous.csv' 
    filename = 'fourth_rendition_geolocated_id_anonymous_cleaned.csv'
    print("predicting file :", filename, '\n')
    with open(predict_dataset+filename) as f: #NOTE:wtf are phrase ids
        df = pd.read_csv(predict_dataset+filename, sep=',', usecols=['username',
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
        sentences = df.text.values

          # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []
        max_length = 300
        #tokenizer = AutoTokenizer.from_pretrained(fp, do_lower_case=False)
        tokenizer = AutoTokenizer.from_pretrained('ltgoslo/norbert2', do_lower_case=False)
        for sent in sentences.tolist():
            encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_length,           # Pad & truncate all sentences.
                            truncation = True,
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

            input_ids.append(encoded_dict['input_ids'])

            attention_masks.append(encoded_dict['attention_mask'])

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        # print(labels, phraseids)
        #labels = torch.tensor(labels)
        #phraseids = torch.tensor(phraseids)

        batch_size = 16

        prediction_data = TensorDataset(input_ids, attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Prediction on test set

        print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
        model.eval()

        predictions, logits_list0, logits_list1 = [], [], []
    # Predict
        for batch in prediction_dataloader:
        # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
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
              #label_ids = b_labels.to('cpu').numpy()
              #phrase_ids = b_phraseids.to('cpu').numpy()

            print(logits)
            print(np.shape(logits))


            pred_flat = np.argmax(logits, axis=1).flatten()
              #labels_flat = label_ids.flatten()
              #phrase_ids_flat = phrase_ids.flatten()
          # Store predictions and true labels
            logits_list0.append(logits[:,0].flatten())
            logits_list1.append(logits[:,1].flatten())
            predictions.append(pred_flat)
              #true_labels.append(labels_flat)
              #phrase_id1.append(phrase_ids_flat)

        predictions = list(chain.from_iterable(predictions))
        logits0 = list(chain.from_iterable(logits_list0))
        logits1 = list(chain.from_iterable(logits_list1))
        #phrase_id1 = list(chain.from_iterable(phrase_id1))
        #true_labels = list(chain.from_iterable(true_labels))
        df1 = pd.DataFrame(list(zip(sentences, predictions, logits0, logits1)),
                           columns=['text', 'label', 'logits0', 'logits1'])
        out_filename = f'fourth_rendition_geolocated_id_{labelcomb}_predict.csv'

        df['label'] = df1.label.copy()
        df['logits0'] = df1.logits0.copy()
        df['logits1'] = df1.logits1.copy()

        df.to_csv(predict_dataset+out_filename)
        print('    DONE.')


    return epochs_arr, validation_accuracy, training_loss, validation_loss, f1

if __name__ == '__main__':
    apply_model('runs/epochs20_batch_size32_lr4.002e-6_posneutral_f10.87_acc0.81/model')

