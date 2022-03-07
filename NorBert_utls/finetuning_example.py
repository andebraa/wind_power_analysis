#! /bin/env python3
# coding: utf-8

import argparse
import logging
import csv 
import numpy as np
import pandas as pd
import torch
import sklearn
from torch.utils import data
from transformers import AdamW
from transformers import BertForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split

# This is an example of fine-tuning NorBert for the sentence classification task
# A Norwegian sentiment classification dataset is available at
# https://github.com/ltgoslo/NorBERT/tree/main/benchmarking/data/sentiment/no


def multi_acc(y_pred, y_test):
    batch_predictions = torch.log_softmax(y_pred, dim=1).argmax(dim=1)
    correctness = batch_predictions == y_test
    acc = torch.sum(correctness).item() / y_test.size(0)
    return acc

def f1_loss(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    #https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2
    
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)
        
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1

def gen_train_test(dataset, train_split = 0.2):
    #https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/4
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=train_split)
    train_data = data.Subset(dataset, train_idx)
    test_data = data.Subset(dataset, test_idx)
    return train_data, test_data


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", 10000)

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--model",
        "-m",
        help="Path to a BERT model (/cluster/shared/nlpl/data/vectors/latest/216/ "
        "or ltgoslo/norbert are possible options)",
        required=False,
    )
    arg("--dataset", "-d", help="Path to a document classification dataset", required=True)
    arg("--gpu", "-g", help="Use GPU?", action="store_true")
    arg("--epochs", "-e", type=int, help="Number of epochs", default=10) # set ephocks
    


    args = parser.parse_args()
    modelname = args.model
    dataset = args.dataset


    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "ltgoslo/norbert2", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 3, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )


    tokenizer = AutoTokenizer.from_pretrained('ltgoslo/norbert2', do_lower_case=False)
    #tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)
    #if args.gpu:
    #    model = BertForSequenceClassification.from_pretrained(modelname, num_labels=3).to("cuda")
    #else:
    #    model = BertForSequenceClassification.from_pretrained(modelname, num_labels=3)
    model.train()
    
    print(args.gpu)

    optimizer = AdamW(model.parameters(), lr=1e-4) # set learning rate

    logger.info("Reading train data...")
    df = pd.read_csv(dataset)
    df.columns = ["text", "label"]
    logger.info("Train data reading complete.")

    test_size = 0.2

    df_train, df_test = sklearn.model_selection.train_test_split(df, 
                                                                 train_size=round(df.shape[0]*0.8), 
                                                                 test_size=round(df.shape[0]*0.2),
                                                                 shuffle=True)
    sentences = df.text.values
    labels    = df.label.values

    # We can freeze the base model and optimize only the classifier on top of it:
    freeze_model = True
    if freeze_model:
        for param in model.base_model.parameters():
            param.requires_grad = False

    logger.info("Tokenizing...")
    '''
    for sent in sentences:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    print('Max sentence length: ', max_len)

    '''


    input_ids = []
    attention_masks = []
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 120,           # Pad & truncate all sentences.
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
    labels          = labels.astype('int')
    labels          = torch.tensor(labels)
    logger.info("Tokenizing finished.")

    #https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    

    batch_size = 16
    train_dataset = data.TensorDataset(input_ids, attention_masks, labels)
    train_iter = data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)

    losses = []
    accuracies = [] 

    for epoch in range(args.epochs):
        losses = 0
        total_train_acc = 0
        for i, (text, mask, label) in enumerate(train_iter):
            optimizer.zero_grad()
            outputs = model(text, attention_mask=mask, labels=label)
            loss = outputs.loss
            losses += loss.item()
            predictions = outputs.logits
            accuracy = multi_acc(predictions, label)
            total_train_acc += accuracy
            loss.backward()
            optimizer.step()
        train_acc = total_train_acc / len(train_iter)
        train_loss = losses / len(train_iter)
        
        logger.info(f"Epoch: {epoch}, Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    model.save_pretrained(f'ltgoslo/norbert_accuracy{train_acc:.4f}')
    

    #for i, (text, mask, label) in enumerate(test_iter):



    # We can try the fine-tuned model on a couple of sentences:
    predict = False

    if predict:
        model.eval()

        sentences = [
            "Nei til vindkraft!!!",
            "Jeg elsker vindkraft!!",
        ]
        path_to_winddir = '~/wind_power_analysis/data/second_rendition_data/'
        data = pd.read_csv(path_to_winddir+ 'second_rendition_geolocated_noemoji.csv',
                            usecols = ['text'] 
                           )
        
        #data = data.iloc[len(data) -10 :]

        results = np.zeros(len(data)) 

        for s in sentences:
            logger.info(s)
            encoding = tokenizer(
                [s], return_tensors="pt", padding=True, truncation=True, max_length=256
            )
            if args.gpu:
                encoding = encoding.to("cuda")
            input_ids = encoding["input_ids"]
            logger.info(tokenizer.convert_ids_to_tokens(input_ids[0]))
            attention_mask = encoding["attention_mask"]
            outputs = model(input_ids, attention_mask=attention_mask)
            #logger.info(outputs)
        labels = []
        print('starting work on dataset')
        for i, line in data.iterrows():
            line = line.to_string()
            
            #logger.info(line)
            encoding = tokenizer(
                    [line], return_tensors = 'pt', padding=True, truncation = True, max_length = 256
                    ) 
            input_ids = encoding['input_ids'] 
            logger.info(tokenizer.convert_ids_to_tokens(input_ids[0])) 
            attention_mask = encoding['attention_mask'] 
            outputs = model(input_ids, attention_mask = attention_mask)[0] #note; logits are [0, 1], i think| 
            
            labels.append(outputs) 

            #logger.info(outputs) 
        
        with open('labels_output_test.csv','w') as f:
            writer = csv.writer(f)
            writer.writerows(labels)

