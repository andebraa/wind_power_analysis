#! /bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from transformers import AdamW
from transformers import BertForSequenceClassification, AutoTokenizer
import argparse
import logging
import csv





def apply_model():
    
    gpu = False
    predict = True 
    modelname = 'ltgoslo_80/norbert'
    model = BertForSequenceClassification.from_pretrained(modelname, num_labels=2)


    pd.set_option("display.max_colwidth", 10000)

    logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
    logger = logging.getLogger(__name__)
    tokenizer = AutoTokenizer.from_pretrained('ltgoslo/norbert', use_fast=False) #use old tokenizer

    if predict:
        model.eval()

        sentences = [
            "Nei til vindkraft!!!",
            "Jeg elsker vindkraft!!",
            "@abjartnes @Klimastiftelsen @VVandvik Sett fra mitt ståsted så er folk mer opptatt av naturvern, ref vindmøller, enn de er opptatt av klima. \n Etter mitt syn, burde det vært motsatt. \n Det viktigste må være klima, men naturvern må også vektlegges.",
"RT @LaNaturenLeve: Ordføreropprøret mot vindkraft brer seg, @www_ks_no, @HoyreTina, @Rotevatn. \n https://t.co/7G9m2Kw7ut"]
        path_to_winddir = '~/wind_power_analysis/'
        data = pd.read_csv(path_to_winddir+ 'full_geodata_longlat_noforeign.csv',
                            usecols = ['text']
                           )

        #data = data.iloc[len(data) -10 :]

        results = np.zeros(len(data))

        for s in sentences:
            logger.info(s)
            encoding = tokenizer(
                [s], return_tensors="pt", padding=True, truncation=True, max_length=256
            )
            if gpu:
                encoding = encoding.to("cuda")
            input_ids = encoding["input_ids"]
            logger.info(tokenizer.convert_ids_to_tokens(input_ids[0]))
            attention_mask = encoding["attention_mask"]
            outputs = model(input_ids, attention_mask=attention_mask)
            #logger.info(outputs)
        """
        labels = []
        print('starting work on dataset')
        for i, line in data.iterrows():
            line = line.to_string()

            #logger.info(line)
            encoding = tokenizer(
                    [line], return_tensors = 'pt', padding=True, truncation = True, max_length = 256
                    )
            input_ids = encoding['input_ids']
            print('logger')
            logger.info(tokenizer.convert_ids_to_tokens(input_ids[0]))
            attention_mask = encoding['attention_mask']
            outputs = model(input_ids, attention_mask=attention_mask)["logits"].detach().numpy()
            #note; logits are [0, 1], i think|

            labels.append(outputs)

            #logger.info(outputs)

        with open('labels_output.csv','w') as f:
            writer = csv.writer(f)
            writer.writerows(labels)
        """
if __name__ == '__main__':
    apply_model() 
