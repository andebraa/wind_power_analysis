
import random
import time
import datetime
import os 
import torch
import shutil
import pandas as pd
import numpy as np
import sklearn.model_selection
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig
from sklearn.metrics import precision_recall_fscore_support

import torch
def train_and_verify(batch_size, lr, eps, epochs, model_path):
    

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

    df = pd.read_csv('~/wind_power_analysis/data/annotaion_3000_012label.csv',delimiter=",", usecols=['text', 'label'])
    df = df.reset_index(drop=True)
    #df = df.loc[1:1934,['text', 'label']]

    # shuffle to get CV 
    df_train, df_test = sklearn.model_selection.train_test_split(df, train_size=round(df.shape[0]*0.8), test_size=round(df.shape[0]*0.2),shuffle=True)
    df_train.shape

    sentences = df.text.values
    labels    = df.label.values

    # Load the BERT tokenizer.
    tokenizer = AutoTokenizer.from_pretrained('ltgoslo/norbert2', do_lower_case=False)

    max_len = 0

    # For every sentence...
    for sent in sentences:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    print('Max sentence length: ', max_len)


    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
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
                            max_length = 120,          # Pad & truncate all sentences.NOTE: what does this do
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids       = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels          = labels.astype('int')
    labels          = torch.tensor(labels)



    # Combine the training inputs into a TensorDataset.
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
                batch_size = int(batch_size) # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                batch_size = int(batch_size) # Evaluate with this batch size.
            )


    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        "ltgoslo/norbert2", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 3, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = torch.optim.AdamW(model.parameters(),
                      lr = lr, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = eps # args.adam_epsilon  - default is 1e-8.
                    )



    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)


    def flat_accuracy(preds, labels):
        '''
        Function to calculate the accuracy of our predictions vs labels
        '''
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    def format_time(elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))

        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))





    # This training code is based on the `run_glue.py` script here:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

    # Set the seed value all over the place to make this reproducible.
    seed_val = 69

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val) #might cause issues

    training_stats = []

    total_t0 = time.time()

    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0

        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            LL = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            total_train_loss += LL.loss.item()

            #loss.backward()
            LL[0].backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)

        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in validation_dataloader:

            # Unpack this training batch from our dataloader.
            
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                LL = model(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)

            # Accumulate the validation loss.
            total_eval_loss += LL.loss.item()

            # Move logits and labels to CPU
            logits = LL.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)


        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training_Loss': avg_train_loss,
                'Valid_Loss': avg_val_loss,
                'Valid_Accur': avg_val_accuracy,
                'Training_Time': training_time,
                'Validation_Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

    
    #if folder exists, delete folder
    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    os.mkdir(model_path)


    #pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    # A hack to force the column headers to wrap.
    #df = df.style.set_table_styles([dict(selector="th",props=[('max-width', '70px')])])
    print(df_stats)
    df_stats.to_csv(model_path + 'training_stats.csv')
    model.save_pretrained(f'ltgoslo/norbert_lr{lr}_eps{eps}_batchsize{batch_size}_epochs{epochs}')


    if False:
        plt.rcParams["figure.figsize"] = (12,6)

        # Plot the learning curve.
        plt.plot(df_stats['Training_Loss'], 'b-o', label="Training loss")
        plt.plot(df_stats['Valid_Loss'], 'g-o', label="Validation loss")
        plt.plot(df_stats['Valid_Accur'], 'r-o', label= 'validation accuracy')
        # Label the plot.
        plt.title("Training loss, Validation Loss & accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks(np.arange(epochs))

        plt.savefig(model_path +'train_val_loss.png')
        plt.clf() #clear plots

    #performance on test set
    df = df_test
    df = df.reset_index(drop=True)
    df.shape


    # Create sentence and label lists
    sentences = df.text.values
    labels = df.label.values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
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

    # Set the batch size.
    batch_size = 16

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)


    # Prediction on test set

    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions , true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
      # Add batch to GPU
      batch = tuple(t.to(device) for t in batch)

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

      # Store predictions and true labels
      predictions.append(logits)
      true_labels.append(label_ids)

    print('    DONE.')


    pred_softmax = []
    for i in range(len(true_labels)):
      pred_labels = np.argmax(predictions[i], axis=1).flatten()
      pred_softmax.append(pred_labels)

    precisions = []
    recalls    = []
    f1scores   = []
    for i in range(len(true_labels)):
        #NOTE certain true labels don't appear in pred. this causes warnings that are surpresseed by zero_division = 0
        measures   = precision_recall_fscore_support(true_labels[i], pred_softmax[i], average=None, zero_division = 0)
        precisions.append(measures[0])
        recalls.append(measures[1])
        f1scores.append(measures[2])

    precisions = pd.DataFrame(precisions)
    recalls    = pd.DataFrame(recalls)
    f1scores   = pd.DataFrame(f1scores)

    test_set_stat = pd.concat([pd.DataFrame(precisions.mean(axis = 0),columns=['precision']),pd.DataFrame(recalls.mean(axis = 0),columns=['recall']),pd.DataFrame(f1scores.mean(axis = 0),columns=['f1score'])],axis = 1)
    test_set_stat.to_csv(model_path + 'test_set_stat.csv')

    print(pd.concat([pd.DataFrame(precisions.mean(axis = 0),columns=['precision']),pd.DataFrame(recalls.mean(axis = 0),columns=['recall']),pd.DataFrame(f1scores.mean(axis = 0),columns=['f1score'])],axis = 1))

    print(type(test_set_stat))
    print(test_set_stat.columns)
    return df_stats, test_set_stat

if __name__ == '__main__':

    lrs = np.linspace(1e-8, 0.001, 16)
    #lrs = [1e-5]
    batch_sizes = [16]

    #batch_sizes = np.array((16, 32)) 
    
    f1_scores = np.zeros((len(lrs),len(batch_sizes)))
    val_acc_scores = np.zeros((len(lrs),len(batch_sizes)))
 
    epochs = 10
    eps = 1e-8
    
    #df_stats, test_set_stat = train_and_verify(batch_sizes[0], lrs[0], eps, epochs, model_path) 
    
    fig, ax = plt.subplots(4,4)
     
    plt.suptitle("Training loss, Validation Loss & accuracy")
    ax = ax.ravel()
    figure_ = 0
    for i,batch_size in enumerate(batch_sizes):
        print('batch_size: ', batch_size)
        for j,lr in enumerate(lrs):
            model_path = f'runs/norbert_accuracy_lr{lr}_eps{eps}_batchsize{batch_size}_epochs{epochs}_bool/'
            print('learning rate ', lr)
            df_stats, test_set_stat = train_and_verify(batch_size, lr, eps = eps, epochs = epochs, model_path=model_path)
            # Plot the learning curve.
            print(df_stats)
            ax[figure_].plot(df_stats['Training_Loss'], 'b-o', label="Training loss")
            ax[figure_].plot(df_stats['Valid_Loss'], 'g-o', label="Validation loss")
            ax[figure_].plot(df_stats['Valid_Accur'], 'r-o', label= 'validation accuracy')
            ax[figure_].set_title(f'batch size {batch_size}, lr {lr}')
            ax[figure_].set_xticks(np.arange(epochs))
            ax[figure_].set_xlabel('Epochs')
            ax[figure_].set_ylabel('Loss')
            f1_scores[j,i] = test_set_stat['f1score'].max() 
            val_acc_scores[j,i] = df_stats['Valid_Accur'].max()
            plt.legend() 
            print(figure_)
            figure_ += 1


    fig.savefig('runs/loss_accuracy_validationgrid.png')


    plt.clf()







    plt.subplot(2,1,1)
    plt.plot(lrs,f1_scores[:,0], '-o', label = 'batch_size 16') 
    #plt.plot(lrs,f1_scores[:,1], '-o', label = 'batch_size 32')
    plt.ylabel('F1 score')
    plt.title(f'F1 scores as a function of LR, max of {epochs} epochs. eps: {eps}')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(lrs, val_acc_scores[:,0], '-o', label = 'batch_size 16')
    #plt.plot(lrs, val_acc_scores[:,1], '-o', label = 'batch_size 32')
    plt.title(f'highest validation accuracy over {epochs} epochs. eps: {eps}')
    plt.xlabel('learning rate')
    plt.ylabel('validation accuracy')
    plt.legend()

    plt.subplots_adjust(wspace=1)

    plt.savefig('f1_acc_val.png')
    


