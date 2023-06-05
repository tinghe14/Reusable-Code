from config import *
import pandas as pd 
import numpy as np
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
# @TO-DO:
# from seqeval.metrics import classification_report

def get_train_sent():
    '''
    return np.array of lists
    '''
    sentences = np.load(TRAIN_SENT_PATH, allow_pickle=True)
    return sentences

def get_test_sent():
    sentences = np.load(TEST_SENT_PATH, allow_pickle=True)
    return sentences

def get_train_tag():
    tags = np.load(TRAIN_TAG_PATH, allow_pickle=True)
    return tags
    
def get_test_tag():
    tags = np.load(TEST_TAG_PATH, allow_pickle=True)
    return tags

def get_tag_set():
    tag_set = np.load(TAG_SET_PATH, allow_pickle=True)
    return tag_set

class Data(Dataset):
    def __init__(self, sentences, lists_of_tags): 
        self.sentences = sentences
        self.lists_of_tags = lists_of_tags
        self.tokenizer = AutoTokenizer.from_pretrained(PRE_TRAIN_MODEL, use_fast=True)
        self.seq_len = MAX_SEQ_LEN 
    def __len__(self):
        return len(self.sentences)   
    def __getitem__(self, idx):        
        sentence = self.sentences[idx]
        tags = self.lists_of_tags[idx]       
        # input ids are token indices, numerical representation of tokens building the sequences that will be used as input by the model
        ids, targets = [], []
        # tokenization
        # splitting the sequence into tokens available in the tokenizer vocabulary
        for i, word in enumerate(sentence):
            # wordpeice
            tokens = self.tokenizer(
                word,
                add_special_tokens=False).input_ids           
            ids.extend(tokens)
            token_len = len(tokens)
            if token_len > 1 and tags[i] >= 1 and tags[i] <= 3:
                targets.extend([tags[i]] + [tags[i] + 3] * (token_len - 1))
            else:    
                targets.extend([tags[i]] * (token_len)) 
        # bert requirement
        # start of sentence token = 101, end = 102
        ids = ids[:self.seq_len - 2]
        ids = [101] + ids + [102]
        targets = targets[:self.seq_len - 2]
        targets = [0] + targets + [0]
        # attention and padding
        padding_len = self.seq_len - len(ids) 
        attention_mask = [1] * len(ids) + [0] * padding_len
        # padding token = 1                       
        ids = ids + [1] * padding_len
        targets = targets + [0] * padding_len
        #token_type = [0] * self.seq_len
        return {'input_ids': torch.tensor(ids), 
                'attention_mask': torch.tensor(attention_mask),
                #'token_type_ids': torch.tensor(token_type),
                'targets': torch.tensor(targets) # label
                }

def init_data(train_sents, train_tags, test_sents, test_tags):
    '''
    Function to initialize the Data class
    '''
    train_dataset = Data(
        sentences=train_sents,
        lists_of_tags=train_tags)
    test_dataset = Data(
        sentences=test_sents,
        lists_of_tags=test_tags)
    return train_dataset, test_dataset

class SaveBestModel:
    '''
    class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state
    '''
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
    def __call__(self, curr_valid_loss, epoch, model, optimizer):
        if curr_valid_loss < self.best_valid_loss:
            self.best_valid_loss = curr_valid_loss
            print(f'\nBest validation loss: {self.best_valid_loss}')
            print(f'\nSaving the best model for epoch: {epoch+1}\n')
            torch.save(
                {'epoch': epoch+1,
                 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                 },
                SAVED_BEST_TRAIN_MODEL_PATH
            )

def save_model(epoch, model, optimizer):
    '''
    function to save the trained model after the training completes
    to disk:
    @TO-DO:
    ------
    early sctop might be can add here
    '''
    print(f'Saving the final model...')
    torch.save(
        {
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, SAVED_LAST_TRAIN_MODEL_PATH
    )

def save_plots(history):
    '''
    Function to save the loss and accuracy graphs for training and validaiton
    '''
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(range(len(history['train_loss'])), history['train_loss'], label="train_loss")
    plt.plot(range(len(history['valid_loss'])), history['valid_loss'], label="valid_loss")
    plt.title('Loss in Train and Valid')
    plt.legend(loc="best")
    plt.xlabel("epochs")
    plt.ylabel("losses and accuracies")
    plt.savefig(SAVED_TRAIN_VALID_LOSS_PLOT_PATH)
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(range(len(history['valid_overall_accur'])), history['valid_overall_accur'],
             label=f"valid_overall_accur = {history['valid_overall_accur'][-1]:.4f}")
    plt.plot(range(len(history['valid_key_tag_accur'])), history['valid_key_tag_accur'],
             label=f"minority_classes_valid_accur = {history['valid_key_tag_accur'][-1]:.4f}")
    plt.title('Overall Accuracy and Major Accuracy in Valid')
    plt.legend(loc="best")
    plt.xlabel("epochs")
    plt.ylabel("losses and accuracies")
    plt.savefig(SAVED_TRAIN_VALID_ACC_PLOT_PATH)


if __name__ == '__main__':
    train_sents = get_train_sent()
    test_sents = get_test_sent()
    train_tags = get_train_tag()
    test_tags = get_test_tag()
    tag_set = get_tag_set()

    train_dataset, test_dataset = load_data(train_sents, train_tags, test_sents, test_tags)
    print(type(train_dataset), len(train_dataset))
    print(train_dataset)

