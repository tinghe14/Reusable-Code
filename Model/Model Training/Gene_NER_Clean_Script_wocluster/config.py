# REFERENCE
# 1. save best validation plot and move to script: https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/

ORIGIN_TRAIN_FILE = '/Gene_NER_Clean_Script_wocluster/input/train.tsv'
ORIGIN_TEST_FILE = '/Gene_NER_Clean_Script_wocluster/input/test.tsv'
ORIGIN_DEVOP_FILE = '/Gene_NER_Clean_Script_wocluster/input/devel.tsv'
ORIGIN_MERGE_FILE = '/Users/tinghe/Desktop/NER Improvement Project/Gene_NER_Clean_Script_wocluster/input/merge.tsv'

TEMP_SENT_FILE = '/Gene_NER_Clean_Script_wocluster/input/merge.sent.txt'

RANDOM_STATE = 1 # split training and test datasets

TRAIN_SENT_PATH = '/Gene_NER_Clean_Script_wocluster/input/clean/train_sent.npy'
TEST_SENT_PATH = '/Gene_NER_Clean_Script_wocluster/input/clean/test_sent.npy'
TRAIN_TAG_PATH = '/Gene_NER_Clean_Script_wocluster/input/clean/train_tag.npy'
TEST_TAG_PATH = '/Gene_NER_Clean_Script_wocluster/input/clean/test_tag.npy'
TAG_SET_PATH = '/Gene_NER_Clean_Script_wocluster/input/clean/tag_set.npy'

WORD_PAD_ID = 0
WORD_PAD = '<VOID>'

MAX_SEQ_LEN = 115
PRE_TRAIN_MODEL = 'dmis-lab/biobert-v1.1'

LEARNING_RATE = 3e-5
TRAIN_BATCH_SIZE = 16
SEED_VAL = 2
EPOCHS = 5

import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SAVED_LAST_TRAIN_MODEL_PATH = '/Gene_NER_Clean_Script_wocluster/output/BERT_train_last.pth'

SAVED_BEST_TRAIN_MODEL_PATH = '/Gene_NER_Clean_Script_wocluster/output/BERT_valid_best.pth'
SAVED_TRAIN_VALID_LOSS_PLOT_PATH = '/Gene_NER_Clean_Script_wocluster/output/BERT_valid_best_loss.png'
SAVED_TRAIN_VALID_ACC_PLOT_PATH = '/Gene_NER_Clean_Script_wocluster/output/BERT_valid_best_accuracy.png'

SAVED_TEST_REPORT_FINAL_PATH = '/Gene_NER_Clean_Script_wocluster/output/BERT_test_final_report.csv'
SAVED_TEST_REPORT_BEST_PATH = '/Gene_NER_Clean_Script_wocluster/output/BERT_test_best_report.csv'

