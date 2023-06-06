import pandas as pd
from sklearn import preprocessing, model_selection
import numpy as np
import os
from task import *


def get_text(txt_path):
    with open(txt_path) as file:
        return file.read()

# Merge Files
def merge_files(args):
    original_train_file = os.path.join('../' + args.input_dir + 'train.tsv')
    original_test_file = os.path.join('../' + args.input_dir + 'test.tsv')
    original_dev_file = os.path.join('../' + args.input_dir + 'devel.tsv')
    data = get_text(original_train_file)
    data2 = get_text(original_test_file)
    data3 = get_text(original_dev_file )
    data += data2
    data += data3
    original_merge_file = os.path.join('../' + args.input_dir + 'merge.tsv')
    with open(original_merge_file, 'w') as f:
        f.write(data) 

# Generate One Token per Row
def clean_token(args):
    '''
    one token per row
    columns like: token, token_tag_lst, sentenceid
    '''
    sentence_id = 0
    original_merge_file = os.path.join(args.input_dir + 'merge.tsv')
    with open('../' + original_merge_file, 'r') as f:
        token, token_tag_lst, sentenceid = [], [], []
        for line in f:
            if len(line.split()) == 2:
                token.append(line.split()[0])
                token_tag_lst.append(line.split()[1])
                sentenceid.append(sentence_id)
                # print('token: ', line.split()[0], 'token tag: ', line.split()[1], 'id: ', sentence_id)
            # move to next sentence
            if len(line.split()) == 0:
                sentence_id += 1
                # print('id + 1: ', sentence_id)
                # exit()
    new_df = pd.DataFrame(list(zip(token, token_tag_lst, sentenceid)), \
                      columns=['token', 'token_tag_lst', 'sentenceid'])
    return new_df    

# Describe the Data
def summary(df):
    print(df['token_tag_lst'].value_counts())

# Vectorize Tag of Token and Split Data into Train and Test Set
def prepare_data(df, args):
    tag_encoder = preprocessing.LabelEncoder()
    df['TAG_ID'] = tag_encoder.fit_transform(df['token_tag_lst'])
    df['TAG_ID'] = 1 + df['TAG_ID']
    # a list of lists which each sub-list contain the token in that sentence
    sentences = df.groupby('sentenceid')['token'].apply(list).values
    tags = df.groupby('sentenceid')['TAG_ID'].apply(list).values
    tag_set = tag_encoder.classes_ 
    tag_set = np.insert(tag_set, args.pad_id, args.pad) # padding tag
    (
        train_sentences,
        test_sentences,
        train_tags,
        test_tags
    ) = model_selection.train_test_split(sentences, tags, random_state=args.split_random_state, test_size=args.split_test_fraction)

    clean_train_sents = os.path.join('../' + args.clean_dir + 'train_sent.npy')
    clean_test_sents = os.path.join('../' + args.clean_dir + 'test_sent.npy')
    clean_train_tags = os.path.join('../' + args.clean_dir + 'train_tag.npy')
    clean_test_tags = os.path.join('../' + args.clean_dir + 'test_tag.npy')
    clean_tag_set = os.path.join('../' + args.clean_dir + 'tag_set.npy')
    np.save(clean_train_sents, train_sentences)
    np.save(clean_test_sents, test_sentences)
    np.save(clean_train_tags, train_tags)
    np.save(clean_test_tags, test_tags)
    np.save(clean_tag_set, tag_set)

if __name__ == '__main__':
    # --- task.py ---#
    args = get_args()

    # Merge File
    merge_files(args)

    # Generate Token per Row
    df = clean_token(args)

    # Describe the Data
    summary(df)

    # Vectorize Tag of Token and Split Data into Train and Test Set
    prepare_data(df, args)




