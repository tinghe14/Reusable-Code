from glob import glob
import pandas as pd
from config import *
from sklearn import preprocessing, model_selection
import numpy as np

def get_text(txt_path):
    with open(txt_path) as file:
        return file.read()

# Merge Files
def merge_files(ORIGIN_TRAIN_FILE, ORIGIN_TEST_FILE, ORIGIN_DEVOP_FILE):
    data = get_text(ORIGIN_TRAIN_FILE)
    data2 = get_text(ORIGIN_TEST_FILE)
    data3 = get_text(ORIGIN_DEVOP_FILE)
    data += data2
    data += data3
    with open(ORIGIN_MERGE_FILE, 'w') as f: 
        f.write(data) 

# Generate One Token per Row
def clean_token(ORIGIN_MERGE_FILE):
    '''
    one token per row
    columns like: token, token_tag_lst, sentenceid
    '''
    sentence_id = 0
    with open(ORIGIN_MERGE_FILE, 'r') as f:
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
def prepare_data(df):
    tag_encoder = preprocessing.LabelEncoder()
    df['TAG_ID'] = tag_encoder.fit_transform(df['token_tag_lst'])
    df['TAG_ID'] = 1 + df['TAG_ID']
    # a list of lists which each sub-list contain the token in that sentence
    sentences = df.groupby('sentenceid')['token'].apply(list).values
    tags = df.groupby('sentenceid')['TAG_ID'].apply(list).values
    tag_set = tag_encoder.classes_ 
    tag_set = np.insert(tag_set, WORD_PAD_ID, WORD_PAD) # padding tag
    (
        train_sentences,
        test_sentences,
        train_tags,
        test_tags
    ) = model_selection.train_test_split(sentences, tags, random_state=RANDOM_STATE, test_size=0.2)
    np.save(TRAIN_SENT_PATH, train_sentences)
    np.save(TEST_SENT_PATH, test_sentences)
    np.save(TRAIN_TAG_PATH, train_tags)
    np.save(TEST_TAG_PATH, test_tags)
    np.save(TAG_SET_PATH, tag_set)

if __name__ == '__main__':
    # Merge File
    #merge_files(ORIGIN_TRAIN_FILE, ORIGIN_TEST_FILE, ORIGIN_DEVOP_FILE)

    # Generate Token per Row
    df = clean_token(ORIGIN_MERGE_FILE)

    # Describe the Data
    summary(df)

    # Vectorize Tag of Token and Split Data into Train and Test Set
    prepare_data(df)




