from config import *
from data_process import *
from model import *
from test import *
from train import *
from utils import *
import argparse


if __name__ == '__main__':
    # ---- data_process.py ---- #
    # Merge File
    merge_files(ORIGIN_TRAIN_FILE, ORIGIN_TEST_FILE, ORIGIN_DEVOP_FILE)

    # Generate Token per Row
    df = clean_token(ORIGIN_MERGE_FILE)

    # Vectorize Tag of Token and Split Data into Train and Test Set
    prepare_data(df)

    # ---- train.py ---- #
    train_sents = get_train_sent()
    test_sents = get_test_sent()
    train_tags = get_train_tag()
    test_tags = get_test_tag()
    tag_set = get_tag_set()

    #  return detailed error message when using GPU training
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    train_bioberttagger(train_sents, train_tags, test_sents, test_tags, tag_set)

    # ---- test.py ---- #
    best_model_cp, last_model_cp = load_model()

    test_last_model(last_model_cp, train_sents, train_tags, test_sents, test_tags)
    test_best_model(best_model_cp, train_sents, train_tags, test_sents, test_tags)