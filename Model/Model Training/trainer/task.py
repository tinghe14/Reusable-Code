from data_process import *
from train import *
from utils import *
from test import *
import argparse

def get_args():
    '''
    Argument parser
    @Return: dict of arguments
    '''
    parser = argparse.ArgumentParser(description='Gene NER')
    # input
    parser.add_argument('--input_dir', type=str, default='../Gene_NER_Clean_Script_wocluster/input/', help='[GCP location] directory of original trian, test, dev and merge files')
    parser.add_argument('--split_random_state', type=int, default=1, help='random state to split train/test sets (default:1)')
    parser.add_argument('--split_test_fraction', type=float, default=0.2, help='fraction of test dataset among train dataset (default:0.2)')

    # clean
    parser.add_argument('--clean_dir', type=str, default='../Gene_NER_Clean_Script_wocluster/input/clean/', help='[GCP location] directory of clean train, test, train_tag,  test_tag and tag_set files')

    # output
    parser.add_argument('--saved_dir', type=str, default='../Gene_NER_Clean_Script_wocluster/output/', help='[GCP_location] directory of saved models, plots and results')

    # train
    parser.add_argument('--epochs', type=int, default=3, help='number of epochs to train (default:3)')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate (default:3e-5)')
    parser.add_argument('--train_batch_size', type=int, default=16, help='training batch size (default:16)')

    # environment
    parser.add_argument('--seed', type=int, default=1, help='random seed (default:1)')

    # embedding
    parser.add_argument('--pad_id', type=int, default=0, help='embedding word_pad_id (default:0)')
    parser.add_argument('--pad', type=str, default='<VOID>', help='embedding word_pad (default:<VOID>)')

    # training
    # parser.add_argument('--pretrained_model', type=str, default='dmis-lab/biobert-v1.1', help='pretrained model (default:biobert-v1.1)')
    # parser.add_argument('--max_seq_len', type=int, default=115, help='maximum sequence length (default: 115)')

    args = parser.parse_args()
    return args

def main():
    '''
    run the project
    '''
    # ---- main.py ---- #
    args = get_args()
    #device = torch.device("cuda:%d" % args.gpu_device)
    args.result = {}

    # ---- data_process.py ---- #
    # Merge File
    merge_files(args)

    # Generate Token per Row
    df = clean_token(args)

    # Describe the Data
    summary(df)

    # Vectorize Tag of Token and Split Data into Train and Test Set
    prepare_data(df, args)

    # ---- train.py ---- #
    train_sents = get_train_sent(args)
    test_sents = get_test_sent(args)
    train_tags = get_train_tag(args)
    test_tags = get_test_tag(args)
    tag_set = get_tag_set(args)

    #  return detailed error message when using GPU training
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_bioberttagger(train_sents[:100], train_tags[:100], test_sents[:100], test_tags[:100], tag_set, args, DEVICE)

    # ---- test.py ---- #
    best_model_cp, last_model_cp = load_model(args)

    test_last_model(last_model_cp, train_sents[:100], train_tags[:100], test_sents[:100], test_tags[:100], args)
    test_best_model(best_model_cp, train_sents[:100], train_tags[:100], test_sents[:100], test_tags[:100], args)

if __name__ == '__main__':
    # ---- main.py ---- #
    main()