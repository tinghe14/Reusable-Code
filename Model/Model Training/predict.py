# the only one script contain bug but not very important here

from utils import *
from model import *
from config import *
import os

if __name__ == '__main__':
    '''
    @TO-DO:
    ------
    has bug in the code
    '''
    test_sents = get_test_sent()
    test_tags = get_test_tag()
    tag_set = get_tag_set()
    single_sent = test_sents[3]
    single_tag = test_tags[3]
    single_data = Data(sentences=[single_sent], lists_of_tags=[single_tag])

    print('single data', single_data[0]) # contain the dictionary of 3 keys

    # lacking of putting into dataloader??


    model = torch.load(SAVED_TRAIN_MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    for k, v in single_data[0].items():
        single_data[0][k] = v.to(DEVICE)
    output = model(**single_data0)
    print(output)
    print(output['tag_seq'])
