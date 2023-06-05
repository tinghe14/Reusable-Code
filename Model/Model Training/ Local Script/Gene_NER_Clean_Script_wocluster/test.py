from utils import *
from model import *
from config import *
import pandas as pd
import torch
from sklearn.metrics import classification_report
import os

def load_model():
    '''
    Function to load best and final models (visualization of overfitting)
    '''
    model = BioBERTTagger()
    model.to(DEVICE)
    # load the best model checkpoint
    best_model_cp = torch.load(SAVED_BEST_TRAIN_MODEL_PATH)
    best_model_epoch = best_model_cp['epoch']
    print(f"[INFO]: Best model was saved at {best_model_epoch} epochs\n")
    # load the final model checkpoint
    last_model_cp = torch.load(SAVED_LAST_TRAIN_MODEL_PATH)
    last_model_epoch = last_model_cp['epoch']
    print(f"[INFO]: Last model was saved at {last_model_epoch} epochs\n")
    return best_model_cp, last_model_cp

def test_performance(model, train_sentences, train_tags, test_sentences, test_tags):
    _, test_dataset = init_data(train_sentences, train_tags, test_sentences, test_tags)
    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )
    model.eval()
    with torch.no_grad():
        y_pred, y_target = [], []
        test_loss_in_epoch = []
        for data in test_data_loader:
            for key, val in data.items():
                data[key] = val.to(torch.device(DEVICE))

            output = model(**data)
            test_loss_in_epoch.append(output["loss"].item())

            for t, p in zip(data['targets'], output["tag_seq"]):
                length = torch.sum(t > 0)
                t = t[1:length + 1].cpu()
                y_pred += p
                y_target += t
        test_y_target = torch.tensor(y_target).detach().cpu().numpy()
        test_y_pred = torch.tensor(y_pred).detach().cpu().numpy()
        report = classification_report(test_y_target, test_y_pred, zero_division=0, digits=4)
        print("[INFO]: ", '_'*50)
        print(report)
        report_dict = classification_report(test_y_target, test_y_pred, zero_division=0, digits=4, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
    return report_df

def test_last_model(last_model_cp, train_sentences, train_tags, test_sentences, test_tags):
    model = BioBERTTagger()
    print('[INFO]: Loading last epoch saved model weights...')
    model.load_state_dict(last_model_cp['model_state_dict'])
    report_df = test_performance(model, train_sentences, train_tags, test_sentences, test_tags)
    print('SAVED PERFORMANCE REPORT OF LAST MODEL FOR TEST DATA IN DISK')
    report_df.to_csv(SAVED_TEST_REPORT_FINAL_PATH)

def test_best_model(best_model_cp, train_sentences, train_tags, test_sentences, test_tags):
    model = BioBERTTagger()
    print('Loading best epoch saved model weights...')
    model.load_state_dict(best_model_cp['model_state_dict'])
    report_df = test_performance(model, train_sentences, train_tags, test_sentences, test_tags)
    print('SAVED PERFORMANCE REPORT OF BEST MODEL FOR TEST DATA IN DISK')
    report_df.to_csv(SAVED_TEST_REPORT_BEST_PATH)

if __name__ == '__main__':
    train_sents = get_train_sent()
    test_sents = get_test_sent()
    train_tags = get_train_tag()
    test_tags = get_test_tag()
    tag_set = get_tag_set()

    best_model_cp, last_model_cp = load_model()

    test_last_model(last_model_cp, train_sents[:100], train_tags[:100], test_sents[:100], test_tags[:100])
    test_best_model(best_model_cp, train_sents[:100], train_tags[:100], test_sents[:100], test_tags[:100])





