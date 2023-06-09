from utils import *
from model import *
import pandas as pd
import torch
from sklearn.metrics import classification_report
import os
from torch.utils.data import DataLoader


def load_model(args, DEVICE):
    '''
    Function to load best and final models (visualization of overfitting)
    '''
    model = BioBERTTagger()
    model.to(DEVICE)
    # load the best model checkpoint
    best_model_path = os.path.join('../' + args.saved_dir + 'best_model.pth')
    best_model_cp = torch.load(best_model_path)
    best_model_epoch = best_model_cp['epoch']
    print(f"[INFO]: Best model was saved at {best_model_epoch} epochs\n")
    # load the final model checkpoint
    last_model_path = os.path.join('../' + args.saved_dir + 'last_model.pth')
    last_model_cp = torch.load(last_model_path)
    last_model_epoch = last_model_cp['epoch']
    print(f"[INFO]: Last model was saved at {last_model_epoch} epochs\n")
    return best_model_cp, last_model_cp

def test_performance(model, train_sentences, train_tags, test_sentences, test_tags, DEVICE):
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
                data[key] = val.to(DEVICE)

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

def test_last_model(last_model_cp, train_sentences, train_tags, test_sentences, test_tags, args, DEVICE):
    model = BioBERTTagger()
    print('[INFO]: Loading last epoch saved model weights...')
    model.load_state_dict(last_model_cp['model_state_dict'])
    model.to(DEVICE)
    report_df = test_performance(model, train_sentences, train_tags, test_sentences, test_tags, DEVICE)
    print('SAVED PERFORMANCE REPORT OF LAST MODEL FOR TEST DATA IN DISK')
    saved_last_report_path = os.path.join('../' + args.saved_dir + 'last_report.csv')
    report_df.to_csv(saved_last_report_path)

def test_best_model(best_model_cp, train_sentences, train_tags, test_sentences, test_tags, agrs, DEVICE):
    model = BioBERTTagger()
    print('Loading best epoch saved model weights...')
    model.load_state_dict(best_model_cp['model_state_dict'])
    model.to(DEVICE)
    report_df = test_performance(model, train_sentences, train_tags, test_sentences, test_tags, DEVICE)
    print('SAVED PERFORMANCE REPORT OF BEST MODEL FOR TEST DATA IN DISK')
    saved_best_report_path = os.path.join('../' + args.saved_dir + 'final_report.csv')
    report_df.to_csv(saved_best_report_path)

if __name__ == '__main__':
    # --- task.py ---#
    args = get_args()

    train_sents = get_train_sent(args)
    test_sents = get_test_sent(args)
    train_tags = get_train_tag(args)
    test_tags = get_test_tag(args)
    tag_set = get_tag_set(args)

    #  return detailed error message when using GPU training
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    best_model_cp, last_model_cp = load_model(args)

    test_last_model(last_model_cp, train_sents[:100], train_tags[:100], test_sents[:100], test_tags[:100], args)
    test_best_model(best_model_cp, train_sents[:100], train_tags[:100], test_sents[:100], test_tags[:100], args)
