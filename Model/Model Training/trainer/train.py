from utils import *
from model import *
import random
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import random_split
import os
import time
import datetime
import dateutil.relativedelta
from transformers import get_linear_schedule_with_warmup
from task import *
from torch.utils.data import DataLoader


# construct the argument parser @function

def split_train_valid(train_sentences, train_tags, test_sentences, test_tags, args):
    '''
    Function to split Data into train_data_loader and valid_data_loader
    '''
    train_dataset, _ = init_data(train_sentences, train_tags, test_sentences, test_tags)

    # split training data into a 0.8 training and 0.2 validation set
    test_abs = int(len(train_dataset) * 0.8)
    generator1 = torch.Generator().manual_seed(args.seed)
    train_sub, val_sub = random_split(train_dataset, [test_abs, len(train_dataset) - test_abs], generator=generator1)

    train_data_loader = DataLoader(
        dataset=train_sub,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=2
    )
    valid_data_loader = DataLoader(
        dataset=val_sub,
        batch_size=16,
        shuffle=True,
        num_workers=2
    )
    return train_data_loader, valid_data_loader

def train_model(model, train_data_loader, optimizer, scheduler, DEVICE):
    '''
    Standard function for training model
    '''
    model.train()
    t0 = time.time()
    t0 = datetime.datetime.fromtimestamp(t0)
    total_train_step = 0
    train_loss_in_epoch = []
    for data in train_data_loader:
        for key, val in data.items():
            data[key] = val.to(DEVICE)
        # zero the parameter gradients
        model.zero_grad()
        # forward + backward + optimize
        output = model(**data)
        output["loss"].backward()
        optimizer.step()
        scheduler.step()
        total_train_step += 1
        train_loss_in_epoch.append(output["loss"].item())

    t1 = time.time()
    t1 = datetime.datetime.fromtimestamp(t1)
    spent = dateutil.relativedelta.relativedelta(t1, t0)
    print("[INFO]: Finished Training in 1 epoch, Running Time: %d hours, %d minutes and %d seconds" % (spent.hours, spent.minutes, spent.seconds)) # [11:19] hours, miniutes and seconds
    return train_loss_in_epoch


def valid_model(model, valid_data_loader, tag_set, DEVICE):
    '''
    Standard function for validating model
    '''
    model.eval()
    t0 = time.time()
    t0 = datetime.datetime.fromtimestamp(t0)
    total_val_step = 0
    total_pred, corr_pred, valid_tags, corr_tags = 0, 0, 0, 0
    valid_loss_in_epoch = []
    # predicted_labels, true_labels = [], []
    for data in valid_data_loader:
        with torch.no_grad():
            for key, val in data.items():
                data[key] = val.to(DEVICE)
            output = model(**data)
            valid_loss_in_epoch.append(output["loss"].item())
            for t, p in zip(data['targets'], output["tag_seq"]):
                length = torch.sum(t > 0)
                t = t[1:length + 1]
                #p = p[1:length + 1]
                total_pred += len(p)
                p = torch.tensor(p, device=torch.device(DEVICE))
                t = torch.tensor(t, device=torch.device(DEVICE))
                corr_pred += torch.sum(p == t).item()
                valid_tags += torch.sum(t != list(tag_set).index('O')).item()
                corr_tags += torch.sum((p == t) * (t != list(tag_set).index('O'))).item()
                # Add predicted and true labels to lists
                #p = torch.tensor(p).detach().cpu().numpy()
                #t = torch.tensor(t).detach().cpu().numpy()
                #predicted_labels.extend(p.cpu().numpy())
                #true_labels.extend(t.cpu().numpy())
            total_val_step += 1
    # ------- TO DO-------#
    if valid_tags == 0:
            valid_tags = 1
    t1 = time.time()
    t1 = datetime.datetime.fromtimestamp(t1)
    spent = dateutil.relativedelta.relativedelta(t1, t0)
    print("[INFO]: Finished Validing in 1 epoch, Running Time: %d hours, %d minutes and %d seconds" % (spent.hours, spent.minutes, spent.seconds)) # [11:19] hours, miniutes and seconds
    return valid_loss_in_epoch, corr_pred, total_pred, corr_tags, valid_tags

def train_bioberttagger(train_sentences, train_tags, test_sentences, test_tags, tag_set, args, DEVICE):
    '''
    @TO-DO:
    -------
    1. valid_tags seems always be zero
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = BioBERTTagger()
    model.to(DEVICE)

    # initialize SaveBestModel class
    # invoke this after the training and validation steps of each epoch
    save_best_model = SaveBestModel(args)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    train_data_loader, valid_data_loader = split_train_valid(train_sentences, train_tags, test_sentences, test_tags, args)

    # warms up for num_warmup_steps and then linearly decays to 0 by the end of training
    num_train_steps = int(len(train_data_loader)/int(args.train_batch_size) * 20)
    num_warmup_steps = int(num_train_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

    # loop dataset
    history = {'train_loss': [], 'valid_loss': [], 'valid_overall_accur': [], 'valid_key_tag_accur': []}
    for epoch in range(args.epochs):
        print("[INFO]: -------第 {} 轮训练开始-------".format(epoch + 1))

        train_loss_in_epoch = train_model(model, train_data_loader, optimizer, scheduler, DEVICE)
        history['train_loss'].append(np.mean(train_loss_in_epoch))

        valid_loss_in_epoch, corr_pred, total_pred, corr_tags, valid_tags = valid_model(model, valid_data_loader, tag_set, DEVICE)
        history['valid_loss'].append(np.mean(valid_loss_in_epoch))
        history['valid_overall_accur'].append(corr_pred / total_pred)
        history['valid_key_tag_accur'].append(corr_tags / valid_tags)
        # In each epoch, if the loss has improved compared to the previous best loss, then a new best model gets saved to disk
        save_best_model(np.mean(valid_loss_in_epoch), epoch, model, optimizer)
        print("[INFO]: saved best model in this epoch")

    # after training completes, save the trained model from the final epochs
    # to be able to draw the loss and accuracy plot, but will use best model for testing
    save_model(epoch, model, optimizer, args)
    # save the loss and accuracy plots
    save_plots(history, args)
    print('TRAINING COMPLETE')


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

    train_bioberttagger(train_sents[:100], train_tags[:100], test_sents[:100], test_tags[:100], tag_set, args, DEVICE)

