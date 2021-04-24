# -*- coding: utf-8 -*-

"""
Created on Sat Apr 3 12:47:39 2020

@author: matthieufuteral-peter
"""



import os
import pdb
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import ERPENet, EEGNet, ConvNet
from util import load_data, compute_precision_recall, compute_auc, in_ipynb, custom_bce_loss
from uuid import uuid4
from logger import logger
from tqdm.auto import tqdm
if in_ipynb():
    import ipdb
    pdb = ipdb


def train(epoch, model, train_loader, optimizer, use_cuda, train_loss, device, args):

    optimizer.zero_grad()
    model.train()
    train_batch_loss = []

    for batch_idx, (sample, target) in tqdm(enumerate(train_loader, 1)):

        batch_size, t = sample.size(0), sample.size(1)

        if args.gaussian_noise:
            torch.normal(mean=0.0, std=args.gaussian_noise * torch.ones_like(sample))

        if use_cuda:
            sample, target = sample.cuda(), target.cuda()

        if args.model == "erpenet":
            output, reconstruct = model(sample, batch_size, t)
            output = output.flatten()
        else:
            output = model(sample)

        # Handle unbalanced dataset
        wgts = torch.where(target == 1, .3, .7)
        criterion = nn.BCELoss(wgts)
        loss = criterion(output.flatten(), target)
        if args.model == "erpenet":
            criterion2 = nn.MSELoss()
            loss_reconstruct = criterion2(reconstruct, sample)
            loss += 3/2 * loss_reconstruct
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_batch_loss.append(loss.data.item())

    train_loss.append(np.mean(train_batch_loss))
    logger.info('Train loss Epoch {} : {}'.format(epoch, train_loss[-1]))

    return model, train_loss


def test(epoch, model, loader, use_cuda, val_loss, device, mode, args):

    model.eval()
    val_batch_loss, val_batch_auc = [], []
    pred = []
    if mode == "val":
        labels = []

    criterion = nn.BCELoss()

    for batch_idx, (sample, target) in tqdm(enumerate(loader, 1)):

        batch_size, t = sample.size(0), sample.size(1)
        if use_cuda:
            sample = sample.cuda()
            if mode == "val":
                target = target.cuda()

        with torch.no_grad():
            if args.model == "erpenet":
                output, reconstruct = model(sample, batch_size, t)
                output = output.flatten()
            else:
                output = model(sample)

        if mode == "val":
            loss = criterion(output.flatten(), target)
            val_batch_loss.append(loss.data.item())

            pred.append(output.cpu().numpy())
            labels.append(target.cpu().numpy())

        else:
            pred.append(output.cpu().numpy())

    if mode == "val":
        val_loss.append(np.mean(val_batch_loss))
        logger.info("Validation loss Epoch {} : {}".format(epoch, val_loss[-1]))

        pred = np.concatenate(pred)
        labels = np.concatenate(labels)

        auc_score = compute_auc(pred, labels)
        logger.info("Validation AUROC score Epoch {} : {}".format(epoch, auc_score))

        return val_loss, auc_score

    else:

        return np.concatenate(pred)


def main(model, epochs, train_loader, val_loader, optimizer, use_cuda, device, args):

    train_loss, val_loss, val_auc, epoch_time = [], [], [], []

    for epoch in range(1, epochs + 1):
        logger.info("Epoch {} - Start TRAINING".format(epoch))
        t0 = time.time()

        # Training step
        model, train_loss = train(epoch, model, train_loader, optimizer, use_cuda, train_loss, device, args)

        # Testing mode - test on the validation set
        val_loss, auc_score = test(epoch, model, val_loader, use_cuda, val_loss, device, mode="val", args=args)
        val_auc.append(auc_score)

        if epoch == 1:
            logger.info("Save model ... Epoch 1")
            index = 0
            ref_score = auc_score
            torch.save(model.state_dict(), os.path.join("results", args.name_saved, "model.pt"))
        else:
            if auc_score > ref_score:
                ref_score = auc_score
                index = 0
                torch.save(model.state_dict(), os.path.join("results", args.name_saved, "model.pt"))
                logger.info("Save model ... Epoch {}".format(epoch))
            else:
                index += 1

        if index > args.patience:
            print("Stop training... Patience reached")
            break

        time_elapsed = time.time() - t0
        logger.info("Epoch {} - Time elapsed : {}".format(epoch, time_elapsed))
        epoch_time.append(time_elapsed)

    logger.info("Average time per epoch : {}".format(np.mean(epoch_time)))

    return model, train_loss, val_loss, val_auc, epoch_time


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--features', type=str, default="time_domain")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=.25)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--name_saved', type=str)
    parser.add_argument('--gaussian_noise', type=float, default=0.0)
    args = parser.parse_args()

    print("Model : {}".format(args.model.upper()))
    use_cuda = torch.cuda.is_available()
    print(f"USE CUDA : {use_cuda}")
    # torch.manual_seed(42)

    os.makedirs(os.path.join("results", args.name_saved), exist_ok=False)

    if not args.debug:
        pdb.set_trace = lambda: None

    RUN_ID = str(uuid4())[0:5]
    print('RUN_ID : {}'.format(RUN_ID))
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load data
    def load_np(dataset, file):
        with open(os.path.join("inria-bci-challenge", "features", args.features, dataset, file), 'rb') as f:
            return np.load(f)

    train_features = load_np("train", "features.npy")
    train_labels = load_np("train", "labels.npy")
    dev_features = load_np("dev", "features.npy")
    dev_labels = load_np("dev", "labels.npy")

    train_set = TensorDataset(torch.Tensor(train_features), torch.Tensor(train_labels))
    dev_set = TensorDataset(torch.Tensor(dev_features), torch.Tensor(dev_labels))

    # train_set = load_data(mode="train", setup=args.model)
    # dev_set = load_data(mode="val", setup=args.model)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True, num_workers=0)



    # Model
    if args.model == "erpenet":
        model = ERPENet(dropout=args.dropout)  # Same channels as original paper
    elif args.model == "eegnet":
        model = EEGNet(dropout_rate=args.dropout)
    elif args.model == "cnn":
        model = ConvNet(dropout_rate=args.dropout)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    model, train_loss, val_loss, val_auc, epoch_time = main(model, args.epochs, train_loader, val_loader, optimizer,
                                                            use_cuda, device, args)


    # Test

    test_features = load_np("test", "features.npy")
    test_set = TensorDataset(torch.Tensor(test_features), torch.zeros(test_features.shape[0]))
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info("Start Testing...")
    model.load_state_dict(torch.load(os.path.join("results", args.name_saved, "model.pt"), map_location=device))
    model.eval()

    pred = test(epoch=1, model=model, loader=test_loader, use_cuda=use_cuda, val_loss=[], device=device, mode="test",
                args=args)

    sample_submission = pd.read_csv(os.path.join("inria-bci-challenge", "SampleSubmission.csv"), sep=',', header=0)
    sample_submission["Prediction"] = pred
    sample_submission.to_csv(os.path.join("results", args.name_saved, "results.csv"), sep=',', index=False)

    #np.savetxt(os.path.join("results", args.name_saved, "results.txt"), pred, delimiter=',')

    logger.info("Testing Done !")










