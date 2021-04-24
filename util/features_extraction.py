# -*- coding: utf-8 -*-

"""
Created on Sat Apr 3 12:47:39 2020

@author: matthieufuteral-peter
"""


import os
import argparse
from logger import logger
import pdb
import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from data import load_data
from sklearn.decomposition import FastICA


class features_extraction(load_data):

    def __init__(self, setup="eegnet", mode="train", freq=256, spectral=False, ica=False, normalized=False):
        super(features_extraction, self).__init__(setup=setup)
        assert mode in ["train", "test", "val"], "Mode error : Train or test"
        
        self.mode = mode
        self.freq = freq
        self.spectral = spectral
        self.ica = ica
        self.normalized = normalized

        path_data = os.path.join(os.getcwd(), "inria-bci-challenge")
        fnames = sorted(glob.glob(path_data + f"/{mode}/Data_S*_Sess*.csv"))

        if self.mode != 'test':
            path_labels = os.path.join(path_data, "TrainLabels.csv")
            self.labels = self.process_labels(pd.read_csv(path_labels, sep=r',', header=0))

        self.dataset = {idx: fname for idx, fname in enumerate(fnames)}
        self.ICA = FastICA(n_components=57)

    def load_epoch(self):

        logger.info(f"Start extracting data and building epochs : {self.mode} set")
        samples, labels = [], []
        for item, data_file in tqdm(self.dataset.items()):
            sample, _, labs = self.__getitem__(item)
            samples.append(sample.numpy())
            if self.mode != "test":
                labels.append(labs.numpy())

        if self.mode != "test":
            return np.concatenate(samples, axis=0), np.concatenate(labels, axis=0)
        else:
            return np.concatenate(samples, axis=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name_features", type=str, required=True)
    parser.add_argument("--ica", type=int, default=0)
    parser.add_argument("--spectral", type=int, default=0)
    parser.add_argument("--wavelet", type=int, default=0)
    parser.add_argument("--normalized", type=int, default=0)
    parser.add_argument("--setup", type=str, default="eegnet")
    args = parser.parse_args()

    path = os.path.join(os.getcwd(), "inria-bci-challenge", "features", args.name_features)
    os.makedirs(path, exist_ok=False)
    path_train = os.path.join(path, "train")
    path_dev = os.path.join(path, "dev")
    path_test = os.path.join(path, "test")
    os.makedirs(path_train, exist_ok=False)
    os.makedirs(path_dev, exist_ok=False)
    os.makedirs(path_test, exist_ok=False)


    extractor = features_extraction(setup=args.setup, mode="train", spectral=args.spectral, ica=args.ica)
    samples_train, labels_train = extractor.load_epoch()
    extractor = features_extraction(setup=args.setup, mode="val", spectral=args.spectral, ica=args.ica)
    samples_dev, labels_dev = extractor.load_epoch()

    with open(os.path.join(path_train, "features.npy"), "wb") as f:
        np.save(f, samples_train)
    with open(os.path.join(path_train, "labels.npy"), "wb") as f:
        np.save(f, labels_train)

    with open(os.path.join(path_dev, "features.npy"), "wb") as f:
        np.save(f, samples_dev)
    with open(os.path.join(path_dev, "labels.npy"), "wb") as f:
        np.save(f, labels_dev)

    del samples_train, samples_dev, labels_train, labels_dev

    extractor = features_extraction(setup=args.setup, mode="test", spectral=args.spectral, ica=args.ica)
    samples_test = extractor.load_epoch()

    with open(os.path.join(path_test, "features.npy"), "wb") as f:
        np.save(f, samples_test)

