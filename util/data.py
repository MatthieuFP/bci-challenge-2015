# -*- coding: utf-8 -*-

"""
Created on Sat Apr 3 12:47:39 2020

@author: matthieufuteral-peter
"""

import os
from logger import logger
from tqdm import tqdm
import pdb
import glob
import re
import numpy as np
import pandas as pd
import torch
from util import bandpass
from scipy.fft import fft
from scipy.signal import blackman
from sklearn.decomposition import FastICA


class load_data:

    def __init__(self, setup, mode="train", freq=256, normalized=False, spectral=False, ica=False):
        assert mode in ["train", "test", "val"], "Mode error : Train or test"
        assert setup in ["erpenet", "eegnet", "cnn"], "Setup error : erpenet, eegnet or cnn"

        self.mode = mode
        self.freq = freq
        self.setup = setup
        self.normalized = normalized
        self.spectral = spectral
        self.ica = ica

        path_data = os.path.join(os.getcwd(), "inria-bci-challenge")
        fnames = sorted(glob.glob(path_data + f"/{mode}/Data_S*_Sess*.csv"))

        if self.mode != 'test':
            path_labels = os.path.join(path_data, "TrainLabels.csv")
            self.labels = self.process_labels(pd.read_csv(path_labels, sep=r',', header=0))

        self.dataset = {idx: fname for idx, fname in enumerate(fnames)}
        if self.ica:
            self.ICA = FastICA(n_components=57)

    def __getitem__(self, item):

        data_file = self.dataset[item]
        fname = os.path.basename(data_file)

        data = pd.read_csv(data_file, sep=r',', header=0)

        regex = re.compile(r'\d+')
        subject, sess = regex.findall(fname)

        if self.mode != 'test':
            labs = {k[-1]: v for k, v in self.labels.items() if subject == k[0] and sess == k[1]}

        signal, EOG, feedback = data.values[:, 1:-2], data.values[:, -2], data.values[:, -1]
        if self.setup == "erpenet":
            signal = signal[:, 11:]  # drop First channels as mentioned in the paper (focus on P300)

        filtered_signal = bandpass(signal, [1.0, 40.0], self.freq)

        sample = self.windowing(filtered_signal, EOG, feedback, freq=self.freq, spectral=self.spectral, ica=self.ica)

        if self.normalized:
            sample = (sample - sample.mean(axis=0)) / sample.std(axis=0)

        if self.setup == "erpenet":
            signals = torch.FloatTensor(sample[:, :, :-1]).view(sample.shape[0], sample.shape[1], 5, 9).unsqueeze(-3)
            EOG = torch.FloatTensor(sample[:, :, -1])
        elif self.setup == "eegnet" or self.setup == "cnn":
            signals = torch.FloatTensor(sample[:, :, :-1]).permute(0, 2, 1)
            EOG = torch.FloatTensor(sample[:, :, -1])

        if self.mode != 'test':
            return signals, EOG, torch.FloatTensor([labs[k + 1] for k in np.arange(sample.shape[0])])
        else:
            return signals, EOG, None

    def __len__(self):
        return len(self.dataset)

    def preprocessing(self):
        pass

    @staticmethod
    def windowing(signal: np.array, EOG: np.array, feedback: np.array, freq=256, spectral=False, ica=False):
        """
        from Time to spectral domain with windowing => equivalent to SFTF
        200 Freq = 1s
        """
        feedback_idx = np.where(feedback == 1)[0]
        sample = []
        if spectral:
            w = blackman(freq).repeat(signal[:freq].shape[-1]).reshape(*signal[:freq].shape)
        for id in feedback_idx:
            if ica:
                complete_signal = np.concatenate((signal[id: (id + freq)], np.expand_dims(EOG[id: (id + freq)], axis=-1)),
                                                 axis=1)
                ica_signal = FastICA(n_components=57).fit_transform(complete_signal)
                signal_processed = ica_signal[:, :-1]
            else:
                signal_processed = signal[id: (id + freq)]
            if spectral:
                signal_processed = np.abs(fft(signal_processed * w, axis=-1))
                # sgn_spec = (sgn_spec - sgn_spec.mean(axis=0)) / sgn_spec.std(axis=0)
                # Normalize for deep learning purpose
            else:
                signal_processed = signal[id: (id + freq)]
            sample.append(np.concatenate((signal_processed, np.expand_dims(EOG[id: (id + freq)], axis=-1)),
                                         axis=1))
        return np.stack(sample, axis=0)

    @staticmethod
    def process_labels(labels):
        regex = re.compile(r'\d+')

        def str_to_int(tup: tuple):
            assert len(tup) == 3, "Len error"
            return tup[0], tup[1], int(tup[2])

        return {str_to_int(tuple(regex.findall(fname))): lab for fname, lab in zip(labels.IdFeedBack,
                                                                                   labels.Prediction)}


class load_raw:

    def __init__(self, dataset):

        self.dataset = dataset

        path_data = os.path.join(os.getcwd(), "inria-bci-challenge")
        fnames = sorted(glob.glob(path_data + f"/{dataset}/Data_S*_Sess*.csv"))

        if self.dataset != 'test':
            path_labels = os.path.join(path_data, "TrainLabels.csv")
            self.labels = self.process_labels(pd.read_csv(path_labels, sep=r',', header=0))

        self.dataset_names = {idx: fname for idx, fname in enumerate(fnames)}

    def __getitem__(self, item):

        data_file = self.dataset_names[item]
        fname = os.path.basename(data_file)

        data = pd.read_csv(data_file, sep=r',', header=0)

        regex = re.compile(r'\d+')
        subject, sess = regex.findall(fname)

        if self.dataset != 'test':
            labs = {k[-1]: v for k, v in self.labels.items() if subject == k[0] and sess == k[1]}

        signal, EOG, feedback = data.values[:, 1:-2], data.values[:, -2], data.values[:, -1]

        sample = self.windowing(signal, EOG, feedback)

        if self.dataset != 'test':
            return sample, np.array([labs[k + 1] for k in np.arange(sample.shape[0])])
        else:
            return sample, None

    def __call__(self, *args, **kwargs):

        logger.info(f"Start extracting data and building epochs : {self.dataset} set")
        samples, labels = [], []
        for item, data_file in tqdm(self.dataset_names.items()):
            sample, labs = self.__getitem__(item)
            samples.append(sample)
            if self.dataset != "test":
                labels.append(labs)

        if self.dataset != "test":
            return np.concatenate(samples, axis=0), np.concatenate(labels, axis=0)
        else:
            return np.concatenate(samples, axis=0)

    @staticmethod
    def process_labels(labels):
        regex = re.compile(r'\d+')

        def str_to_int(tup: tuple):
            assert len(tup) == 3, "Len error"
            return tup[0], tup[1], int(tup[2])

        return {str_to_int(tuple(regex.findall(fname))): lab for fname, lab in zip(labels.IdFeedBack,
                                                                                   labels.Prediction)}
    @staticmethod
    def windowing(signal: np.array, EOG: np.array, feedback: np.array, freq=256):
        feedback_idx = np.where(feedback == 1)[0]
        sample = []
        for id in feedback_idx:
            sample.append(np.concatenate((signal[id: (id + freq)], np.expand_dims(EOG[id: (id + freq)], axis=-1)),
                                         axis=1))
        return np.stack(sample, axis=0)


if __name__ == "__main__":

    train = load_raw(dataset="train")
    dev = load_raw(dataset="val")
    test = load_raw(dataset="test")

    path = os.path.join("inria-bci-challenge", "features", "raw_data")
    os.makedirs(path, exist_ok=False)
    path_train = os.path.join(path, "train")
    os.makedirs(path_train)
    path_dev = os.path.join(path, "dev")
    os.makedirs(path_dev)
    path_test = os.path.join(path, "test")
    os.makedirs(path_test)

    sample_train, labels_train = train()
    sample_dev, labels_dev = dev()

    with open(os.path.join(path_train, "features.npy"), "wb") as f:
        np.save(f, sample_train)
    with open(os.path.join(path_train, "labels.npy"), "wb") as f:
        np.save(f, labels_train)

    with open(os.path.join(path_dev, "features.npy"), "wb") as f:
        np.save(f, sample_dev)
    with open(os.path.join(path_dev, "labels.npy"), "wb") as f:
        np.save(f, labels_dev)

    sample_test = test()
    with open(os.path.join(path_test, "features.npy"), "wb") as f:
        np.save(f, sample_test)

