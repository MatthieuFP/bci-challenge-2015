# -*- coding: utf-8 -*-

"""
Created on Sat Apr 3 12:47:39 2020

@author: matthieufuteral-peter
"""

import os
import torch
import numpy as np
import matplotlib.pyplot
import scipy.fft as fft
from scipy.signal import *
from sklearn.metrics import auc, roc_curve
from pywt import wavedec
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace


def bandpass(sig, band, fs):
    B, A = butter(5, np.array(band)/(fs/2), btype='bandpass')
    return lfilter(B, A, sig, axis=0)


def compute_auc(pred, gt):
    fpr, tpr, thresholds = roc_curve(gt, pred, pos_label=1)
    return auc(fpr, tpr)


def compute_precision_recall(pred, gt):
    pass


def custom_bce_loss(output, target, weights: torch.Tensor, eps=1e-30):
    sample_loss = - weights[1] * target * torch.log(output.clamp(min=eps)) \
                  - weights[0] * (1 - target) * torch.log((1 - output).clamp(min=eps))
    return sample_loss.mean()


def in_ipynb():
    try:
        get_ipython()
        return True
    except NameError:
        return False


def load_features(path, mode="train"):
    with open(os.path.join(path, "features.npy"), 'rb') as f:
        features = np.load(f)
    if mode == "train":
        with open(os.path.join(path, 'labels.npy'), 'rb') as f:
            labels = np.load(f)
        return features, labels
    else:
        return features



def wavelet_transform(features):
    coeffs = wavedec(features, wavelet="db1")[:-1]  # Exclude wavelets coefficients from first high pass filtering
    # Too high frequencies to be relevant
    return np.concatenate([coef.reshape(coef.shape[0], -1) for coef in coeffs], axis=-1)


def processed_features(features, wavelets=True, only=False):
    coeffs = wavelet_transform(features)
    features = features.reshape(features.shape[0], -1)
    if wavelets and not only:
        features = np.concatenate((features, coeffs), axis=-1)
        return features
    elif wavelets and only:
        return coeffs
    else:
        return features


def xdawn_features(features, labels=None, XC=None, tangent_space=None):
    if labels is not None:
        XC = XdawnCovariances(nfilter=5)
        tangent_space = TangentSpace(metric='riemann')
        new_features = XC.fit_transform(features, labels)
        output_features = tangent_space.fit_transform(new_features)
        return output_features, XC, tangent_space
    else:
        new_features = XC.transform(features)
        new_features = tangent_space.transform(new_features)
        return new_features



