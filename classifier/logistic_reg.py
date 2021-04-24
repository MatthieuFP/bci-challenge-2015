# -*- coding: utf-8 -*-

"""
Created on Sat Apr 3 12:47:39 2020

@author: matthieufuteral-peter
"""


import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'util'))
from logger import logger
import argparse
import numpy as np
import pandas as pd
from util import compute_auc, load_features, processed_features, xdawn_features, wavelet_transform
from sklearn.linear_model import LogisticRegression


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--wavelets', type=int, default=0)
    parser.add_argument('--xdawn', type=int, default=0)
    parser.add_argument('--save_name', type=str, required=True)
    parser.add_argument('--iter', type=int, default=400)
    parser.add_argument('--penalty', type=str, default="l2")
    parser.add_argument('--solver', type=str, default="lbfgs")
    args = parser.parse_args()

    path_train = os.path.join("inria-bci-challenge", "features", "time_domain", "train")
    path_val = os.path.join("inria-bci-challenge", "features", "time_domain", "dev")
    path_test = os.path.join("inria-bci-challenge", "features", "time_domain", "test")

    train_features, train_labels = load_features(path_train)
    val_features, val_labels = load_features(path_val)
    test_features = load_features(path_test, mode="test")

    if args.xdawn and not args.wavelets:
        train_features, XC, tangent_space = xdawn_features(train_features, labels=train_labels)
        val_features = xdawn_features(val_features, XC=XC, tangent_space=tangent_space)
        test_features = xdawn_features(test_features, XC=XC, tangent_space=tangent_space)

    elif args.xdawn and args.wavelets:

        # Spatial features
        train_features, XC, tangent_space = xdawn_features(train_features, labels=train_labels)
        val_features = xdawn_features(val_features, XC=XC, tangent_space=tangent_space)
        test_features = xdawn_features(test_features, XC=XC, tangent_space=tangent_space)

        train_features2, train_labels = load_features(path_train)
        val_features2, val_labels = load_features(path_val)
        test_features2 = load_features(path_test, mode="test")

        # Time-frequency domain features - wavelets
        train_features2 = wavelet_transform(train_features2)
        val_features2 = wavelet_transform(val_features2)
        test_features2 = wavelet_transform(test_features2)

        # Concatenate both features
        train_features = np.concatenate((train_features, train_features2), axis=-1)
        val_features = np.concatenate((val_features, val_features2), axis=-1)
        test_features = np.concatenate((test_features, test_features2), axis=-1)

    else:
        train_features = processed_features(features=train_features, wavelets=args.wavelets)
        val_features = processed_features(features=val_features, wavelets=args.wavelets)
        test_features = processed_features(features=test_features, wavelets=args.wavelets)

    linear_model = LogisticRegression(penalty=args.penalty, max_iter=args.iter, verbose=True, solver=args.solver)
    linear_model.fit(train_features, train_labels)

    preds = linear_model.predict_proba(val_features)[:, 1]

    print(f"AUC on dev set : {compute_auc(preds, val_labels)}")

    logger.info(f"Test results : {args.save_name} ... in progress")
    test_preds = linear_model.predict_proba(test_features)[:, 1]

    os.makedirs(os.path.join("results", args.save_name), exist_ok=False)
    sample_submission = pd.read_csv(os.path.join("inria-bci-challenge", "SampleSubmission.csv"), sep=',', header=0)
    sample_submission["Prediction"] = test_preds
    sample_submission.to_csv(os.path.join("results", args.save_name, "results.csv"), sep=',', index=False)
