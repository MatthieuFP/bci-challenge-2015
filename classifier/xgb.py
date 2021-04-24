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
import pandas as pd
from util import compute_auc, load_features, processed_features, xdawn_features
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--wavelets', type=int, default=0)
    parser.add_argument('--xdawn', type=int, default=0)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--ica', type=int, default=0)
    parser.add_argument('--save_name', type=str, required=True)
    args = parser.parse_args()

    if not args.ica:
        path_train = os.path.join("inria-bci-challenge", "features", "time_domain", "train")
        path_val = os.path.join("inria-bci-challenge", "features", "time_domain", "dev")
        path_test = os.path.join("inria-bci-challenge", "features", "time_domain", "test")
    else:
        path_train = os.path.join("inria-bci-challenge", "features", "ica_time_domain", "train")
        path_val = os.path.join("inria-bci-challenge", "features", "ica_time_domain", "dev")
        path_test = os.path.join("inria-bci-challenge", "features", "ica_time_domain", "test")

    train_features, train_labels = load_features(path_train)
    val_features, val_labels = load_features(path_val)
    test_features = load_features(path_test, mode="test")

    if args.xdawn:
        train_features, XC, tangent_space = xdawn_features(train_features, labels=train_labels)
        val_features = xdawn_features(val_features, XC=XC, tangent_space=tangent_space)
        test_features = xdawn_features(test_features, XC=XC, tangent_space=tangent_space)

    else:
        train_features = processed_features(features=train_features, wavelets=args.wavelets)
        val_features = processed_features(features=val_features, wavelets=args.wavelets)
        test_features = processed_features(features=test_features, wavelets=args.wavelets)

    xgb = XGBClassifier(objective="binary:logistic", max_depth=args.max_depth, n_estimators=args.n_estimators)
    xgb.fit(train_features, train_labels)

    preds = xgb.predict_proba(val_features)[:, 1]

    print(f"AUC on dev set : {compute_auc(preds, val_labels)}")

    logger.info(f"Test results : {args.save_name} ... in progress")

    test_preds = xgb.predict_proba(test_features)[:, 1]

    os.makedirs(os.path.join("results", args.save_name), exist_ok=False)
    sample_submission = pd.read_csv(os.path.join("inria-bci-challenge", "SampleSubmission.csv"), sep=',', header=0)
    sample_submission["Prediction"] = test_preds
    sample_submission.to_csv(os.path.join("results", args.save_name, "results.csv"), sep=',', index=False)





