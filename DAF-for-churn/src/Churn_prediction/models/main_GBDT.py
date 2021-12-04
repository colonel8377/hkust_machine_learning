# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 16:50:10 2021

@author: lenovo
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model, metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import os
#from Models import *


def GBDT(estimators, subsample, GBDT_lr, feature_file):
    datapath = "../../data"
    user_path = "dataset_split"
    user_file = ["train.uid.npy", "dev.uid.npy"]

    test_auc = 0
    f1_score = 0
    test_acc = 0
    for fold in range(1, 6):
        # Cross validation
        user_file = [
            user_path + "/fold-%d/train.uid.npy" % (fold),
            user_path + "/fold-%d/dev.uid.npy" % (fold)
        ]
        uid_list = []
        for file in user_file:
            uid_list.append(np.load(os.path.join(datapath, file)))
        #Load Data
        feature_file = feature_file if "all" not in feature_file else feature_file.replace(
            "XXX", str(fold))
        feature_df = pd.read_csv(os.path.join(datapath, feature_file))
        label = pd.read_csv("../../data/labels.csv")
        label = label[["user_id", "label"]]
        feature_df = feature_df.drop(columns=["label"]).merge(label,
                                                              on=["user_id"])

        #Generate data
        drop_columns = ["label"]
        if "user_id" in feature_df.columns:
            drop_columns.append("user_id")
        if "interval_length" in feature_df.columns:
            drop_columns.append("interval_length")

        train_df = feature_df.loc[feature_df.user_id.isin(uid_list[0])]
        val_df = feature_df.loc[feature_df.user_id.isin(uid_list[1])]

        X_train = train_df.drop(drop_columns, axis=1).to_numpy()
        y_train = train_df["label"].to_numpy()
        X_val = val_df.drop(drop_columns, axis=1).to_numpy()
        y_val = val_df["label"].to_numpy()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        GBDT = GradientBoostingClassifier(random_state=2020,
                                          learning_rate=GBDT_lr,
                                          n_estimators=estimators,
                                          max_depth=6,
                                          min_samples_split=2,
                                          min_samples_leaf=1,
                                          subsample=subsample)
        GBDT.fit(X_train, y_train)
        result = GBDT.predict(X_val)
        try:
            prob = GBDT.predict_proba(X_val)[:, 1]
        except:
            prob = result
        test_auc = test_auc + metrics.roc_auc_score(y_val, prob)
        test_acc = test_acc + metrics.accuracy_score(y_val, result)
        f1_score = f1_score + metrics.f1_score(y_val, result)

    if "diff" in feature_file:
        train_type = "Basic+DIF"
    elif "all" in feature_file:
        train_type = "Basic+DIF+DDI"
    else:
        train_type = "Basic"

    print(train_type)
    print("test_auc", test_auc / 5)
    print("test_acc", test_acc / 5)
    print("f1_score", f1_score / 5)
    print("-------")

    with open("results/GBDT/GBDT_" + train_type + "_results.txt", "w") as F:
        F.write("test_auc: " + str(test_auc / 5) + "\n" + "test_acc: " +
                str(test_acc / 5) + "\n" + "f1_score: " + str(f1_score / 5) +
                "\n")


if __name__ == "__main__":
    GBDT(500, 0.9, 0.3, "Churn-Features/feature_data.csv")
    GBDT(600, 1.0, 0.3, "Churn-Features/feature_data_diff.csv")
    GBDT(600, 0.9, 0.25, "Churn-Features/feature_data_all_fold-XXX.csv")
