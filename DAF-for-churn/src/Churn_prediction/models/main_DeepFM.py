# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
# from src.Churn_prediction import Models

from deepctr_torch import models
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch import callbacks

import torch

max_epoch = 1000  #default=1000
device = 'cpu'  #default='cpu'
batch_size = 256  #default=256
earlystop_patience = 50  #default=50
dnn_hidden_units = 256, 256
l2_reg_linear = 1e-05  #default=1e-05
l2_reg_dnn = 1e-4  #default=0
dnn_dropout = 0.5  #default=0
dnn_activation = 'relu'  # default='relu'
dnn_use_bn = 0  #default=0
lr = 0.001  # default=0.02
random_seed = 2020  # default=2020


def DeepFM(dnn_hidden_units, l2_reg_linear, l2_reg_dnn, dnn_dropout,
           feature_file):
    datapath = "../../data"
    user_path = "dataset_split"
    label_file = "../../data/labels.csv"

    auc = []
    acc = []
    f1_score = []

    for fold in range(1, 6):
        # Cross validation
        user_file = [
            user_path + "/fold-%d/train.uid.npy" % (fold),
            user_path + "/fold-%d/dev.uid.npy" % (fold)
        ]
        feature_file = feature_file if "all" not in feature_file else feature_file.replace(
            "XXX", str(fold))

        train_type = "basic"
        if "diff" in feature_file:
            train_type = "diff"
        elif "all" in feature_file:
            train_type = "diff_inf"

        # load data
        feature_df = pd.read_csv(os.path.join(datapath, feature_file))
        label = pd.read_csv(label_file)
        label = label[["user_id", "label"]]
        feature_df = feature_df.drop(columns=["label"]).merge(label,
                                                              on=["user_id"])
        uid_list = []
        for file in user_file:
            uid_list.append(np.load(os.path.join(datapath, file)))

        # process feature
        sparse_features = ["first_interval", "last_interval"]
        dense_features = feature_df.columns.drop([
            "user_id", "label", "interval_length", "first_interval",
            "last_interval"
        ]).tolist()
        feature_df[sparse_features] = feature_df[sparse_features].fillna(
            '-1', )
        feature_df[dense_features] = feature_df[dense_features].fillna(0, )
        target = ['label']
        for feat in sparse_features:
            lbe = LabelEncoder()
            feature_df[feat] = lbe.fit_transform(feature_df[feat])
        mms = StandardScaler()
        feature_df[dense_features] = mms.fit_transform(
            feature_df[dense_features])
        fixlen_feature_columns = [
            SparseFeat(feat,
                       vocabulary_size=feature_df[feat].nunique(),
                       embedding_dim=4)
            for i, feat in enumerate(sparse_features)
        ] + [DenseFeat(
            feat,
            1,
        ) for feat in dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns)

        # generate data
        train_df = feature_df.loc[feature_df.user_id.isin(uid_list[0])]
        val_df = feature_df.loc[feature_df.user_id.isin(uid_list[1])]

        X_train = {name: train_df[name] for name in feature_names}
        X_val = {name: val_df[name] for name in feature_names}
        y_train = train_df[target].values
        y_val = val_df[target].values

        # DeepFM model
        use_bn = False if dnn_use_bn == 0 else True
        deep_FM = models.DeepFM(linear_feature_columns,
                                dnn_feature_columns,
                                dnn_hidden_units=dnn_hidden_units,
                                l2_reg_linear=l2_reg_linear,
                                l2_reg_dnn=l2_reg_dnn,
                                dnn_dropout=dnn_dropout,
                                dnn_activation=dnn_activation,
                                dnn_use_bn=use_bn,
                                task='binary',
                                device=device,
                                seed=random_seed)
        optimizer = torch.optim.Adam(deep_FM.parameters(), lr=lr)
        deep_FM.compile(
            'adam',
            "binary_crossentropy",
            metrics=["binary_crossentropy",
                     "auc"])  # ,val_metrics=["auc","binary_crossentropy"]

        # fit
        model_callback = [
            callbacks.EarlyStopping(patience=earlystop_patience,
                                    monitor='val_binary_crossentropy',
                                    mode="min")
        ]

        deep_FM.fit(X_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=max_epoch,
                    validation_data=(X_val, y_val),
                    callbacks=model_callback,
                    verbose=0)

        # model predict
        no_pred = True

        pred = deep_FM.predict(X_val)
        try:
            prob = deep_FM.predict_proba(X_val)[:, 1]
        except:
            prob = pred
        if no_pred:
            pred = pred > 0.5

        auc.append(metrics.roc_auc_score(y_val, prob))
        acc.append(metrics.accuracy_score(y_val, pred))
        f1_score.append(metrics.f1_score(y_val, pred))

    print(train_type)
    print("test_auc", np.mean(auc))
    print("test_acc", np.mean(acc))
    print("f1_score", np.mean(f1_score))
    print("-------")

    with open("results/DeepFM/DeepFM_" + train_type + "_results3.txt",
              "w") as F:
        F.write("test_auc: " + str(np.mean(auc)) + "\n" + "test_acc: " +
                str(np.mean(acc)) + "\n" + "f1_score: " +
                str(np.mean(f1_score)) + "\n")


dnn_hidden_units = [256, 256]
l2_reg_linear = 1e-05
l2_reg_dnn = 1e-4
dnn_dropout = 0.5
DeepFM(dnn_hidden_units, l2_reg_linear, l2_reg_dnn, dnn_dropout,
       "Churn-Features/feature_data.csv")

dnn_hidden_units = [256, 256]
l2_reg_linear = 1e-4
l2_reg_dnn = 1e-4
dnn_dropout = 0.9
DeepFM(dnn_hidden_units, l2_reg_linear, l2_reg_dnn, dnn_dropout,
       "Churn-Features/feature_data_diff.csv")

dnn_hidden_units = [256, 128]
l2_reg_linear = 1e-4
l2_reg_dnn = 1e-05
dnn_dropout = 0.5
DeepFM(dnn_hidden_units, l2_reg_linear, l2_reg_dnn, dnn_dropout,
       "Churn-Features/feature_data_all_fold-XXX.csv")
