import os

import joblib
import numpy as np
import pandas as pd
import logging
import argparse
import os
import sys
import torch
from deepctr_torch import callbacks
from deepctr_torch import models
import sys

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics, svm

sys.path.append(os.path.join(os.getcwd(), "."))
sys.path.append(os.path.join(os.getcwd(), ".."))
from helpers import utils, data_loader


class base_model:
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--lr",
                            type=float,
                            default=0.02,
                            help="Learning rate")
        return parser

    def __init__(self):
        super().__init__()
        pass

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X, y, print_recall=False, no_pred=False):
        pred = self.classifier.predict(X)
        try:
            prob = self.classifier.predict_proba(X)[:, 1]
        except:
            prob = pred
        if no_pred:
            pred = pred > 0.5
        # Accuracy
        acc = metrics.accuracy_score(y, pred)
        # AUC
        auc = metrics.roc_auc_score(y, prob)
        # F1 score
        f1 = metrics.f1_score(y, pred)
        # Precision and Recall
        precision = metrics.precision_score(y, pred)
        recall = metrics.recall_score(y, pred)
        if print_recall:
            print("acc: {0:.3f}, auc: {1:.3f} f1: {2:.3f}".format(
                acc, auc, f1))
            print(metrics.classification_report(y, pred, digits=3))
        return [auc, acc, f1, precision, recall]

    def test(self, data_loader, no_pred=False):
        train_r = self.predict(data_loader.X_train,
                               data_loader.y_train,
                               no_pred=no_pred)
        val_r = self.predict(data_loader.X_val,
                             data_loader.y_val,
                             no_pred=no_pred)
        return train_r, val_r


class DeepFM(base_model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--max_epoch",
                            type=int,
                            default=1000,
                            help="Max iterations for training.")
        parser.add_argument("--device", type=str, default='cpu')
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument(
            '--earlystop_patience',
            type=int,
            default=50,
            help=
            'Tolerance epochs of early stopping, set to -1 if not use early stopping.'
        )
        parser.add_argument(
            '--dnn_hidden_units',
            nargs='+',
            type=int,
            help='The layer number and units in each layer of DNN.')
        parser.add_argument('--l2_reg_linear', type=float, default=1e-05)
        parser.add_argument('--l2_reg_dnn', type=float, default=0)
        parser.add_argument('--dnn_dropout', type=float, default=0)
        parser.add_argument('--dnn_activation', type=str, default='relu')
        parser.add_argument('--dnn_use_bn', type=int, default=0)
        return base_model.parse_model_args(parser)

    def __init__(self, args, data_loader):
        use_bn = False if args.dnn_use_bn == 0 else True
        self.classifier = models.DeepFM(data_loader.linear_feature_columns,
                                        data_loader.dnn_feature_columns,
                                        dnn_hidden_units=args.dnn_hidden_units,
                                        l2_reg_linear=args.l2_reg_linear,
                                        l2_reg_dnn=args.l2_reg_dnn,
                                        dnn_dropout=args.dnn_dropout,
                                        dnn_activation=args.dnn_activation,
                                        dnn_use_bn=use_bn,
                                        task='binary',
                                        device=args.device,
                                        seed=args.random_seed)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=args.lr)
        self.classifier.compile('adam',
                                "binary_crossentropy",
                                metrics=["binary_crossentropy", "auc"], )
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.earlystop_patience = args.earlystop_patience

    def fit(self, X, y, val_X, val_y):
        model_callback = [
            callbacks.EarlyStopping(patience=self.earlystop_patience,
                                    monitor='val_binary_crossentropy',
                                    mode="min")
        ]

        self.classifier.fit(X,
                            y,
                            batch_size=self.batch_size,
                            epochs=self.max_epoch,
                            validation_data=(val_X, val_y),
                            callbacks=model_callback,
                            verbose=0)

    def model_predict(self, X):
        return self.classifier.predict(X)


class GBDT(base_model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--estimators",
                            type=int,
                            default=200,
                            help="The number of boosting stages to perform.")
        parser.add_argument(
            "--subsample",
            type=float,
            default=1,
            help=
            "The fraction of samples to be used for fitting the individual base learners.(<=1.0)"
        )
        parser.add_argument(
            "--max_depth",
            type=int,
            default=3,
            help="Maximum depth of the individual regression estimators.")
        parser.add_argument(
            "--min_samples_split",
            type=int,
            default=2,
            help=
            "The minimum number of samples required to split an internal node."
        )
        parser.add_argument(
            "--min_samples_leaf",
            type=int,
            default=1,
            help="The minimum number of samples required to be at a leaf node."
        )
        return base_model.parse_model_args(parser)

    def __init__(self, args):
        self.classifier = GradientBoostingClassifier(
            random_state=args.random_seed,
            learning_rate=args.lr,
            n_estimators=args.estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            subsample=args.subsample)
        self.model_path = "Checkpoints/GBDT/"
        self.feature_type = args.feature_file

    def model_predict(self, X):
        self.classifier.predict(X)

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(os.path.join(model_path, "GBDT.pkl"))
        joblib.dump(
            self.classifier,
            os.path.join(model_path, "GBDT_%s.pkl" % (self.feature_type)))
        logging.info('Save model to ' + model_path[:50] + '...')


class LR(base_model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--epoches",
                            type=int,
                            default=1000,
                            help="Max iteractions")
        parser.add_argument("--tol",
                            type=float,
                            default=1e-4,
                            help="Max toleration.")
        parser.add_argument("--solver", type=str, default="lbfgs")
        return base_model.parse_model_args(parser)

    def __init__(self, args):
        self.classifier = LogisticRegression(max_iter=args.epoches,
                                             tol=args.tol,
                                             solver=args.solver,
                                             random_state=args.random_seed)
        self.feature_type = args.feature_file
        self.model_path = "Checkpoints/LR/"

    def model_predict(self, X):
        self.classifier.predict(X)

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(os.path.join(model_path, "LR.pkl"))
        joblib.dump(
            self.classifier,
            os.path.join(model_path, "LR_%s.pkl" % (self.feature_type)))
        logging.info('Save model to ' + model_path[:50] + '...')


class MLP(base_model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--embed_size",
                            type=int,
                            default=100,
                            help="Hidden layer size for MLP.")
        parser.add_argument("--hidden_size",
                            type=int,
                            default=100,
                            help="Hidden layer size for MLP.")
        parser.add_argument("--epoches",
                            type=int,
                            default=3000,
                            help="Max iteractions")
        return base_model.parse_model_args(parser)

    def __init__(self, args):
        hidden_size = [args.embed_size]
        if args.hidden_size > 1:
            hidden_size.append(args.hidden_size)

        self.classifier = MLPClassifier(hidden_layer_sizes=hidden_size,
                                        learning_rate_init=args.lr,
                                        max_iter=args.epoches,
                                        learning_rate="adaptive",
                                        random_state=args.random_seed,
                                        verbose=False,
                                        solver="adam",
                                        alpha=5e-4)

    def model_predict(self, X):
        self.classifier.predict(X)


class RF(GBDT):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument(
            "--max_features",
            type=float,
            default=1.0,
            help="The number of features considered in each time of splitting."
        )
        return GBDT.parse_model_args(parser)

    def __init__(self, args):
        self.classifier = RandomForestClassifier(
            random_state=args.random_seed,
            oob_score=True,
            n_estimators=args.estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features)
        self.model_path = "Checkpoints/RF/"
        self.feature_type = args.feature_file

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        utils.check_dir(os.path.join(model_path, "RF.pkl"))
        joblib.dump(
            self.classifier,
            os.path.join(model_path, "RF_%s.pkl" % (self.feature_type)))
        logging.info('Save model to ' + model_path[:50] + '...')


class SVM(base_model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--C", type=float, default=1.0)
        parser.add_argument("--kernel",
                            type=str,
                            default='rbf',
                            help="Kernel for SVM classifier.")
        parser.add_argument(
            "--gamma",
            type=float,
            default=-1,
            help="Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.")
        parser.add_argument(
            "--degree",
            type=int,
            default=3,
            help="Degree of the polynomial kernel function (‘poly’).")
        return base_model.parse_model_args(parser)

    def __init__(self, args):
        if args.kernel == 'linear':
            self.classifier = svm.SVC(C=args.C, kernel=args.kernel)
        else:
            gamma = args.gamma
            if gamma == -1:
                gamma = 'scale'
            if args.kernel in ["rbf", "sigmoid"]:
                self.classifier = svm.SVC(C=args.C,
                                          kernel=args.kernel,
                                          gamma=gamma)
            if args.kernel == "poly":
                self.classifier = svm.SVC(C=args.C,
                                          kernel=args.kernel,
                                          gamma=gamma,
                                          degree=args.degree)

    def model_predict(self, X):
        self.classifier.predict(X)


def parse_global_args(parser):
    parser.add_argument("--test_only",
                        type=bool,
                        default=False,
                        help="Whether test the dataset only")
    parser.add_argument("--note",
                        type=str,
                        default="test",
                        help="Note to add for log file name.")
    parser.add_argument('--verbose',
                        type=int,
                        default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--log_file',
                        type=str,
                        default='',
                        help='Logging file path')
    parser.add_argument('--random_seed',
                        type=int,
                        default=2020,
                        help='Random seed for all, numpy and pytorch.')
    parser.add_argument("--save_model", type=int, default=0)
    return parser


if __name__ == "__main__":
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument(
        "--model_name",
        type=str,
        default="GBDT",
        help="model name(LR, SVM, MLP, GBDT, DeepFM, or RF)")
    init_args, init_extras = init_parser.parse_known_args()
    model_name = eval('{0}'.format(init_args.model_name))

    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = model_name.parse_model_args(parser)
    parser = data_loader.Data_loader.parse_data_args(parser)
    args, extras = parser.parse_known_args()
    args.model_name = init_args.model_name

    log_args = [args.note, str(args.random_seed)]
    log_file_name = '_'.join(log_args).replace(' ', '_')
    log_file_dir = args.model_name

    if args.log_file == '':
        args.log_file = 'logs/{}/{}.txt'.format(log_file_dir, log_file_name)

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = [
        'check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
        'regenerate', 'sep', 'train', 'verbose'
    ]
    arg_str = utils.format_arg_str(args, exclude_lst=exclude)
    logging.info(arg_str)

    fold_mean = [[], [], []]
    for fold in range(1, 6):
        # Cross validation
        user_file = [
            args.user_path + "/fold-%d/train.uid.npy" % (fold),
            args.user_path + "/fold-%d/dev.uid.npy" % (fold)
        ]
        feature_file = args.feature_file if "all" not in args.feature_file else args.feature_file.replace(
            "XXX", str(fold))
        train_type = "basic"
        if "diff" in args.feature_file:
            train_type = "diff"
        elif "all" in args.feature_file:
            train_type = "diff_inf"

        if args.model_name in ["DeepFM"]:
            loader = data_loader.FM_Data_loader(args.datapath, feature_file,
                                                args.label_file, user_file)
            loader.generate_data()
            classifier = model_name(args, loader)
            classifier.fit(loader.X_train, loader.y_train, loader.X_val,
                           loader.y_val)
            no_pred = True
        else:
            loader = data_loader.Data_loader(args.datapath, feature_file,
                                             args.label_file, user_file)
            loader.generate_data()
            classifier = model_name(args)
            classifier.fit(loader.X_train, loader.y_train)
            no_pred = False

        results = classifier.test(loader, no_pred=no_pred)

        Output = [args.model_name, str(fold), train_type, arg_str]
        for i, state in enumerate(["train", "validation"]):
            logging.info("Fold {} : #{} set results:".format(fold, state))
            for j, metric in enumerate(
                    ["auc", "acc", "f1", "precision", "recall"]):
                logging.info("--# {0}: {1:.3f}".format(metric, results[i][j]))
                Output.append("{:.3f}".format(results[i][j]))

        fold_mean[0].append(results[1][0])
        fold_mean[1].append(results[1][1])
        fold_mean[2].append(results[1][2])

        if args.save_model:
            model_path = os.path.join("Checkpoints", args.model_name,
                                      str(fold))
            os.makedirs(os.path.join("Checkpoints", args.model_name),
                        exist_ok=True)
            classifier.save_model()

        with open("All_results.csv", "a") as F:
            F.write("\t".join(Output) + "\n")

    logging.info(
        "Cross validation  auc: %.3f acc: %.3f f1: %.3f" %
        (np.mean(fold_mean[0]), np.mean(fold_mean[1]), np.mean(fold_mean[2])))
