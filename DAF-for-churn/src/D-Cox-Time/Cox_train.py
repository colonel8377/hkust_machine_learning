import argparse
import gc
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torchtuples as tt
from pycox.evaluation import EvalSurv
from pycox.models import CoxTime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn

sys.path.append(os.path.join(os.getcwd(), ".."))
from utils import utils


class LBSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)


class Hazard_net(nn.Module):
    def __init__(self, feature_num, length):
        super().__init__()
        beta = torch.randn((feature_num, length)).float()
        self.beta = nn.Parameter(beta)

        self.feature_num = feature_num
        self.length = length

    def forward(self, x_input, t):
        # x_input: batch_size * feature_num * length
        x_input = x_input.float()
        y = self.time_mul(x_input, t)  # y: batch_size * 1
        return y

    def time_mul(self, x_input, t):
        # x_input: batch_size * feature * length
        # t: batch_size * 1
        x_beta = (x_input * self.beta).sum(dim=1)  # batch_size*length
        y = torch.gather(x_beta, dim=1,
                         index=t.long() - 1)  # one hot embedding
        return y

    def predict(self, x_input, t):
        return self.forward(x_input, t)


class Dataset_loader():
    # Load data from dataset
    def __init__(
            self,
            data_path,
            fold_define,
            max_length=30,
            continous_features=[],
            categorial_features=[],
            distance_features=[],
            onehot=False,
            distance_level=5,
            onehot_dim=5,
    ):
        '''
        Args:
            data_path: path with dataset
            max_length: maximum length for D-Cox-Time (30 in paper )
            continous_features: continue feature list
            categorial_features: discrete feature list
            distance_features: difficulty distance feature (i.e. PPD)
            onehot: whether do onehot embedding for basic features X
            distance_level: embedding dim for PPD
            onehot_dim: embedding dim for basic features ( works only when onehot=True)
        '''
        self.data_path = data_path
        self.fold_define = fold_define
        self.max_length = max_length
        self.continous_features = continous_features
        self.categorial_features = categorial_features
        self.distance_features = distance_features
        self.onehot = onehot
        self.bins = onehot_dim
        self.distance_level = distance_level
        self._read_data()

    def construct_input_data(self, fold=1):
        self.t0 = time.time()
        train = np.load(
            os.path.join(self.fold_define, "fold-%d/train.uid.npy" % (fold)))
        test = np.load(
            os.path.join(self.fold_define, "fold-%d/dev.uid.npy" % (fold)))

        self._get_dataset_idx(train, test)
        self._standardize(bins=self.bins, diff_level=self.distance_level)
        logging.info('Consturction done! [{:<.2f} s]'.format(time.time() -
                                                             self.t0) +
                     os.linesep)

    def load_data(self, dataset="train"):
        # construct cox-format data for training
        logging.info("Generating data for {} set...".format(dataset))
        if dataset == "train":
            return self._generate_data(self.x_train_encode,
                                       self.User_list[self.dataset_idx[0]],
                                       self.user_start[0], self.user_end[0])
        elif dataset == "test":
            return self._generate_data(self.x_test_encode,
                                       self.User_list[self.dataset_idx[1]],
                                       self.user_start[1], self.user_end[1])
        else:
            logging.info("Dataset unknown: {}".format(dataset))
            return None

    def _read_data(self):
        # read data
        logging.info("Loading data from \"{}\" ...".format(self.data_path))
        self.X_features = np.load(
            os.path.join(self.data_path, "X_features.npy"))
        self.User_list = np.load(os.path.join(self.data_path, "User_list.npy"))
        feature_list = np.load(os.path.join(self.data_path,
                                            "feature_list.npy"),
                               allow_pickle=True)
        self.feature_idx = dict(zip(feature_list, range(len(feature_list))))
        print(self.feature_idx)
        self.start_idx = np.where(
            self.X_features[:, self.feature_idx["day_depth"]] == 0)[0]
        self.start_idx = np.append(self.start_idx, self.X_features.shape[0])
        uid_set = set(self.User_list)
        self.uid_dict = dict(zip(sorted(list(uid_set)), range(len(uid_set))))

    def _get_dataset_idx(self, train_list, test_list):
        # split dataset to get train and test sets
        logging.info("Splitting dataset ...")
        dataset_idx = [[], []]
        self.user_start = [[], []]
        self.user_end = [[], []]

        def add_data(dataset_type=0, i=0):
            self.user_start[dataset_type].append(len(
                dataset_idx[dataset_type]))
            dataset_idx[dataset_type] += list(
                range(self.start_idx[i], self.start_idx[i + 1]))
            self.user_end[dataset_type].append(len(dataset_idx[dataset_type]))

        TRAIN, TEST = 0, 1

        for i, idx in enumerate(self.start_idx[:-1]):
            if self.User_list[idx] in train_list:
                add_data(dataset_type=TRAIN, i=i)
            else:
                add_data(dataset_type=TEST, i=i)

        self.dataset_idx = dataset_idx
        self.x_train = self.X_features[dataset_idx[0], :]
        self.x_test = self.X_features[dataset_idx[1], :]

    def _standardize(self, bins=5, diff_level=5):
        logging.info("Standardize features ...")

        if self.onehot == True:
            category_len = {}
            category_bins = []
            for feature in self.categorial_features:
                idx = self.feature_idx[feature]
                f_bins = sorted(list(set(self.x_train[:, idx])))
                f_len = len(f_bins)
                if f_len > 30:
                    print("Category feature is too long: %s, %d" %
                          (feature, f_len))
                    continue
                category_len[feature] = f_len  # calculate number of category
                category_bins.append(f_bins)
            new_dim = len(self.continous_features) * (bins - 1) + sum(
                list(category_len.values())) + diff_level * 2 + 2
        else:
            new_dim = self.x_train.shape[1] + diff_level * 2
        self.x_train_encode = np.zeros((self.x_train.shape[0], new_dim))
        self.x_test_encode = np.zeros((self.x_test.shape[0], new_dim))

        encode_id = 0
        self.feature_idx_encode = {}
        self.difficulty_idx = {}
        self.difficulty_bins = {}
        if not self.onehot:
            self.ss, self.mm = StandardScaler(), MinMaxScaler()

        for feature in self.continous_features:
            idx = self.feature_idx[feature]
            if not self.onehot:
                self.x_train_encode[:, encode_id] = self.ss.fit_transform(
                    self.x_train[:, idx].reshape(-1, 1)).reshape(-1)
                self.x_test_encode[:, encode_id] = self.ss.transform(
                    self.x_test[:, idx].reshape(-1, 1)).reshape(-1)
                encode_id += 1
            else:
                this_bins = bins
                train_labels, train_bins = pd.qcut(self.x_train[:, idx],
                                                   q=this_bins,
                                                   labels=False,
                                                   retbins=True,
                                                   duplicates='drop')
                test_labels = pd.cut(self.x_test[:, idx],
                                     bins=train_bins,
                                     labels=False)
                self.x_train_encode[:, encode_id:encode_id + len(train_bins) -
                                                 2] = pd.get_dummies(
                    train_labels).to_numpy()[:, :-1]
                self.x_test_encode[:, encode_id:encode_id + len(train_bins) -
                                                2] = pd.get_dummies(
                    test_labels).to_numpy()[:, :-1]
                encode_id += len(train_bins) - 2

        for i, feature in enumerate(self.categorial_features):
            idx = self.feature_idx[feature]
            if self.onehot == False:
                self.x_train_encode[:, encode_id] = self.mm.fit_transform(
                    self.x_train[:, idx].reshape(-1, 1)).reshape(-1)
                self.x_test_encode[:, encode_id] = self.mm.transform(
                    self.x_test[:, idx].reshape(-1, 1)).reshape(-1)
                encode_id += 1
            else:
                this_bins = category_len[feature]
                train_bins = np.array(category_bins[i]) - 0.01
                train_bins = np.append(train_bins, train_bins[-1] + 1)
                train_labels = pd.cut(self.x_train[:, idx],
                                      bins=train_bins,
                                      labels=False)
                test_labels = pd.cut(self.x_test[:, idx],
                                     bins=train_bins,
                                     labels=False)
                label_set = set(train_labels)
                if len(label_set) != len(set(test_labels)):
                    for l in label_set:
                        if l not in test_labels:
                            for k in range(len(test_labels)):
                                if test_labels[k] in [l - 1, l + 1]:
                                    print("[{}] replace {} with {}".format(
                                        feature, test_labels[k], l))
                                    test_labels[k] = l
                                    break
                self.x_train_encode[:, encode_id:encode_id + len(train_bins) -
                                                 2] = pd.get_dummies(
                    train_labels).to_numpy()[:, :-1]
                self.x_test_encode[:, encode_id:encode_id + len(train_bins) -
                                                2] = pd.get_dummies(
                    test_labels).to_numpy()[:, :-1]
                encode_id += len(train_bins) - 2

        for feature in self.distance_features:
            this_bins = diff_level
            idx = self.feature_idx[feature]
            f = self.x_train[:, idx]
            _, train_bins = pd.qcut(f[np.nonzero(f)],
                                    q=this_bins,
                                    labels=False,
                                    retbins=True,
                                    duplicates='drop')
            train_labels = pd.cut(self.x_train[:, idx],
                                  bins=train_bins,
                                  labels=False)
            test_labels = pd.cut(self.x_test[:, idx],
                                 bins=train_bins,
                                 labels=False)

            def dummy(labels, value):
                encode = pd.get_dummies(labels).to_numpy()
                encode[value == 0, :] = 0
                return encode

            self.x_train_encode[:, encode_id:encode_id + len(train_bins) - 1] = dummy(train_labels,
                                                                                      self.x_train[:, idx])
            self.x_test_encode[:, encode_id:encode_id + len(train_bins) - 1] = dummy(test_labels, self.x_test[:, idx])
            self.difficulty_idx[feature] = encode_id
            self.difficulty_bins[feature] = train_bins
            encode_id += len(train_bins) - 1

        del self.x_train
        del self.x_test
        gc.collect()
        self.x_train_encode = self.x_train_encode[:, :encode_id]
        self.x_test_encode = self.x_test_encode[:, :encode_id]

    def _generate_data(self, X_all, u_all, start_idx, end_idx):
        start_idx = np.array(start_idx)
        end_idx = np.array(end_idx)
        unsort_length = end_idx - start_idx
        idx_sort_origin = np.argsort(unsort_length)

        X_data = np.zeros((len(start_idx), X_all.shape[1], self.max_length))
        uid_list = np.zeros((len(start_idx)))
        y_length = np.zeros((len(start_idx)))
        y_censor = np.zeros((len(start_idx)))

        for i in np.arange(len(start_idx)):
            j = idx_sort_origin[i]
            s_start, s_end = start_idx[j], end_idx[j]
            s_length = s_end - s_start
            uid_list[i] = self.uid_dict[u_all[s_start]]
            if s_length <= self.max_length:
                X_data[i, :, :(s_end - s_start)] = X_all[s_start:s_end, :].T
                y_length[i] = s_length
                y_censor[i] = 1
            else:
                X_data[i, :, :] = X_all[s_start:s_start + self.max_length, :].T
                y_length[i] = self.max_length

        idx_sort = np.argsort(y_length, kind='stable')
        if (idx_sort == np.arange(0, len(idx_sort))).all():
            logging.info("Sort correct!")
            return X_data, uid_list, (y_length, y_censor)
        else:
            logging.info("Sort incorrect!")
            R = X_data[idx_sort, :, :], uid_list[idx_sort], (
                y_length[idx_sort], y_censor[idx_sort])
            return R


def analyse_beta(
        this_beta,
        Scale_list,
        name="lose",
):
    diff_level = len(Scale_list) - 1
    labels = [
        "d in [%.2f,%.2f)" % (Scale_list[i], Scale_list[i + 1])
        for i in range(diff_level)
    ]
    labels[0] = "d < %.2f" % (Scale_list[1])
    labels[-1] = "d >= %.2f" % (Scale_list[-2])
    logging.info("Range: {}".format(labels))

    Length = int(this_beta.shape[0] * 0.9)
    Start, End = [0, 0, int(Length / 3),
                  int(Length / 3) * 2
                  ], [Length,
                      int(Length / 3),
                      int(Length / 3) * 2, Length]
    Range_label = [
        "All",
        "First %d interactions" % (Length / 3),
        "Middel %d interactions" % (Length / 3),
        "Last %d interactions" % (Length / 3)
    ]
    for s, e, label in zip(Start, End, Range_label):
        median_y = np.median(this_beta[s:e, :], axis=0)
        median_y = [round(y, 3) for y in median_y]
        logging.info("Median of {} beta: {} ({})".format(
            label, median_y, name))
    mean = np.mean(this_beta[:Length, :], axis=0)
    mean = [round(x, 3) for x in mean]
    logging.info("Mean of beta({}): {}".format(name, mean))


def parse_train_args(parser):
    parser.add_argument('--lr',
                        type=float,
                        default=0.05,
                        help='Learning rate.')
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument('--batch_size',
                        type=float,
                        default=2048,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=256, help='Max epochs')
    parser.add_argument(
        '--earlystop_patience',
        type=int,
        default=10,
        help=
        'Tolerance epochs of early stopping, set to -1 if not use early stopping.'
    )
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument('--cross_validation',
                        type=int,
                        default=0,
                        help='Whether to use cross validation.')
    parser.add_argument('--distance_level',
                        type=int,
                        default=5,
                        help='One-hot embedding dimension for PPD.')
    parser.add_argument('--max_window_length',
                        type=int,
                        default=30,
                        help='Maximum length of data to clamp.')
    parser.add_argument(
        '--one_hot',
        type=int,
        default=0,
        help="Whether use one-hot encoding for other features.")
    return parser


def parse_global_args(parser):
    parser.add_argument('--data_path',
                        type=str,
                        default='../../data/',
                        help='Input data path.')
    parser.add_argument('--fold_define',
                        type=str,
                        default='../../data/',
                        help='Uids for each fold.')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        help='Device to train the model.')
    parser.add_argument('--gpu',
                        type=str,
                        default='0',
                        help='Set CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--verbose',
                        type=int,
                        default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50')
    parser.add_argument('--train_verbose',
                        type=int,
                        default=1,
                        help='Verbose while training.')
    parser.add_argument('--log_file',
                        type=str,
                        default='',
                        help='Logging file path')
    parser.add_argument('--random_seed',
                        type=int,
                        default=2021,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--load',
                        type=int,
                        default=0,
                        help='Whether load model and continue to train')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of workers to load data.')
    parser.add_argument('--model_name', type=str, default="Cox_model_test")
    parser.add_argument("--fix_seed", type=int, default=0)
    parser.add_argument("--predict", type=int, default=0)
    return parser


def training(args, fold, random_seed, data_loader):
    # Random seed
    if args.fix_seed:
        np.random.seed(2020)
        torch.manual_seed(2020)
        torch.cuda.manual_seed(2020)
    else:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

    # Read data and generate training, val, test set
    # Define columns
    data_loader.construct_input_data(fold)

    x_train, u_train, y_train = data_loader.load_data("train")
    print(x_train.shape)
    print(u_train.shape)
    print([y.shape for y in y_train])
    logging.info("Training data size: {}".format(x_train.shape))
    x_test, u_test, y_test = data_loader.load_data("test")
    logging.info("Test data size:{}".format(x_test.shape))

    logging.info("Defining Cox Hazard model.")
    Hazard_model = Hazard_net(feature_num=x_train.shape[1],
                              length=x_train.shape[2])

    if args.device == "cuda" and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    Cox_model = None
    if args.optimizer == "Adam":
        Cox_model = CoxTime(
            Hazard_model,
            tt.optim.Adam(weight_decay=args.weight_decay),
            device=device,
        )
    elif args.optimizer == "AdamWR":
        Cox_model = CoxTime(
            Hazard_model,
            tt.optim.AdamWR(decoupled_weight_decay=args.weight_decay),
            device=device,
        )
    # training
    if args.load:
        logging.info("Loading exist model...")
        Cox_model.load_net(
            os.path.join("./outputs/fold-%d/" % (fold), args.model_name,
                         args.model_name + ".pt"))
    if args.predict > 0:
        logging.info("Loading exist model...")
        base = './Checkpoints/fold-%d/' % (fold)
        print(base + args.model_name + '/' +
              args.model_name + '.pt')
        model = Cox_model.load_net(
            base + args.model_name + '/' +
            args.model_name + '.pt')
        model.predict_surv_df(x_test)
    else:
        utils.check_dir(
            os.path.join("./outputs/fold-%d/" % (fold), args.model_name,
                         args.model_name + '.pt'))
        logging.info("Training Cox hazard model...")
        if args.earlystop_patience >= 0:
            callbacks = [
                tt.callbacks.EarlyStopping(
                    patience=args.earlystop_patience,
                    file_path='outputs/fold-%d/%s/earlystop.pt' %
                              (fold, args.model_name))
            ]
        else:
            callbacks = []

        lrfinder = Cox_model.lr_finder(x_train, y_train, int(args.batch_size), tolerance=2)
        try:
            lrfinder.to_pandas().to_csv('./outputs/lrfinder' + str(fold) + '.csv')
            print('best lr: ' + str(lrfinder.get_best_lr()))
            lrfinder.plot()
        except Exception:
            pass
        Cox_model.optimizer.set_lr(args.lr)
        log = Cox_model.fit(x_train,
                            y_train,
                            int(args.batch_size),
                            args.epochs,
                            callbacks,
                            args.train_verbose,
                            val_data=tt.tuplefy(x_test, y_test),
                            num_workers=args.num_workers)
        # log.plot()
        # plt.savefig('../logs/log_' + str(fold) + '.png')
        logging.info("Training Done!")
        logging.info("Min val loss: {:<.4f}".format(
            log.to_pandas().val_loss.min()))
        beta = Cox_model.net.beta.cpu().detach().numpy()

        # save models
        logging.info("Saving best model...")
        Cox_model.save_net(
            os.path.join("outputs/fold-%d" % (fold), args.model_name,
                         args.model_name + ".pt"))

        # calculating figures
        logging.info("Analysing difficulty hazrd parameters...")
        d_idx = data_loader.difficulty_idx
        diff_level = args.distance_level
        diff_beta = beta[d_idx["PPD"]:d_idx["PPD"] + diff_level, :].T
        analyse_beta(
            diff_beta,
            data_loader.difficulty_bins["PPD"],
            "difficulty",
        )
        Scale_list = data_loader.difficulty_bins["PPD"][1:-1]

        # test
        logging.info("Calculating performance of model on test set...")
        logging.info("Calculating baseline hazards...")
        baseline_hazards = Cox_model.compute_baseline_hazards()
        logging.info("Predicting training set...")
        surv = Cox_model.predict_surv(x_train, num_workers=args.num_workers)
        # surv_df_1 = Cox_model.predict_surv_df(x_test, num_workers=args.num_workers)
        # surv_df_1.to_csv('./outputs/surv_df_1_' + str(fold) + '.csv')

        surv_df = pd.DataFrame(surv.T)
        # surv_df.to_csv('./outputs/surv_df_' + str(fold) + '.csv')

        ev = EvalSurv(surv_df, y_train[0], y_train[1], censor_surv='km')
        c_index = ev.concordance_td("antolini")

        ev.brier_score(np.linspace(y_train[0].min(), y_train[1].max(), 100)).to_csv(
            './outputs/brier_score_train' + str(fold) + '.csv')

        logging.info("training set C INDEX: {:.3f}".format(c_index))
        logging.info("Predicting test set...")
        surv = Cox_model.predict_surv(x_test, num_workers=args.num_workers)
        surv_df = pd.DataFrame(surv.T)
        ev = EvalSurv(surv_df, y_test[0], y_test[1], censor_surv='km')

        ev.brier_score(np.linspace(y_test[0].min(), y_test[1].max(), 100)).to_csv(
            './outputs/brier_score_test' + str(fold) + '.csv')

        c_index = ev.concordance_td("antolini")
        logging.info("test set C INDEX: {:.3f}".format(c_index))
        time_grid = np.linspace(y_test[0].min(), y_test[0].max(),
                                x_train.shape[2] + 1)
        ibs = ev.brier_score(time_grid).mean()
        logging.info("test set Brier score: {:.3f}".format(ibs))

        np.save(
            os.path.join("outputs/fold-%d" % (fold), args.model_name, "beta"),
            beta)
        np.save(
            os.path.join("outputs/fold-%d" % (fold), args.model_name,
                         "diff_beta"), diff_beta)
        np.save(
            os.path.join("outputs/fold-%d" % (fold), args.model_name,
                         "diff_scale"), Scale_list)
        return c_index, ibs, log, diff_beta, Scale_list


def main(args):
    logging.info('-' * 45 + ' BEGIN: ' + utils.get_time() + ' ' + '-' * 45)
    exclude = [
        'check_epoch', 'log_file', 'model_path', 'path', 'pin_memory',
        'regenerate', 'sep', 'train', 'verbose'
    ]
    logging.info(utils.format_arg_str(args, exclude_lst=exclude))

    # GPU
    if args.device == 'cuda' and torch.cuda.is_available():
        _device_count = min(int(args.gpu), torch.cuda.device_count())
        os.environ["CUDA_VISIBLE_DEVICES"] = str(_device_count)
        logging.info("# cuda devices: {}".format(_device_count))

    r_list = [[], [], []]
    logging.info("Loading data...")
    continous_features = [
        "level_num", "play_num", "duration", "item_all", "session_num",
        "last_session_play", "session_length", "last_session_level",
        "last_session_item", "last_session_duration", "last5_duration",
        "last5_passrate", "last5_item", "day_depth", "gold_amount",
        "coin_amount"
    ]
    categorial_features = [
        "weekday", "last_session_end_hour", "last_win", "remain_energy"
    ]
    distance_features = ["PPD"]

    data_loader = Dataset_loader(
        data_path=os.path.join(args.data_path, 'D-Cox-Time'),
        fold_define=os.path.join(args.fold_define, 'dataset_split'),
        max_length=args.max_window_length,
        continous_features=continous_features,
        categorial_features=categorial_features,
        distance_features=distance_features,
        onehot=args.one_hot,
        distance_level=args.distance_level,
    )

    for k in range(max(args.cross_validation, 1)):
        c_index, ibs, log, beta_diff, diff_range = training(
            args,
            fold=k + 1,
            random_seed=args.random_seed,
            data_loader=data_loader)
        r_list[0].append(round(c_index, 3))
        r_list[1].append(round(ibs, 3))
        r_list[2].append(log.to_pandas().val_loss.min())
    logging.info("Cross validation results:")
    logging.info("C INDEX: {}".format(r_list[0]))
    logging.info("IBS: {}".format(r_list[1]))
    logging.info("Val Loss: {}".format(r_list[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    args, extras = parser.parse_known_args()

    log_args = [utils.get_date(), str(args.random_seed)]
    log_file_name = '__'.join(log_args)
    if args.log_file == '':
        args.log_file = 'logs/{}/{}.txt'.format(args.model_name, log_file_name)

    utils.check_dir(args.log_file)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    Model = main(args)
