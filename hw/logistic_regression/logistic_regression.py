<<<<<<< HEAD
import numpy as np
from sklearn import datasets
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def initialize_weights(self, n_features, initial_par):
    # 初始化参数
    # 参数范围[-1/sqrt(N), 1/sqrt(N)]
    limit = np.sqrt(1 / n_features)
    w = np.random.uniform(-limit, limit, (n_features, 1))
    b = 0
    self.w = np.insert(w, 0, b, axis=0)


class LogisticRegression():
    """
        Parameters:
        -----------
        n_iterations: int
            梯度下降的轮数
        learning_rate: float
            梯度下降学习率
    """
    def __init__(self, learning_rate=.1, n_iterations=4000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def initialize_weights(self, n_features):
        # 初始化参数
        # 参数范围[-1/sqrt(N), 1/sqrt(N)]
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)

    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.initialize_weights(n_features)
        # 为X增加一列特征x1，x1 = 0
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))

        # 梯度训练n_iterations轮
        for i in range(self.n_iterations):
            h_x = X.dot(self.w)
            y_pred = sigmoid(h_x)
            w_grad = X.T.dot(y_pred - y)
            self.w = self.w - self.learning_rate * w_grad

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        h_x = X.dot(self.w)
        y_pred = np.round(sigmoid(h_x))
        return y_pred.astype(int)


def main():
    # Load dataset
    data = datasets.load_iris()
    X = normalize(data.data[data.target != 0])
    y = data.target[data.target != 0]
    y[y == 1] = 0
    y[y == 2] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, seed=1)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred = np.reshape(y_pred, y_test.shape)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Reduce dimension to two using PCA and plot the results
    Plot().plot_in_2d(X_test, y_pred, title="Logistic Regression", accuracy=accuracy)


if __name__ == "__main__":
    main()
=======
from numpy import *

filename = 'logistic_test'


def load_data_set():
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1]), float(line_arr[2]),])
        label_mat.append(int(line_arr[3]))
    return data_mat, label_mat


def sigmoid(in_x):
    return 1.0 / (1 + exp(-in_x))


def grad_ascent(data_mat, label_mat, weights):
    data_matrix = mat(data_mat)
    class_label = mat(label_mat).transpose()
    m, n = shape(data_matrix)
    alpha = 0.01
    max_cycle = 500000
    weights = ones((n, 1))
    error = 0
    for k in range(max_cycle):
        h = sigmoid(data_matrix * weights)
        error = (class_label - h)
        weights = weights + alpha * data_matrix.transpose() * error
    print(error)
    return weights


def plot_base_fit(weights):  # 画出最终分类的图
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_data_set()
    data_arr = array(data_mat)
    n = shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def main():
    data_mat, label_mat = load_data_set()
    weights = grad_ascent(data_mat, label_mat, [-2.0, 1.0, 1.0]).getA()
    print(weights)
    # plot_base_fit(weights)


if __name__ == '__main__':
    main()
>>>>>>> origin/main
