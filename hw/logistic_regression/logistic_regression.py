import mpmath
import numpy as np
from numpy import *
filename = 'logistic_test'


def load_data_set():
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, double(line_arr[0]), double(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(in_x):
    return np.where((1.0 / (1 + exp(-in_x))) >= 0.5, 1, 0)
    # return 1.0 / (1 + exp(-in_x))


def grad_ascent(data_mat, label_mat, weights, max_cycle, alpha):
    data_matrix = mat(data_mat)
    class_label = mat(label_mat).transpose()
    m, n = shape(data_matrix)
    weight = mat(weights).transpose()

    # weights = ones((n, 1))
    error = 0
    for k in range(max_cycle):
        h = sigmoid(data_matrix * weight)
        error = (class_label - h)
        weight = weight + alpha * (1/m) * (error.transpose() * data_matrix).transpose()
        # print(weight)
    return weight,error


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
    # data_mat = np.random.randint(0, 10, (100, 15))
    # data_mat[:, 0] = 0
    # label_mat = np.random.randint(0, 100, (100, 1))
    # for i in range(5):
    #     data_mat[random.randint(0, 100)][0] = 1
    data_mat, label_mat = load_data_set()
    loops = 1000000
    print('Number of loops ' + str(loops))
    weight,error = grad_ascent(data_mat, label_mat, [1, -1.5, -1.5], loops, 0.1)
    # weight, error = grad_ascent(data_mat, label_mat, [1, 1, 1, 1], loops, 0.1)
    # plot_base_fit(weights)
    np.set_printoptions(threshold=np.inf)
    print(weight.getA().transpose())
    print(error)
    # plot_base_fit(weights)


if __name__ == '__main__':
    main()
