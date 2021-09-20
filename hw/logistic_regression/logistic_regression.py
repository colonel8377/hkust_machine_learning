from numpy import *

filename = 'logistic_test'


def load_data_set():
    data_mat = []
    label_mat = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(in_x):
    return 1.0 / (1 + exp(-in_x))


def grad_ascent(data_mat, label_mat, weights):
    data_matrix = mat(data_mat)
    class_label = mat(label_mat).transpose()
    m, n = shape(data_matrix)
    alpha = 0.1
    max_cycle = 500
    weights = mat(weights).transpose()
    for k in range(max_cycle):
        h = sigmoid(data_matrix * weights)
        error = (class_label - h)
        weights = weights + alpha * (1/m) * data_matrix.transpose() * error
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
    plot_base_fit(weights)


if __name__ == '__main__':
    main()
