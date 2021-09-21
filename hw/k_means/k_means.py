import sys
from math import sqrt

import matplotlib.pyplot as plt
from numpy import *


def find_mean(points):
    x_axis = [p.x for p in points]
    y_axis = [p.y for p in points]
    length = len(points)
    return Point(sum(x_axis) / length, sum(y_axis) / length)


def cluster_process(points, k, centroids, epsilon):
    assert k == len(centroids)
    change_list = [100] * k
    clusters = {}
    while max(change_list) >= epsilon:
        for i in range(k):
            clusters[i] = list()
        for i in range(len(points)):
            distance = sys.maxsize
            pos_j = -1
            for j in range(k):
                if (centroids[j]).norm2_distance(points[i]) < distance:
                    distance, pos_j = centroids[j].norm2_distance(points[i]), j
            clusters[pos_j].append(points[i])
        for cluster_key in clusters:
            new_mean_p = find_mean(clusters[cluster_key])
            change_list[cluster_key] = centroids[cluster_key].norm1_distance(new_mean_p)
            centroids[cluster_key] = new_mean_p
    return clusters, centroids


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def norm2_distance(self, p):
        return sqrt((p.x - self.x) ** 2 + (p.y - self.y) ** 2)

    def norm1_distance(self, p):
        return abs(p.x - self.x) + abs(p.y - self.y)

    def __str__(self):
        return '(' + str(self.x) + ',' + str(self.y) + ')'


def generate_points(inputs):
    res = []
    for i in range(len(inputs)):
        res.append(Point(inputs[i][0], inputs[i][1]))
    return res


if __name__ == '__main__':
    data = generate_points([(55, 50), (43, 50), (55, 52), (43, 54), (58, 53), (41, 47), (50, 41), (50, 70)])
    initial_means = generate_points([(50, 41), (50, 70), (43, 50), (55, 50)])
    eps = 1
    final_clusters, centrals = cluster_process(data, len(initial_means), initial_means, eps)

    for clu in final_clusters:
        print(clu)
        for p in (final_clusters[clu]):
            print(p)
    print()
    for central in centrals:
        print(central)

    color = ["pink", "yellow", "red", "yellow", "black", "green", "orange"]
    for i in range(len(final_clusters)):
        plt.scatter([p.x for p in final_clusters[i]], [p.y for p in final_clusters[i]], c=color[i])
    for central in centrals:
        plt.scatter([central.x], [central.y], c=color[len(final_clusters) + 2])
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
