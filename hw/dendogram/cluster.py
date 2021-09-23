from __future__ import division
from __future__ import absolute_import

import numpy
import copy
import argparse

from operator import itemgetter
from collections import defaultdict
from itertools import combinations, product

import numpy as np

from api import AbstractClusterer
from dendrogram import Dendrogram
from linkage import linkage_fn
from distance import *

from sklearn.metrics.pairwise import pairwise_distances


class CooccurrenceMatrix(numpy.ndarray):
    """ Represents a co-occurrence matrix. """

    def __new__(cls, data, dtype=None):
        if not isinstance(data, CooccurrenceMatrix):
            data, rownames, colnames = CooccurrenceMatrix.convert(data)
        else:
            rownames, colnames = data.rownames, data.colnames
        obj = numpy.asarray(data).view(cls)
        obj.rownames = rownames
        obj.colnames = colnames
        return obj

    def __array_finialize__(self, obj):
        if obj is None: return
        self.rownames = getattr(obj, 'rownames', None)
        self.colnames = getattr(obj, 'colnames', None)

    def row(self, row):
        return self[self.rownames.get(row)]

    def col(self, col):
        return self[:, self.colnames.get(col)]

    def cell(self, row, col):
        return self[self.rownames.get(row), self.colnames.get(col)]

    @classmethod
    def convert(cls, data):
        matrix = numpy.zeros((len(set(k for k, v in data)),
                              len(set(v for k, v in data))))
        colnames, rownames = {}, {}
        for k, v in sorted(data):
            if k not in rownames:
                rownames[k] = len(rownames)
            if v not in colnames:
                colnames[v] = len(colnames)
            matrix[rownames[k], colnames[v]] += 1
        # rownames = [k for k,v in sorted(rownames.items(), key=itemgetter(1))]
        # colnames = [k for k,v in sorted(colnames.items(), key=itemgetter(1))]
        return matrix, rownames, colnames

    def tfidf(self):
        """
        Returns a matrix in which for all entries in the co-occurence matrix
        the 'term frequency-inverse document frequency' is calculated.
        """
        matrix = numpy.zeros(self.shape)
        # the number of words in a document
        words_per_doc = numpy.asarray(self.sum(axis=1), dtype=float)
        # the number of documents in which a word is attested.
        word_frequencies = numpy.asarray(numpy.sum(self > 0, axis=0), dtype=float)
        # calculate the term frequencies
        for i in range(self.shape[0]):
            tf = self[i] / words_per_doc[i]  # array of tf's
            matrix[i] = tf * (numpy.log(self.shape[0] / word_frequencies))
        return matrix


class DistanceMatrix(numpy.ndarray):
    """
    Simple wrapper around numpy.ndarray, to provide some custom
    Distance Matrix functionality like plotting the distance matrix
    with matplotlib.
    """

    def __new__(cls, data, dist_metric=euclidean_distance, lower=True):
        if (not isinstance(data, (numpy.ndarray, DistanceMatrix))
                or len(data) != len(data[0])
                or not max(numpy.diag(data)) == 0):
            data = DistanceMatrix.convert_to_distmatrix(data, dist_metric, lower=lower)
        obj = numpy.asarray(data).view(cls)
        obj.distance_metric = dist_metric
        return obj

    def __array_finialize__(self, obj):
        if obj is None: return
        self.distance_metric = getattr(obj, 'distance_metric', None)

    def row(self, row):
        return self[self.rownames.get(row)]

    def col(self, col):
        return self[:, self.colnames.get(col)]

    def cell(self, row, col):
        return self[self.rownames.get(row), self.colnames.get(col)]

    def rows(self):
        return [k for k, v in sorted(self.rownames.items(), key=itemgetter(1))]

    @classmethod
    def convert_to_distmatrix(cls, data, distance, lower=True):
        matrix = numpy.zeros((len(data), len(data)))
        for i, j in combinations(range(len(data)), 2):
            matrix[i][j] = distance(data[i], data[j])
            if lower == True:
                matrix[j][i] = matrix[i][j]
        # add a nan-diagonal, useful for further computations.
        numpy.fill_diagonal(matrix, numpy.nan)
        return matrix

    def diag_is_zero(self):
        """Check if the diagonal contains only distances of 0."""
        return max(numpy.diag(self)) == 0

    def remove(self, idx):
        """
        Delete a row and column with index IDX.
        WARNING this function is NOT destructive!
        """
        indices = range(len(self))
        indices.remove(idx)
        return self.take(indices, axis=0).take(indices, axis=1)

    def draw(self, save=False, format="pdf"):
        """Make a nice colorful plot of the distance matrix."""
        try:
            import pylab
        except ImportError:
            raise ImportError("Install pylab.")
        fig = pylab.figure()
        axmatrix = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        im = axmatrix.matshow(self, aspect='auto', origin='upper',
                              cmap=pylab.cm.YlGnBu)
        axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.8])
        pylab.colorbar(im, cax=axcolor)
        fig.show()
        if save:
            fig.savefig('distance-matrix.%s' % (format,))

    def summary(self):
        """Return a small summary of the matrix."""
        print('DistanceMatrix (n=%s)' % len(self))
        print('Distance metric = %s' % self.distance_metric.__name__)
        print(self)


class Clusterer(AbstractClusterer):
    """
    The Hierarchical Agglomerative Clusterer starts with each of the N vectors
    as singleton clusters. It then iteratively merges pairs of clusters which
    have the smallest distance according to function LINKAGE. This continues
    until there is only one cluster.
    """

    def __init__(self, data, linkage='ward', num_clusters=1):
        self._num_clusters = num_clusters
        vector_ids = [[i] for i in range(len(data))]
        self._dendrogram = Dendrogram(vector_ids)
        numpy.fill_diagonal(data, numpy.inf)
        self._dist_matrix = data
        self.linkage = linkage_fn(linkage)

    def smallest_distance(self, clusters):
        """
        Return the smallest distance in the distance matrix.
        The smallest distance depends on the possible connections in
        the distance matrix.

        @param clusters: an object of the class L{DistanceMatrix} holding the
            clusters at a specific state in the clustering procedure.
        @type clusters: L{DistanceMatrix}
        @return: a tuple containing the smallest distance and the indexes of
            the clusters yielding the smallest distance.
        """
        i, j = numpy.unravel_index(numpy.argmin(clusters), clusters.shape)
        return clusters[i, j], i, j

    def cluster(self, verbose=0, sum_ess=False):
        clusters = copy.copy(self._dist_matrix)
        # clusters = self._dist_matrix
        summed_ess = 0.0

        while len(clusters) > max(self._num_clusters, 1):
            if verbose >= 1:
                print('k=%s' % len(clusters))
                if verbose == 2:
                    print(clusters)

            best, i, j = self.smallest_distance(clusters)
            print(str(best))

            # In Ward (1963) ess is summed at each iteration
            # in R's hclust and Python's hcluster and some text books it is not.
            # Here it is optional...
            if sum_ess:
                summed_ess += best
            else:
                summed_ess = best
            clusters = self.update_distmatrix(i, j, clusters)
            self._dendrogram.merge(i, j)
            self._dendrogram[i].distance = summed_ess
            indices = numpy.arange(clusters.shape[0])
            indices = indices[indices != j]
            clusters = clusters.take(indices, axis=0).take(indices, axis=1)
            print(clusters)

    def update_distmatrix(self, i, j, clusters):
        """
        Update the distance matrix using the specified linkage method so that
        it represents the correct distances to the newly formed cluster.
        """
        return self.linkage(clusters, i, j, self._dendrogram)

    @property
    def dendrogram(self):
        """Return the dendrogram object."""
        return self._dendrogram

    def num_clusters(self):
        return self._num_clusters

    def __repr__(self):
        return """<Hierarchical Agglomerative Clusterer(linkage method: %r,
                  n=%d clusters>""" % (self.linkage.__name__, self._num_clusters)


class VNClusterer(Clusterer):
    """
    Variability Neighbor Clustering Class. A subclass of the regular Clusterer
    where the order of clustering can be predetermined. In the normal clustering
    procedure, all clusters can be clustered with all other clusters. In this
    class, the clusters that are allowed to be clustered follow a specific order.
    """

    def __init__(self, data, linkage='ward', num_clusters=1):
        Clusterer.__init__(self, data, linkage, num_clusters=num_clusters)

    def iterate_clusters(self, clusters):
        for i in range(1, len(clusters)):
            yield i - 1, i

    def smallest_distance(self, clusters):
        best = None
        for i, j in self.iterate_clusters(clusters):
            if best is None or clusters[i][j] <= best[0]:
                best = (clusters[i][j], i, j)
        print(best)
        return best

    def cluster(self, verbose=False):
        # we must sum the error sum of squares in order not to obtain
        # singleton clustering.
        Clusterer.cluster(self, verbose=verbose, sum_ess=True)


class EuclideanNeighborClusterer(VNClusterer):

    def iterate_clusters(self, x, y):
        n_features, n_samples = x, y
        offset = (0, -1, 1)
        indices = ((i, j) for i in range(n_features) for j in range(n_samples))
        for i, j in indices:
            all_neigh = ((i + x, j + y) for x in offset for y in offset)
            valid = ((i * n_features + j) for i, j in all_neigh if (0 <= i < n_features) and (0 <= j < n_samples))
            target = valid.next()
            for neighbor in list(valid):
                yield target, neighbor


def demo():
    """
    Demo to show some basic functionality.
    """
    # declare dummy input vector with two dimensions:
    # vectors = numpy.array([[0, 0], [5, 6], [1, 1], [3, 2], [4, 0], [2, 2], [8, 9], [8, 11]])

    # compute the distance matrix on the basis of the vectors via sklearn:
    # dist_matrix = pairwise_distances(vectors, metric='cityblock')
    dist_matrix = np.array([[0.0, 11.0, 5.0, 12.0, 7.0, 13.0, 9.0, 11.0],
                             [11.0, 0.0, 13.0, 2.0, 17.0, 4.0, 15.0, 20.0],
                             [5.0, 13.0, 0.0, 14.0, 1.0, 15.0, 12.0, 12.0],
                             [12.0, 2.0, 14.0, 0.0, 18.0, 5.0, 16.0, 21.0],
                             [7.0, 17.0, 1.0, 18.0, 0.0, 20.0, 15.0, 17.0],
                             [13.0, 4.0, 15.0, 5.0, 20.0, 0.0, 19.0, 22.0],
                             [9.0, 15.0, 12.0, 16.0, 15.0, 19.0, 0.0, 30.0],
                             [11.0, 20.0, 12.0, 21.0, 17.0, 22.0, 30.0, 0.0]])
    print(dist_matrix)
    # plot the distance matrix:
    # dist_matrix.draw() this doesn't work anymore

    # initialize a temporal VNC clusterer, here with the Ward linkage method:
    clusterer = Clusterer(dist_matrix, linkage='average')
    # could also be a plain Clusterer()

    # start the clustering procedure:
    clusterer.cluster(verbose=0)

    labels = ['n' + str(i + 1) for i in range(dist_matrix.shape[0])]
    # plot the result as a dendrogram
    clusterer.dendrogram.draw(save=True, labels=labels, title="VNC Analysis (Group Average's Linkage)")


if __name__ == '__main__':
    demo()
