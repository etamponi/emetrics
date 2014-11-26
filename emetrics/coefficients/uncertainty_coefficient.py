import math
import numpy
from scipy.spatial.distance import pdist
from sklearn.neighbors.unsupervised import NearestNeighbors

__author__ = 'Emanuele Tamponi'


class UncertaintyCoefficient(object):

    def __init__(self, neighbors=5):
        self.neighbors = neighbors

    def __call__(self, inputs, labels):
        min_dist = 1  # pdist(inputs, metric="chebyshev").min()
        nn = NearestNeighbors(n_neighbors=self.neighbors+1, metric="chebyshev").fit(inputs)
        _, labels = numpy.unique(labels, return_inverse=True)
        n_classes = len(numpy.unique(labels))
        label_inputs = [inputs[labels == label] for label in range(n_classes)]
        label_nn = [NearestNeighbors(n_neighbors=self.neighbors+1, metric="chebyshev").fit(l_i) for l_i in label_inputs]

        N = len(inputs)
        input_entropy = self._entropy(inputs, nn, min_dist)
        conditional_entropy = 0
        label_entropy = 0
        for label in range(n_classes):
            N_label = len(label_inputs[label])
            P_label = float(N_label) / N
            conditional_entropy += P_label * self._entropy(label_inputs[label], label_nn[label], min_dist)
            label_entropy -= P_label * math.log(P_label)
        information = input_entropy - conditional_entropy

        return information / label_entropy

    def _entropy(self, data, nn, min_dist):
        N = len(data)
        value = 0
        for x in data:
            max_distance = nn.kneighbors(x)[0][0][-1] / min_dist
            value += math.log(max_distance)
        value /= N
        value += self._digamma(N) - self._digamma(self.neighbors)
        return value

    @staticmethod
    def _digamma(n):
        if n == 1:
            return 0.0
        else:
            value = 0.0
            for i in range(1, n):
                value += 1.0 / i
            return value
