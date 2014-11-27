import math
import numpy
from sklearn.neighbors.unsupervised import NearestNeighbors

__author__ = 'Emanuele Tamponi'


class UncertaintyCoefficient(object):

    def __init__(self, noise_level=0):
        self.noise_level = noise_level

    def __call__(self, inputs, labels):
        classes = numpy.unique(labels)
        inputs = inputs.copy()
        inputs += self.noise_level * numpy.random.randn(*inputs.shape)

        input_entropy = self._entropy(inputs)
        conditional_input_entropy = 0.0
        class_entropy = 0.0
        for label in classes:
            class_inputs = inputs[labels == label]
            class_prob = float(len(class_inputs)) / len(inputs)
            conditional_input_entropy += class_prob * self._entropy(class_inputs)
            class_entropy -= class_prob * math.log(class_prob)
        information = input_entropy - conditional_input_entropy

        return information / min(input_entropy, class_entropy)

    @staticmethod
    def _entropy(data):
        if len(data) == 1:
            return 0.0
        euler_const = 0.5772156649
        nn = NearestNeighbors(n_neighbors=2, metric="chebyshev").fit(data)
        entropy = 0.0
        for x in data:
            nearest_neighbor_distance = nn.kneighbors(x)[0][0].max()
            entropy += math.log(len(data) * nearest_neighbor_distance)
        entropy = entropy / len(data) + math.log(2) + euler_const
        return entropy
