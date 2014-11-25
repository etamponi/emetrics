import numpy
from sklearn.linear_model.base import LinearRegression

__author__ = 'Emanuele Tamponi'


class DeterminationCoefficient(object):

    def __init__(self):
        pass

    def __call__(self, inputs, labels):
        _, labels = numpy.unique(labels, return_inverse=True)
        return LinearRegression().fit(inputs, labels).score(inputs, labels)
