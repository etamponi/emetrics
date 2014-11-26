import unittest
import numpy

from sklearn.datasets.samples_generator import make_classification

from emetrics.coefficients import UncertaintyCoefficient


__author__ = 'Emanuele Tamponi'


class UncertaintyCoefficientTest(unittest.TestCase):

    def test_uncertainty_works(self):
        for i in range(10):
            X = numpy.random.rand(100, 20)
            y = numpy.random.choice(["a", "b"], size=100)
            value = UncertaintyCoefficient()(X, y)
            self.assertTrue(0 < value < 1, msg="Expected [0, 1], got {}".format(value))
