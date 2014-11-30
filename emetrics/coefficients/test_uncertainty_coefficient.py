import unittest

import numpy

from emetrics.coefficients import UncertaintyCoefficient


__author__ = 'Emanuele Tamponi'


class UncertaintyCoefficientTest(unittest.TestCase):

    def test_uncertainty_works(self):
        for i in range(10):
            X = numpy.random.rand(100, 3)
            y = numpy.random.choice(["a", "b"], size=100)
            value = UncertaintyCoefficient()(X, y)
            self.assertTrue(0 < value < 1, msg="Expected [0, 1], got {}".format(value))

    def test_uncertainty_one_element_per_class(self):
        X = numpy.random.rand(100, 3)
        y = numpy.zeros(100)
        y[0] = 1
        try:
            UncertaintyCoefficient()(X, y)
        except ValueError:
            self.fail("Uncertainty failed because class with too few elements")
