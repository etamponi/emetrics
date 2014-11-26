import unittest
import numpy
from emetrics.coefficients import UncertaintyCoefficient

__author__ = 'Emanuele Tamponi'


class UncertaintyCoefficientTest(unittest.TestCase):

    def test_uncertainty_works(self):
        for i in range(10):
            X = numpy.random.rand(1000, 10)
            y = numpy.random.choice(["a", "b", "c"], size=1000)
            value = UncertaintyCoefficient(neighbors=1)(X, y)
            self.assertTrue(0 < value < 1, msg="Expected [0, 1], got {}".format(value))
