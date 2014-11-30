import unittest
import math

import numpy

from emetrics.preparers.noise_injector import NoiseInjector


__author__ = 'Emanuele Tamponi'


class NoiseInjectorTest(unittest.TestCase):

    def test_injection(self):
        X = numpy.random.rand(15, 3)
        y = numpy.random.choice(["a", "b", "c"], size=15)
        injector = NoiseInjector()
        X_noised = injector.apply(X, y)
        self.assertFalse(numpy.all(X == X_noised))
        numpy.testing.assert_array_equal(X.shape, X_noised.shape)
        numpy.testing.assert_array_almost_equal(X, X_noised, decimal=int(-math.log10(injector.stddev) - 1))

    def test_injection_custom_level(self):
        X = numpy.random.rand(15, 3)
        y = numpy.random.choice(["a", "b", "c"], size=15)
        injector = NoiseInjector(stddev=1e-10)
        X_noised = injector.apply(X, y)
        self.assertFalse(numpy.all(X == X_noised))
        numpy.testing.assert_array_equal(X.shape, X_noised.shape)
        numpy.testing.assert_array_almost_equal(X, X_noised, decimal=int(-math.log10(injector.stddev) - 1))
