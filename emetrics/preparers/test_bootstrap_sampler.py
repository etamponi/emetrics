import unittest
import numpy
from emetrics.preparers.bootstrap_sampler import BootstrapSampler

__author__ = 'Emanuele Tamponi'


class BootstrapSamplerTest(unittest.TestCase):

    def test_bootstrap_sampler_works(self):
        X = numpy.random.rand(15, 3)
        y = numpy.random.choice(["a", "b", "c"], size=15)
        sampler = BootstrapSampler()
        X_bootstrap = sampler.apply(X, y)
        self.assertFalse(numpy.all(X == X_bootstrap))
        for row in X_bootstrap:
            self.assertIn(row, X)
        numpy.testing.assert_array_equal(X.shape, X_bootstrap.shape)

    def test_bootstrap_specified_percent(self):
        X = numpy.random.rand(20, 3)
        y = numpy.random.choice(["a", "b", "c"], size=20)
        sampler = BootstrapSampler(sampling_percent=150)
        X_bootstrap = sampler.apply(X, y)
        self.assertFalse(numpy.all(X == X_bootstrap))
        for row in X_bootstrap:
            self.assertIn(row, X)
        numpy.testing.assert_array_equal([30, 3], X_bootstrap.shape)
