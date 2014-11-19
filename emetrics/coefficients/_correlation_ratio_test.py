import unittest
import arff
import numpy
from emetrics.coefficients.correlation_ratio import CorrelationRatio

__author__ = 'Emanuele Tamponi'


class CorrelationRatioTest(unittest.TestCase):

    def test_correlation_ratio_on_dataset(self):
        data = arff.load(open('test_files/correlation_ratio_dataset.arff'))
        X = numpy.asarray([row[:-1] for row in data["data"]])
        y = numpy.asarray([row[-1] for row in data["data"]])
        cr = CorrelationRatio()
        self.assertAlmostEqual(1 - 0.154, cr(X, y), places=3, msg="Wilks Correlation Ratio not working")

    def test_correlation_ratio_zero_cr(self):
        X = numpy.ones((10, 2))
        y = numpy.random.choice(["a", "b"], size=10)
        cr = CorrelationRatio()
        self.assertEqual(0, cr(X, y), msg="CR should be 0, got {}".format(cr(X, y)))

    def test_correlation_ratio_one_cr(self):
        X = numpy.ones((10, 2))
        X[:5, :] = 0
        y = ["a"] * 5 + ["b"] * 5
        cr = CorrelationRatio()
        self.assertEqual(1, cr(X, y), msg="CR should be 1, got {}".format(cr(X, y)))
