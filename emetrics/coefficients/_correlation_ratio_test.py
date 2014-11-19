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
