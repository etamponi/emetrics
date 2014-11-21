import unittest
import arff
import numpy
from emetrics.scores.aggregation_score import AggregationScore

__author__ = 'Emanuele Tamponi'


class AggregationScoreTest(unittest.TestCase):

    def test_score_on_dataset(self):
        # Result taken from Rencher's book Methods for Multivariate Analysis
        with open('test_files/aggregation_score_dataset.arff') as f:
            data = arff.load(f)
            X = numpy.asarray([row[:-1] for row in data["data"]])
            y = numpy.asarray([row[-1] for row in data["data"]])
            cr = AggregationScore()
            self.assertAlmostEqual(1 - 0.154, cr(X, y), places=3, msg="Wilks Correlation Ratio not working")

    def test_score_is_zero(self):
        X = numpy.ones((10, 2))
        y = numpy.random.choice(["a", "b"], size=10)
        cr = AggregationScore()
        self.assertEqual(0, cr(X, y), msg="CR should be 0, got {}".format(cr(X, y)))

    def test_score_is_one(self):
        X = numpy.ones((10, 2))
        X[:5, :] = 0
        y = ["a"] * 5 + ["b"] * 5
        cr = AggregationScore()
        self.assertEqual(1, cr(X, y), msg="CR should be 1, got {}".format(cr(X, y)))
