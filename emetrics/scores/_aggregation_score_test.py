import unittest
import arff
import numpy
from emetrics.scores.aggregation_score import AggregationScore

__author__ = 'Emanuele Tamponi'


class AggregationScoreTest(unittest.TestCase):

    def test_wilks_score_on_dataset(self):
        # Result taken from Rencher's book Methods for Multivariate Analysis
        with open('test_files/aggregation_score_dataset.arff') as f:
            data = numpy.asarray(arff.load(f)["data"])
            X, y = data[:, :-1].astype(numpy.float64), data[:, -1]
            s = AggregationScore()(X, y)
            self.assertAlmostEqual(1 - 0.154, s, places=3, msg="Wilks' Score not working, got {}".format(s))

    def test_score_is_zero(self):
        X = numpy.ones((10, 2))
        y = numpy.random.choice(["a", "b"], size=10)
        cr = AggregationScore(noise=0)
        self.assertEqual(0, cr(X, y), msg="CR should be 0, got {}".format(cr(X, y)))

    def test_score_is_one(self):
        X = numpy.ones((10, 2))
        X[:5, :] = 0
        y = ["a"] * 5 + ["b"] * 5
        cr = AggregationScore()
        self.assertEqual(1, cr(X, y), msg="CR should be 1, got {}".format(cr(X, y)))

    def test_roys_score_on_dataset(self):
        # Result taken from Rencher's book Methods for Multivariate Analysis
        with open('test_files/aggregation_score_dataset.arff') as f:
            data = numpy.asarray(arff.load(f)["data"])
            X, y = data[:, :-1].astype(numpy.float64), data[:, -1]
            s = AggregationScore(score="roy")(X, y)
            self.assertAlmostEqual(0.652, s, places=3, msg="Roy's Score not working, got {}".format(s))

    def test_pillais_score_on_dataset(self):
        with open('test_files/aggregation_score_dataset.arff') as f:
            data = numpy.asarray(arff.load(f)["data"])
            X, y = data[:, :-1].astype(numpy.float64), data[:, -1]
            s = AggregationScore(score="pillai")(X, y)
            self.assertAlmostEqual(0.326, s, places=3, msg="Pillai's Score not working, got {}".format(s))

    def test_lawley_score_on_dataset(self):
        with open('test_files/aggregation_score_dataset.arff') as f:
            data = numpy.asarray(arff.load(f)["data"])
            X, y = data[:, :-1].astype(numpy.float64), data[:, -1]
            s = AggregationScore(score="lawley")(X, y)
            self.assertAlmostEqual(0.422, s, places=3, msg="Pillai's Score not working, got {}".format(s))
