import unittest
import arff
import numpy
from emetrics.coefficients.association_measure import AssociationMeasure

__author__ = 'Emanuele Tamponi'


class AssociationMeasureTest(unittest.TestCase):

    def test_score_is_zero(self):
        X = numpy.ones((10, 2))
        y = numpy.random.choice(["a", "b"], size=10)
        s = AssociationMeasure(noise_level=0)(X, y)
        self.assertEqual(0, s, msg="CR should be 0, got {}".format(s))
        s = AssociationMeasure()(X, y)
        self.assertNotEqual(0, s, msg="CR should not be 0 with noise, got 0!")

    def test_score_is_one(self):
        X = numpy.ones((10, 2))
        X[:5, :] = 0
        y = ["a"] * 5 + ["b"] * 5
        cr = AssociationMeasure(noise_level=0)
        self.assertEqual(1, cr(X, y), msg="CR should be 1, got {}".format(cr(X, y)))

    def test_roys_score_on_dataset(self):
        # Result taken from Rencher's book Methods for Multivariate Analysis
        with open('test_files/aggregation_score_dataset.arff') as f:
            data = numpy.asarray(arff.load(f)["data"])
            X, y = data[:, :-1].astype(numpy.float64), data[:, -1]
            s = AssociationMeasure(measure="roy")(X, y)
            self.assertAlmostEqual(0.652, s, places=3, msg="Roy's Score not working, got {}".format(s))

    def test_pillais_score_on_dataset(self):
        with open('test_files/aggregation_score_dataset.arff') as f:
            data = numpy.asarray(arff.load(f)["data"])
            X, y = data[:, :-1].astype(numpy.float64), data[:, -1]
            s = AssociationMeasure(measure="pillai")(X, y)
            self.assertAlmostEqual(0.326, s, places=3, msg="Pillai's Score not working, got {}".format(s))

    def test_lawley_score_on_dataset(self):
        with open('test_files/aggregation_score_dataset.arff') as f:
            data = numpy.asarray(arff.load(f)["data"])
            X, y = data[:, :-1].astype(numpy.float64), data[:, -1]
            s = AssociationMeasure(measure="lawley")(X, y)
            self.assertAlmostEqual(0.422, s, places=3, msg="Lawley's Score not working, got {}".format(s))

    def test_wilks_score_on_dataset(self):
        # Result taken from Rencher's book Methods for Multivariate Analysis
        with open('test_files/aggregation_score_dataset.arff') as f:
            data = numpy.asarray(arff.load(f)["data"])
            X, y = data[:, :-1].astype(numpy.float64), data[:, -1]
            s = AssociationMeasure()(X, y)
            self.assertAlmostEqual(1 - 0.154, s, places=3, msg="Wilks' Score not working, got {}".format(s))

    def test_armonic_score_on_dataset(self):
        # Result taken from Rencher's book Methods for Multivariate Analysis
        with open('test_files/aggregation_score_dataset.arff') as f:
            data = numpy.asarray(arff.load(f)["data"])
            X, y = data[:, :-1].astype(numpy.float64), data[:, -1]
            s = AssociationMeasure("armonic")(X, y)
            self.assertAlmostEqual(0.374, s, places=3, msg="Armonic Score not working, got {}".format(s))

    def test_inputs_not_changed(self):
        X, y = numpy.random.rand(10, 3), numpy.random.choice(["a", "b"], size=10)
        X_copy, y_copy = X.copy(), y.copy()
        score = AssociationMeasure()(X, y)
        numpy.testing.assert_array_equal(X_copy, X)
        numpy.testing.assert_array_equal(y_copy, y)
