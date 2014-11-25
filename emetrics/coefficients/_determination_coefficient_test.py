import unittest
import numpy
from emetrics.coefficients.determination_coefficient import DeterminationCoefficient

__author__ = 'Emanuele Tamponi'


class DeterminationCoefficientTest(unittest.TestCase):

    def test_determination_works(self):
        for i in range(10):
            inputs = numpy.random.rand(100, 3)
            labels = numpy.random.choice(["a", "b"], size=100)
            value = DeterminationCoefficient()(inputs, labels)
            self.assertTrue(0 < value < 1)

    def test_determination_is_zero(self):
        inputs = numpy.zeros((100, 3))
        labels = numpy.random.choice(["a", "b"], size=100)
        value = DeterminationCoefficient()(inputs, labels)
        self.assertEqual(0, value)

    def test_determination_is_one(self):
        inputs = numpy.zeros((100, 3))
        inputs[50:, :] = 10
        labels = numpy.zeros(100)
        labels[50:] = 1
        value = DeterminationCoefficient()(inputs, labels)
        self.assertEqual(1, value)
