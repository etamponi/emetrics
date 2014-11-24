import unittest

import numpy

from emetrics.label_encoders.onehot_label_encoder import OneHotLabelEncoder


__author__ = 'Emanuele Tamponi'


class OneHotCategoryIteratorTest(unittest.TestCase):

    def test_onehot_iteration(self):
        labels = numpy.asarray(list("aaaaabbbbbccccc"))
        expected_onehot = numpy.zeros((15, 3))
        expected_onehot[0:5, 0] = 1
        expected_onehot[5:10, 1] = 1
        expected_onehot[10:, 2] = 1

        enc = OneHotLabelEncoder()
        onehot = enc(labels)
        numpy.testing.assert_array_equal(expected_onehot, onehot)

    def test_onehot_binary_iteration(self):
        labels = numpy.asarray(list("aaaaabbbbb"))
        expected_onehot = numpy.asarray([5*[1] + 5*[0]]).T
        enc = OneHotLabelEncoder()
        onehot = enc(labels)
        numpy.testing.assert_array_equal(expected_onehot, onehot)
