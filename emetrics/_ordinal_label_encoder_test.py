import unittest
import numpy
from emetrics.ordinal_label_encoder import OrdinalLabelEncoder

__author__ = 'Emanuele Tamponi'


class OrdinalLabelEncoderTest(unittest.TestCase):

    def test_encoder(self):
        labels = list("aaaaabbbbbccccc")
        expected_labels = numpy.asarray([5*[0] + 5*[1] + 5*[2]]).T
        enc = OrdinalLabelEncoder()
        labels = enc(labels)
        numpy.testing.assert_array_equal(expected_labels, labels)
