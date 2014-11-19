import unittest
import numpy
from emetrics.onehot_category_iterator import OneHotCategoryIterator

__author__ = 'Emanuele Tamponi'


class OneHotCategoryIteratorTest(unittest.TestCase):

    def test_onehot_iteration(self):
        labels = numpy.asarray(list("aaaaabbbbbccccc"))
        expected_onehot_labels = numpy.zeros((3, 15))
        expected_onehot_labels[0][:5] = 1
        expected_onehot_labels[1][5:10] = 1
        expected_onehot_labels[2][10:] = 1

        it = OneHotCategoryIterator()
        for i, onehot_labels in enumerate(it(labels)):
            expected_onehot_labels = numpy.zeros(15)
            expected_onehot_labels[i*5:(i+1)*5] = 1
            numpy.testing.assert_array_equal(expected_onehot_labels, onehot_labels,
                                             err_msg="one hot encoding not equal for label {}".format(i))