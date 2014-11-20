import unittest
import numpy
from emetrics.integer_category_iterator import IntegerCategoryIterator

__author__ = 'Emanuele Tamponi'


class IntegerCategoryIteratorTest(unittest.TestCase):

    def test_iterator(self):
        labels = list("aaaaabbbbbccccc")
        expected_vector = numpy.asarray(5*[0] + 5*[1] + 5*[2])
        it = IntegerCategoryIterator()
        vectors = list(it(labels))
        self.assertEqual(1, len(vectors))
        numpy.testing.assert_array_equal(expected_vector, vectors[0])
