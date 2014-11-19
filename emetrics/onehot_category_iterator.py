import numpy

__author__ = 'Emanuele Tamponi'


class OneHotCategoryIterator(object):

    def __call__(self, labels):
        onehot_labels = numpy.zeros(15)
        onehot_labels[:5] = 1
        yield onehot_labels
        onehot_labels[:5] = 0
        onehot_labels[5:10] = 1
        yield onehot_labels
        onehot_labels[5:10] = 0
        onehot_labels[10:] = 1
        yield onehot_labels
