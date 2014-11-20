import numpy
from sklearn.preprocessing.data import OneHotEncoder
from sklearn.preprocessing.label import LabelEncoder

__author__ = 'Emanuele Tamponi'


class OneHotCategoryIterator(object):

    def __call__(self, labels):
        labels = LabelEncoder().fit_transform(labels)
        labels = labels.reshape((len(labels), 1))
        classes = numpy.unique(labels)
        labels = OneHotEncoder(sparse=False).fit_transform(labels)
        if len(classes) == 2:
            yield labels.T[0]
            return
        for column in labels.T:
            yield column
