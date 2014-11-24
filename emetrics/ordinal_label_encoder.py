import numpy
from sklearn.preprocessing import LabelEncoder

__author__ = 'Emanuele Tamponi'


class OrdinalLabelEncoder(object):

    def __call__(self, labels):
        return numpy.asarray(LabelEncoder().fit_transform(labels)).reshape((len(labels), 1))
