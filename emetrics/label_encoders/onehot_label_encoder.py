from sklearn.preprocessing.data import OneHotEncoder
from sklearn.preprocessing.label import LabelEncoder

__author__ = 'Emanuele Tamponi'


class OneHotLabelEncoder(object):

    def __call__(self, labels):
        labels = LabelEncoder().fit_transform(labels)
        labels = labels.reshape((len(labels), 1))
        labels = OneHotEncoder(sparse=False).fit_transform(labels)
        if labels.shape[1] == 2:
            return labels[:, 0].reshape((len(labels), 1))
        else:
            return labels
