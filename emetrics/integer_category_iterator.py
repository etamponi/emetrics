from sklearn.preprocessing import LabelEncoder

__author__ = 'Emanuele Tamponi'


class IntegerCategoryIterator(object):

    def __call__(self, labels):
        yield LabelEncoder().fit_transform(labels)
