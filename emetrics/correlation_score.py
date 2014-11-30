import numpy

from emetrics.coefficients.association_measure import AssociationMeasure
from emetrics.label_encoders.ordinal_label_encoder import OrdinalLabelEncoder


__author__ = 'Emanuele Tamponi'


class CorrelationScore(object):

    def __init__(self,
                 coefficient=AssociationMeasure(),
                 preparer_pipeline=None,
                 label_encoder=OrdinalLabelEncoder()):
        self.coefficient = coefficient
        if preparer_pipeline is None:
            self.preparer_pipeline = []
        else:
            self.preparer_pipeline = preparer_pipeline
        self.label_encoder = label_encoder

    def __call__(self, inputs, labels):
        for preparer in self.preparer_pipeline:
            inputs = preparer.apply(inputs, labels)
        encoded_labels = self.label_encoder(labels)
        values = numpy.zeros(encoded_labels.shape[1])
        for i, y in enumerate(encoded_labels.T):
            values[i] = self.coefficient(inputs, y)
        return values.mean()
