import numpy

__author__ = 'Emanuele Tamponi'


class BootstrapSampler(object):

    def apply(self, inputs, labels):
        indices = numpy.random.choice(len(inputs), size=len(inputs))
        return inputs[indices]
