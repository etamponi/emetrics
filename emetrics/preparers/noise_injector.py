import numpy

__author__ = 'Emanuele Tamponi'


class NoiseInjector(object):

    def __init__(self, stddev=1e-6):
        self.stddev = stddev

    def apply(self, inputs, labels):
        return inputs.copy() + self.stddev * numpy.random.randn(*inputs.shape)
