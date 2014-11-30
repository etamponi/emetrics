import numpy

__author__ = 'Emanuele Tamponi'


class BootstrapSampler(object):

    def __init__(self, sampling_percent=100):
        self.sampling_percent = float(sampling_percent) / 100

    def apply(self, inputs, labels):
        indices = numpy.random.choice(len(inputs), size=int(self.sampling_percent*len(inputs)))
        return inputs[indices]
