import numpy

__author__ = 'Emanuele Tamponi'


class CorrelationRatio(object):

    def __call__(self, inputs, labels):
        classes = numpy.unique(labels)
        feature_num = inputs.shape[1]
        class_means = numpy.zeros((len(classes), feature_num))
        class_sizes = numpy.zeros(len(classes))
        split_inputs = []
        for i, c in enumerate(classes):
            split_inputs.append(inputs[labels == c])
            class_means[i] = split_inputs[-1].mean(axis=0)
            class_sizes[i] = len(split_inputs[-1])
        mean_input = inputs.mean(axis=0)
        b_matrix = numpy.zeros((feature_num, feature_num))
        for class_mean, class_size in zip(class_means, class_sizes):
            class_shift = (class_mean - mean_input).reshape((feature_num, 1))
            b_matrix += class_size * numpy.dot(class_shift, class_shift.transpose())
        w_matrix = numpy.zeros((feature_num, feature_num))
        for class_mean, class_inputs in zip(class_means, split_inputs):
            for x in class_inputs:
                shift = (x - class_mean).reshape((feature_num, 1))
                w_matrix += numpy.dot(shift, shift.transpose())
        numerator = numpy.linalg.det(w_matrix)
        denominator = numpy.linalg.det(w_matrix + b_matrix)
        if numpy.all(b_matrix == 0):
            wilks_lambda = 1
        elif numpy.all(w_matrix == 0):
            wilks_lambda = 0
        else:
            wilks_lambda = numerator / denominator
        return 1 - wilks_lambda
