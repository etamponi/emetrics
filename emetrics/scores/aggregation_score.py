import numpy

__author__ = 'Emanuele Tamponi'


class AggregationScore(object):

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
        try:
            inv_w_matrix = numpy.linalg.inv(w_matrix)
        except numpy.linalg.LinAlgError:
            if numpy.all(b_matrix == 0):
                return 0
            else:
                return 1
        eig_vals = numpy.sort(abs(numpy.linalg.eigvals(numpy.dot(inv_w_matrix, b_matrix))))[::-1]
        s = min(len(classes)-1, feature_num)
        wilks_lambda = 1
        for i in range(s):
            wilks_lambda *= 1.0 / (1.0 + eig_vals[i])
        return 1 - wilks_lambda
