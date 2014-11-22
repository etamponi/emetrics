import numpy

__author__ = 'Emanuele Tamponi'


class AggregationScore(object):

    def __init__(self, score="wilks", noise_level=1e-6):
        self.score = score
        self.noise_level = noise_level

    def __call__(self, inputs, labels):
        inputs = self._inject_noise(inputs)
        w_matrix, b_matrix, rank = self._calculate_matrices(inputs, labels)
        try:
            eigenvalues = self._calculate_eigenvalues(w_matrix, b_matrix, rank)
        except numpy.linalg.LinAlgError:
            if numpy.all(b_matrix == 0):
                return 0
            else:
                return 1
        if self.score in {"wilks", "armonic"}:
            wilks_lambda = 1
            for eigenvalue in eigenvalues:
                wilks_lambda *= 1.0 / (1.0 + eigenvalue)
            if self.score == "armonic":
                return 1 - pow(wilks_lambda, 1.0 / rank)
            else:
                return 1 - wilks_lambda
        if self.score == "roy":
            return eigenvalues[0] / (1 + eigenvalues[0])
        if self.score == "pillai":
            v = (eigenvalues / (1 + eigenvalues)).sum()
            return v / rank
        if self.score == "lawley":
            v = eigenvalues.sum() / rank
            return v / (1 + v)

    def _inject_noise(self, inputs):
        noise = self.noise_level * (2*numpy.random.rand(*inputs.shape) - 1)
        return inputs + noise

    @staticmethod
    def _calculate_matrices(inputs, labels):
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
        rank = min(len(classes) - 1, feature_num)
        return w_matrix, b_matrix, rank

    @staticmethod
    def _calculate_eigenvalues(w_matrix, b_matrix, rank):
        inv_w_matrix = numpy.linalg.inv(w_matrix)
        eigenvalues = numpy.sort(abs(numpy.linalg.eigvals(numpy.dot(inv_w_matrix, b_matrix))))[::-1]
        return eigenvalues[:rank]
