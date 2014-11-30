import time

import numpy

import sklearn
from sklearn.cross_validation import StratifiedKFold

from sklearn.preprocessing.data import normalize, Normalizer

from analysis.dataset_utils import ArffLoader


__author__ = 'Emanuele Tamponi'


class RandomSubsetsExperiment(object):

    def __init__(self, dataset, subset_size, scorers, classifiers, n_folds, n_runs):
        self.dataset = dataset
        self.subset_size = subset_size
        self.scorers = scorers
        self.classifiers = classifiers
        self.n_folds = n_folds
        self.n_runs = n_runs

    def run(self, directory="datasets/"):
        loader = ArffLoader("{}/{}.arff".format(directory, self.dataset))
        if loader.feature_num() <= self.subset_size:
            return None
        X, y = loader.get_dataset()
        n_features = X.shape[1]
        results = {
            "experiment": self,
            "scores": {scorer_name: numpy.zeros(self.n_runs) for scorer_name, _ in self.scorers},
            "score_times": {scorer_name: numpy.zeros(self.n_runs) for scorer_name, _ in self.scorers},
            "errors": {classifier_name: numpy.zeros(self.n_runs) for classifier_name, _ in self.classifiers},
            "classifier_times": {classifier_name: numpy.zeros(self.n_runs) for classifier_name, _ in self.classifiers}
        }
        for run in range(self.n_runs):
            numpy.random.seed(run)
            indices = numpy.random.choice(n_features, size=self.subset_size, replace=False)
            X_subset = X[:, indices].copy()
            for scorer_name, scorer in self.scorers:
                score, t = self._execute_score_run(run, scorer, X_subset, y)
                results["scores"][scorer_name][run] = score
                results["score_times"][scorer_name][run] = t
            for classifier_name, classifier in self.classifiers:
                error, t = self._execute_classifier_run(run, sklearn.clone(classifier), X_subset, y)
                results["errors"][classifier_name][run] = error
                results["classifier_times"][classifier_name][run] = t
        return results

    def _execute_score_run(self, run, scorer, inputs, labels):
        numpy.random.seed(run)
        t_start = time.time()
        mean_score = 0.0
        for train_indices, test_indices in StratifiedKFold(labels, n_folds=self.n_folds):
            inputs_train, labels_train = inputs[train_indices], labels[train_indices]
            inputs_train = normalize(inputs_train, norm="l1")
            mean_score += scorer(inputs_train, labels_train)
        mean_score /= self.n_folds
        t_stop = time.time()
        t = t_stop - t_start
        return mean_score, t

    def _execute_classifier_run(self, run, classifier, inputs, labels):
        numpy.random.seed(run)
        t_start = time.time()
        mean_error = 0.0
        for train_indices, test_indices in StratifiedKFold(labels, n_folds=self.n_folds):
            inputs_train, labels_train = inputs[train_indices], labels[train_indices]
            normalizer = Normalizer(norm="l1").fit(inputs_train)
            classifier.fit(normalizer.transform(inputs_train), labels_train)
            inputs_test, labels_test = inputs[test_indices], labels[test_indices]
            accuracy = classifier.score(normalizer.transform(inputs_test), labels_test)
            mean_error += 1 - accuracy
        mean_error /= self.n_folds
        t_stop = time.time()
        t = t_stop - t_start
        return mean_error, t
