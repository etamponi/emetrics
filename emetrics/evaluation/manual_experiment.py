import time

import numpy
from scipy.stats.stats import pearsonr
import sklearn
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from analysis.dataset_utils import ArffLoader
from emetrics.coefficients.uncertainty_coefficient import UncertaintyCoefficient
from emetrics.correlation_score import CorrelationScore
from emetrics.label_encoders.ordinal_label_encoder import OrdinalLabelEncoder


__author__ = 'Emanuele Tamponi'


def main():
    gaussian_noise = lambda shape: numpy.random.randn(*shape)
    uniform_noise = lambda shape: numpy.random.uniform(-1.0, 1.0, size=shape)

    score = CorrelationScore(
        coefficient=UncertaintyCoefficient(),
        label_encoder=OrdinalLabelEncoder()
    )

    classifiers = {
        "dt": DecisionTreeClassifier(),
        "rf100": RandomForestClassifier(n_estimators=10),
    }

    subset_sizes = range(1, 5)
    n_runs = 10
    n_folds = 10

    for dataset_name in ["iris"]:
        print "Start", dataset_name
        X, y = ArffLoader("datasets/{}.arff".format(dataset_name)).load_dataset()
        n_features = X.shape[1]
        results = {}
        for subset_size in subset_sizes:
            if subset_size >= n_features:
                break
            print "Considering", subset_size, "features in", dataset_name
            scores = numpy.zeros(n_runs)
            errors = {key: numpy.zeros(n_runs) for key in classifiers}
            score_times = numpy.zeros(n_runs)
            class_times = {key: numpy.zeros(n_runs) for key in classifiers}
            for run in range(n_runs):
                numpy.random.seed(run)
                subset = tuple(numpy.random.choice(n_features, size=subset_size, replace=False))
                X_subset = X[:, subset].copy()

                t0 = time.time()
                numpy.random.seed(run)
                for train_indices, test_indices in StratifiedKFold(y, n_folds=n_folds):
                    X_train, y_train = X_subset[train_indices], y[train_indices]
                    X_test, y_test = X_subset[test_indices], y[test_indices]

                    # sampling_indices = numpy.random.choice(len(X_train), size=int(10.0*len(X_train)))
                    # X_train_sampled = X_train[sampling_indices]
                    # y_train_sampled = y_train[sampling_indices]
                    # scores[run] += score(X_train_sampled, y_train_sampled)

                    scores[run] += score(X_train, y_train) / n_folds
                t1 = time.time()
                score_times[run] = t1 - t0

                for name, orig_classifier in classifiers.iteritems():
                    t0 = time.time()
                    numpy.random.seed(run)
                    for train_indices, test_indices in StratifiedKFold(y, n_folds=n_folds):
                        classifier = sklearn.clone(orig_classifier)
                        X_train, y_train = X_subset[train_indices], y[train_indices]
                        X_test, y_test = X_subset[test_indices], y[test_indices]
                        errors[name][run] += \
                            1 - classifier.fit(X_train, y_train).score(X_test, y_test) / n_folds
                    t1 = time.time()
                    class_times[name][run] = t1 - t0
            results[subset_size] = {
                "scores": scores,
                "errors": errors,
                "score_times": score_times,
                "class_times": class_times,
                "n_folds": n_folds
            }
            print "Subset size", subset_size
            for name in classifiers:
                print "Pearson with", name, ":", pearsonr(scores, errors[name])

if __name__ == "__main__":
    main()
