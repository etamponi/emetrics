from itertools import product
import cPickle
import multiprocessing
import os
import signal

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier

from emetrics.coefficients import AssociationMeasure
from emetrics.coefficients.determination_coefficient import DeterminationCoefficient
from emetrics.coefficients.uncertainty_coefficient import UncertaintyCoefficient
from emetrics.correlation_score import CorrelationScore
from emetrics.evaluation import dataset_names
from emetrics.evaluation.random_subsets_experiment import RandomSubsetsExperiment
from emetrics.label_encoders.onehot_label_encoder import OneHotLabelEncoder
from emetrics.label_encoders.ordinal_label_encoder import OrdinalLabelEncoder
from emetrics.preparers.noise_injector import NoiseInjector


__author__ = 'Emanuele Tamponi'


def main():
    datasets = dataset_names()
    should_normalize = [True, False]
    # All sizes up to 10, then 15, 20, 25, 30 and 35
    subset_sizes = range(1, 11) + range(15, 36, 5)

    pool = multiprocessing.Pool(4, init_worker)
    try:
        for dataset, normalize, subset_size in product(datasets, should_normalize, subset_sizes):
            pool.apply_async(run_parallel, args=(dataset, normalize, subset_size))
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        print "Keyboard Interrupt, terminating..."
        pool.terminate()
        pool.join()


def run_parallel(dataset, normalize, subset_size):
    results_file_name = get_results_file_name(dataset, normalize, subset_size)
    print "{:>5s} - {}".format("start", results_file_name)
    if os.path.isfile("results/{}.res".format(results_file_name)):
        print "{:>5s} - {}.".format("done", results_file_name)
        return
    results = get_experiment(dataset, normalize, subset_size).run()
    if results is not None:
        with open("results/{}.res".format(results_file_name), "w") as f:
            cPickle.dump(results, f)
        print "{:>5s} - {}.".format("done", results_file_name)
    else:
        print "{:>5s} - {}.".format("n/a", results_file_name)


def get_experiment(dataset_name, normalize, subset_size):
    return RandomSubsetsExperiment(
        dataset=dataset_name,
        subset_size=subset_size,
        scorers=[
            ("wilks", CorrelationScore(
                coefficient=AssociationMeasure(
                    measure="wilks"
                ),
                preparer_pipeline=[
                    NoiseInjector(stddev=1e-4),
                ],
                label_encoder=OrdinalLabelEncoder()
            )),
            ("deter", CorrelationScore(
                coefficient=DeterminationCoefficient(),
                preparer_pipeline=[
                    NoiseInjector(stddev=1e-4),
                ],
                label_encoder=OneHotLabelEncoder()
            )),
            ("uncer", CorrelationScore(
                coefficient=UncertaintyCoefficient(),
                preparer_pipeline=[
                    NoiseInjector(stddev=1e-4),
                ],
                label_encoder=OrdinalLabelEncoder()
            ))
        ],
        classifiers=[
            ("ab", AdaBoostClassifier(n_estimators=100)),
            ("gb", GradientBoostingClassifier(n_estimators=100)),
            ("ba", BaggingClassifier(n_estimators=100)),
            ("rf", RandomForestClassifier(n_estimators=100, max_features="log2")),
            ("et", ExtraTreesClassifier(n_estimators=100, max_features="log2"))
        ],
        n_folds=5,
        n_runs=10,
        normalize=normalize
    )


def get_results_file_name(dataset, normalize, subset_size):
    return "{}_{}_{:02d}".format(
        dataset,
        "norm" if normalize else "orig",
        subset_size
    )


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


if __name__ == "__main__":
    main()
