from itertools import product
import cPickle
import multiprocessing
import os
import signal
from time import sleep

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
import sys

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

    pool = multiprocessing.Pool(processes=4)
    signal.signal(signal.SIGINT, terminate_pool(pool))

    results = pool.map_async(run_parallel, product(datasets, should_normalize, subset_sizes))
    results.get(24*3600*1000)  # Wait for 1000 days
    pool.join()


def run_parallel(params):
    dataset, normalize, subset_size = params
    results_file_name = get_results_file_name(dataset, normalize, subset_size)
    print "starting experiment {}".format(results_file_name)
    if os.path.isfile("results/{}.res".format(results_file_name)):
        print "results of {} already present.".format(results_file_name)
        return
    results = get_experiment(dataset, subset_size, normalize).run()
    if results is not None:
        with open("results/{}.res".format(results_file_name), "w") as f:
            cPickle.dump(results, f)
        print "results of {} saved.".format(results_file_name)
    else:
        print "experiment {} not available (subset_size >= n_features)".format(results_file_name)


def get_experiment(dataset_name, subset_size, normalize):
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
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("ab", AdaBoostClassifier(n_estimators=100)),
            ("ba", BaggingClassifier(n_estimators=100))
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


def terminate_pool(pool):
    def signal_handler(sig, frame):
        print "Terminating..."
        pool.terminate()
        pool.join()
        sys.exit(0)
    return signal_handler


if __name__ == "__main__":
    main()
