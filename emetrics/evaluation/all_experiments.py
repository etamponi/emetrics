from itertools import product
import cPickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.bagging import BaggingClassifier
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
    # All sizes up to 10, then 15, 20, 25, 30 and 35
    subset_sizes = range(1, 11) + range(15, 36, 5)
    datasets = dataset_names(n_groups=2, group=0)
    for dataset, normalize, subset_size in list(product(datasets, [True, False], subset_sizes)):
        results_file_name = get_results_file_name(dataset, normalize, subset_size)
        print "Running {} with {} features{}".format(dataset, subset_size, " and normalization" if normalize else ""),
        if os.path.isfile("results/{}.res".format(results_file_name)):
            print "--- experiment already ran, continuing..."
            continue
        results = get_experiment(dataset, subset_size, normalize).run()
        if results is not None:
            with open("results/{}.res".format(results_file_name), "w") as f:
                cPickle.dump(results, f)
            print ""
        else:
            print "\r",


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


if __name__ == "__main__":
    main()
