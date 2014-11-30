from itertools import product
import cPickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.svm.classes import SVC
from sklearn.tree import DecisionTreeClassifier
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
    subset_sizes = range(1, 26)
    datasets = dataset_names(n_groups=4, group=0)
    for normalize, dataset, subset_size in product([True, False], datasets, subset_sizes):
        if normalize:
            results_file = "results/{}_{:02d}_norm.res".format(dataset, subset_size)
        else:
            results_file = "results/{}_{:02d}.res".format(dataset, subset_size)
        if os.path.isfile(results_file):
            print "Experiment already ran, continuing..."
            continue
        print "Running {} with {} features".format(dataset, subset_size)
        results = get_experiment(dataset, subset_size, normalize).run()
        if results is not None:
            with open(results_file) as f:
                cPickle.dump(results, f)


def get_experiment(dataset_name, subset_size, normalize):
    RandomSubsetsExperiment(
        dataset=dataset_name,
        subset_size=subset_size,
        scorers=[
            ("wilks", CorrelationScore(
                coefficient=AssociationMeasure(
                    measure="wilks"
                ),
                preparer_pipeline=[
                    NoiseInjector(stddev=1e-6),
                ],
                label_encoder=OrdinalLabelEncoder()
            )),
            ("deter", CorrelationScore(
                coefficient=DeterminationCoefficient(),
                preparer_pipeline=[
                    NoiseInjector(stddev=1e-6),
                ],
                label_encoder=OneHotLabelEncoder()
            )),
            ("uncer", CorrelationScore(
                coefficient=UncertaintyCoefficient(),
                preparer_pipeline=[
                    NoiseInjector(stddev=1e-6),
                ],
                label_encoder=OrdinalLabelEncoder()
            ))
        ],
        classifiers=[
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("bo", AdaBoostClassifier(n_estimators=100)),
            ("sv", SVC())
        ],
        n_folds=5,
        n_runs=10,
        normalize=normalize
    )


if __name__ == "__main__":
    main()
