import time

import numpy
from scipy.stats.stats import pearsonr
import sklearn
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing.data import normalize
from sklearn.tree import DecisionTreeClassifier

from analysis.dataset_utils import ArffLoader
from emetrics.coefficients.association_measure import AssociationMeasure
from emetrics.coefficients.uncertainty_coefficient import UncertaintyCoefficient
from emetrics.correlation_score import CorrelationScore
from emetrics.evaluation.random_subsets_experiment import RandomSubsetsExperiment
from emetrics.label_encoders.ordinal_label_encoder import OrdinalLabelEncoder
from emetrics.preparers.noise_injector import NoiseInjector


__author__ = 'Emanuele Tamponi'


def main():
    subset_sizes = range(5, 6)
    for subset_size in subset_sizes:
        experiment = RandomSubsetsExperiment(
            dataset="colic",
            subset_size=subset_size,
            scorers=[
                ("wilks", CorrelationScore(
                    preparer_pipeline=[
                        NoiseInjector(stddev=1e-6)
                    ]
                ))
            ],
            classifiers=[
                ("dt", DecisionTreeClassifier()),
                ("rf", RandomForestClassifier())
            ],
            n_folds=10,
            n_runs=10
        )
        results = experiment.run()
        print "Average score time: {:.6f}".format(results["score_times"]["wilks"].mean())
        for classifier in results["errors"]:
            print "Average {} time: {:.6f}".format(classifier, results["classifier_times"][classifier].mean())
            corr, _ = pearsonr(results["scores"]["wilks"], results["errors"][classifier])
            print "Pearson Correlation with score: {:.3f}".format(corr)

        print results["errors"]["rf"]


if __name__ == "__main__":
    main()
