from scipy.stats.stats import pearsonr

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from emetrics.coefficients.association_measure import AssociationMeasure

from emetrics.correlation_score import CorrelationScore

from emetrics.evaluation.random_subsets_experiment import RandomSubsetsExperiment
from emetrics.label_encoders.ordinal_label_encoder import OrdinalLabelEncoder
from emetrics.preparers.bootstrap_sampler import BootstrapSampler
from emetrics.preparers.noise_injector import NoiseInjector


__author__ = 'Emanuele Tamponi'


def main():
    subset_sizes = range(1, 6)
    for subset_size in subset_sizes:
        experiment = RandomSubsetsExperiment(
            dataset="iris",
            subset_size=subset_size,
            scorers=[
                ("wilks", CorrelationScore(
                    coefficient=AssociationMeasure(
                        measure="wilks"
                    ),
                    preparer_pipeline=[
                        BootstrapSampler(sampling_percent=100),
                        NoiseInjector(stddev=1e-6),
                    ],
                    label_encoder=OrdinalLabelEncoder()
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
        if results is None:
            continue
        for classifier in results["errors"]:
            corr, _ = pearsonr(results["scores"]["wilks"], results["errors"][classifier])
            print "{} - correlation with {}: {:.3f}".format(subset_size, classifier, corr)


if __name__ == "__main__":
    main()
