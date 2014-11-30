import pickle
import unittest
import numpy

from sklearn.tree.tree import DecisionTreeClassifier

from emetrics.correlation_score import CorrelationScore
from emetrics.evaluation.random_subsets_experiment import RandomSubsetsExperiment


__author__ = 'Emanuele Tamponi'


class RandomSubsetsExperimentTest(unittest.TestCase):
    
    def test_results_in_shape(self):
        experiment = RandomSubsetsExperiment(
            dataset="aggregation_score_dataset",
            subset_size=2,
            scorers=[("wilks", CorrelationScore())],
            classifiers=[("dt", DecisionTreeClassifier())],
            n_folds=5,
            n_runs=10
        )
        results = experiment.run(directory="test_files/")
        self.assertEqual(["wilks"], list(results["scores"].keys()))
        self.assertEqual(["dt"], list(results["errors"].keys()))
        self.assertEqual(10, len(results["scores"]["wilks"]))
        self.assertEqual(10, len(results["errors"]["dt"]))
        for score in results["scores"]["wilks"]:
            self.assertTrue(0 < score < 1)
        for error in results["errors"]["dt"]:
            self.assertTrue(0 < error < 1)
        for time in results["score_times"]["wilks"]:
            self.assertTrue(time > 0)
        for time in results["classifier_times"]["dt"]:
            self.assertTrue(time > 0)
        self.assertEqual(experiment, results["experiment"])

    def test_results_reproducible(self):
        experiment = RandomSubsetsExperiment(
            dataset="aggregation_score_dataset",
            subset_size=2,
            scorers=[("wilks", CorrelationScore())],
            classifiers=[("dt", DecisionTreeClassifier())],
            n_folds=5,
            n_runs=10
        )
        results = experiment.run(directory="test_files/")
        results2 = results["experiment"].run(directory="test_files/")

        numpy.testing.assert_array_equal(results["scores"]["wilks"], results2["scores"]["wilks"])
        numpy.testing.assert_array_equal(results["errors"]["dt"], results2["errors"]["dt"])

        dump = pickle.dumps(results)
        results2 = pickle.loads(dump)["experiment"].run(directory="test_files/")

        numpy.testing.assert_array_equal(results["scores"]["wilks"], results2["scores"]["wilks"])
        numpy.testing.assert_array_equal(results["errors"]["dt"], results2["errors"]["dt"])
