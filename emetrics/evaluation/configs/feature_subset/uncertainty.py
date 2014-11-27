import numpy
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm.classes import SVC
from sklearn.tree.tree import DecisionTreeClassifier

from emetrics.coefficients.association_measure import AssociationMeasure
from emetrics.coefficients.uncertainty_coefficient import UncertaintyCoefficient
from emetrics.correlation_score import CorrelationScore
from emetrics.label_encoders.ordinal_label_encoder import OrdinalLabelEncoder


__author__ = 'Emanuele Tamponi'

dump_prefix = "uncertainty_gaussian_4"

score = CorrelationScore(
    coefficient=UncertaintyCoefficient(noise_level=1e-4),
    label_encoder=OrdinalLabelEncoder()
)

classifiers = {
}

subset_sizes = range(6, 25, 2)

n_runs = 10
n_folds = 10
