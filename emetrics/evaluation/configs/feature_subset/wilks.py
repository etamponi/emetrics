import numpy
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm.classes import SVC
from sklearn.tree.tree import DecisionTreeClassifier

from emetrics.coefficients.association_measure import AssociationMeasure
from emetrics.correlation_score import CorrelationScore
from emetrics.label_encoders.ordinal_label_encoder import OrdinalLabelEncoder


__author__ = 'Emanuele Tamponi'

dump_prefix = "wilks_gaussian_4"

gaussian_noise = lambda shape: numpy.random.randn(*shape)
uniform_noise = lambda shape: numpy.random.uniform(-1.0, 1.0, size=shape)

score = CorrelationScore(
    coefficient=AssociationMeasure(
        measure="wilks",
        noise_level=1e-4,
        noise_generator=gaussian_noise
    ),
    label_encoder=OrdinalLabelEncoder()
)

classifiers = {
    "dt": DecisionTreeClassifier(),
    "rf100": RandomForestClassifier(n_estimators=100),
    "svc": SVC()
}

subset_sizes = range(6, 25, 2)

n_runs = 10
n_folds = 10
