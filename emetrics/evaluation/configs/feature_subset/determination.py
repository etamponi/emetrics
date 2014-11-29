from emetrics.coefficients.determination_coefficient import DeterminationCoefficient
from emetrics.correlation_score import CorrelationScore
from emetrics.label_encoders.onehot_label_encoder import OneHotLabelEncoder


__author__ = 'Emanuele Tamponi'

dump_prefix = "determination_1_25"

score = CorrelationScore(
    coefficient=DeterminationCoefficient(),
    label_encoder=OneHotLabelEncoder()
)

classifiers = {
}

subset_sizes = range(1, 26, 1)

n_runs = 10
n_folds = 10
