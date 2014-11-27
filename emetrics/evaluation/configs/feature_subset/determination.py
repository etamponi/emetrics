from emetrics.coefficients.determination_coefficient import DeterminationCoefficient
from emetrics.correlation_score import CorrelationScore
from emetrics.label_encoders.ordinal_label_encoder import OrdinalLabelEncoder


__author__ = 'Emanuele Tamponi'

dump_prefix = "determination"

score = CorrelationScore(
    coefficient=DeterminationCoefficient(),
    label_encoder=OrdinalLabelEncoder()
)

classifiers = {
}

subset_sizes = range(6, 25, 2)

n_runs = 10
n_folds = 10
