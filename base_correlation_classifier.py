import numpy
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics.metrics import accuracy_score
from sklearn.tree.tree import DecisionTreeClassifier

from eole.analysis.dataset_utils import ArffLoader
from emetrics.correlation_score import CorrelationScore
from emetrics.coefficients.association_measure import AssociationMeasure
from emetrics.label_encoders.ordinal_label_encoder import OrdinalLabelEncoder


__author__ = 'Emanuele Tamponi'

gaussian_noise = lambda shape: numpy.random.randn(*shape)


# X, y = make_classification(
#     n_samples=200,
#     n_features=10,
#     n_redundant=2,
#     n_informative=8,
#     n_classes=2,
#     n_clusters_per_class=2
# )

X, y = ArffLoader("datasets/ionosphere.arff").get_dataset()
classes, y = numpy.unique(y, return_inverse=True)

measure = CorrelationScore(
    coefficient=AssociationMeasure(noise_level=1e-4, noise_generator=gaussian_noise),
    label_encoder=OrdinalLabelEncoder()
)

classes = numpy.unique(y)
# X += 1e-10 * (2*numpy.random.rand(*X.shape) - 1)
curr_accuracies = numpy.zeros(10)
curr_knn = numpy.zeros(10)
for fold, (train_indices, test_indices) in enumerate(StratifiedKFold(y, n_folds=10)):
    numpy.random.seed(fold)
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    n_repeat = 100
    n_experts = 1
    y_preds = numpy.zeros((len(y_test), n_experts), dtype=int)
    for e in range(n_experts):
        indices = numpy.random.choice(len(X_train), size=int(15.0*len(X_train)), replace=True)
        X_train_exp, y_train_exp = X_train[indices], y_train[indices]
        for k, x in enumerate(X_test):
            X_matrix = numpy.tile([x], (n_repeat, 1))
            X_total = numpy.vstack((X_train_exp, X_matrix))
            y_total = numpy.hstack((y_train_exp, numpy.zeros(n_repeat)))
            value = 0
            for p in classes:
                y_total[-n_repeat:] = p
                curr_value = measure(X_total, y_total)
                if curr_value == 0 or curr_value == 1:
                    print "OPS!"
                if curr_value > value:
                    y_preds[k, e] = p
                    value = curr_value
    y_pred = numpy.asarray([numpy.bincount(row).argmax() for row in y_preds])
    curr_accuracies[fold] = accuracy_score(y_test, y_pred)
    print curr_accuracies[fold],
    numpy.random.seed(fold)
    curr_knn[fold] = DecisionTreeClassifier().fit(X_train, y_train).score(X_test, y_test)
    print curr_knn[fold]

print curr_accuracies.mean()
print curr_knn.mean()
