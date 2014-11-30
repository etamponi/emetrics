import cPickle
from matplotlib import pyplot
from scipy.stats.stats import pearsonr
from emetrics.evaluation import dataset_names

__author__ = 'Emanuele Tamponi'


errors_prefix = "wilks_gaussian_4_1_25"
cls_names = {
    "dt": "C4.5",
    "rf100": "Random Forest",
    "svc": "Gaussian SVM"
}


def main():
    scores_prefix = "wilks_gaussian_4_1_25"
    score_name = "\eta^2_\Lambda"
    score_name_file = "wilks"
    count = 0
    count_significant = 0
    dataset_completed = 0
    dataset_passed = 0
    datasets = dataset_names()[:1]
    for dataset in datasets:
        significants = 0
        total = 0
        print "Dataset", dataset
        scores_result_file = "results/{}_{}.res".format(scores_prefix, dataset)
        errors_result_file = "results/{}_{}.res".format(errors_prefix, dataset)
        try:
            with open(scores_result_file) as f:
                scores_data = cPickle.load(f)
            with open(errors_result_file) as f:
                errors_data = cPickle.load(f)
        except IOError:
            continue
        subset_sizes = sorted(scores_data.keys())
        for subset_size in subset_sizes:
            scores = scores_data[subset_size]["scores"]
            classifiers = sorted(errors_data[subset_size]["errors"].keys())
            f, axarr = pyplot.subplots(1, 3, sharey=True)
            for i, classifier in enumerate(classifiers):
                errors = errors_data[subset_size]["errors"][classifier]
                corr, p = pearsonr(scores, errors)
                print "{:2d} features - correlation with {:5s} = {:.3f}".format(subset_size, classifier, corr)
                if p <= 0.05 and corr < 0:
                    count_significant += 1
                    significants += 1
                count += 1
                total += 1
                axarr[i].plot(sorted(errors), scores[errors.argsort()])
                axarr[i].set_title("{} features - {} vs. {}".format(subset_size, score_name, cls_names[classifier]))
                # pyplot.savefig("figures/{}_{:02d}_{}_{}.pdf".format(dataset, subset_size, score_name_file, classifier))
        if significants == total:
            dataset_completed += 1
        if total == 0 or significants > 1:
            dataset_passed += 1
    print "{} significant out of {}".format(count_significant, count)
    print "{} dataset completed out of {}".format(dataset_completed, len(datasets))
    print "{} dataset passed out of {}".format(dataset_passed, len(datasets))


if __name__ == "__main__":
    main()

