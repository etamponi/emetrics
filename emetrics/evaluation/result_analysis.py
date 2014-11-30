import cPickle

from scipy.stats.stats import pearsonr


__author__ = 'Emanuele Tamponi'


errors_prefix = "wilks_gaussian_4_1_25"
cls_names = {
    "dt": "C4.5",
    "rf100": "RF",
    "svc": "SVM"
}


def main():
    import matplotlib
    matplotlib.use('pgf')
    from matplotlib import pyplot
    pgf_with_custom_preamble = {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,     # use inline math for ticks
        "pgf.rcfonts": False,    # don't setup fonts from rc parameters
        "pgf.preamble": [
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{microtype}",
            r"\usepackage{amsfonts}",
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",
            r"\usepackage{booktabs}",
            r"\usepackage{fancyhdr}",
            r"\usepackage{graphicx}",
            r"\usepackage{nicefrac}",
            r"\usepackage{xspace}"
        ]
    }
    matplotlib.rcParams.update(pgf_with_custom_preamble)

    results_dir = "results_old_features"
    scores_prefix = "wilks_gaussian_4_1_25"
    score_name = "$\eta^2_\Lambda$"
    score_name_file = "wilks"
    count = 0
    count_significant = 0
    dataset_completed = 0
    dataset_passed = 0
    datasets = ["colic"]  # dataset_names()
    for dataset in datasets:
        significants = 0
        total = 0
        print "Dataset", dataset
        scores_result_file = "{}/{}_{}.res".format(results_dir, scores_prefix, dataset)
        errors_result_file = "{}/{}_{}.res".format(results_dir, errors_prefix, dataset)
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
            pyplot.title("{} features".format(subset_size))
            pyplot.xlabel(score_name)
            pyplot.ylabel("error")
            for i, classifier in enumerate(classifiers):
                errors = errors_data[subset_size]["errors"][classifier]
                # Correct calculation error in feature_subsets_experiments
                errors = correct_errors(errors, results_dir, errors_data[subset_size]["n_folds"])
                corr, p = pearsonr(scores, errors)
                print "{:2d} features - correlation with {:5s} = {:.3f}".format(subset_size, classifier, corr)
                if p <= 0.05 and corr < 0:
                    count_significant += 1
                    significants += 1
                count += 1
                total += 1
                pyplot.plot(sorted(scores), errors[scores.argsort()], "o-")
            pyplot.grid()
            pyplot.savefig("figures/{}_{:02d}_{}.pdf".format(dataset, subset_size, score_name_file))
            return
        if significants == total:
            dataset_completed += 1
        if total == 0 or significants > 1:
            dataset_passed += 1
    print "{} significant out of {}".format(count_significant, count)
    print "{} dataset completed out of {}".format(dataset_completed, len(datasets))
    print "{} dataset passed out of {}".format(dataset_passed, len(datasets))


def correct_errors(errors, results_dir, n_folds):
    if results_dir == "results":
        return errors
    else:
        return errors - n_folds + 1


if __name__ == "__main__":
    main()

