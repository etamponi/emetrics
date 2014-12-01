from itertools import product
import cPickle
import os

import numpy
from scipy.stats.stats import pearsonr

from emetrics.evaluation import dataset_names
from emetrics.evaluation.all_experiments import get_results_file_name


__author__ = 'Emanuele Tamponi'


def main():
    import matplotlib
    matplotlib.use('pgf')
    pgf_rc = {
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,     # use inline math for ticks
        "pgf.rcfonts": False,    # don't setup fonts from rc parameters
        "pgf.texsystem": "pdflatex",
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
    matplotlib.rcParams.update(pgf_rc)

    from matplotlib import pyplot
    styles = ["bo--", "r^-.", "gs:"]
    names_map = {"bo": "AdaBoost", "rf": "Random Forest", "sv": "SVM"}

    datasets = dataset_names()
    subset_sizes = range(1, 11) + range(15, 36, 5)
    for dataset, normalize, subset_size in product(datasets, [True, False], subset_sizes):
        results_file_name = get_results_file_name(dataset, normalize, subset_size)
        if not os.path.isfile("results/{}.res".format(results_file_name)):
            continue
        with open("results/{}.res".format(results_file_name)) as f:
            results = cPickle.load(f)
        scorers = sorted(results["scores"].keys())
        classifiers = sorted(results["errors"].keys())
        for scorer in scorers:
            figure_file_name = get_figure_file_name(results_file_name, scorer)
            if os.path.isfile("figures/{}.pdf".format(figure_file_name)):
                continue
            for i, classifier in enumerate(classifiers):
                scores = results["scores"][scorer]
                errors = results["errors"][classifier]
                corr, p = pearsonr(scores, errors)
                print "{:>15s} ({}) - {:2d} features - {} vs {} = {:.3f} {}".format(
                    dataset,
                    "norm" if normalize else "orig",
                    subset_size,
                    scorer,
                    classifier,
                    corr,
                    "ok" if p <= 0.05 and corr < 0 else ""
                )
                pyplot.plot(
                    sorted(scores),
                    errors[scores.argsort()],
                    styles[i % len(styles)],
                    linewidth=1.5,
                    markersize=6
                )
            # Plot style
            x_range = pyplot.xlim()[1] - pyplot.xlim()[0]
            y_range = pyplot.ylim()[1] - pyplot.ylim()[0]
            pyplot.axes().set_aspect(x_range / y_range)
            pyplot.grid()
            pyplot.xticks(numpy.linspace(*pyplot.xlim(), num=6))
            pyplot.yticks(numpy.linspace(*pyplot.ylim(), num=6))
            pyplot.legend([names_map[c] for c in classifiers])
            pyplot.xlabel("$\eta^2_\Lambda$")
            pyplot.ylabel("error")

            pyplot.savefig("figures/{}.pdf".format(figure_file_name), bbox_inches="tight")
            pyplot.close()


def get_figure_file_name(results_file_name, scorer):
    return "{}_{}".format(scorer, results_file_name)


if __name__ == "__main__":
    main()
