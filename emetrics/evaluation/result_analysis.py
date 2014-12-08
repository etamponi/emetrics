from collections import defaultdict
from itertools import product
import cPickle
import os
from math import isnan

import matplotlib
import numpy
from scipy.stats.stats import pearsonr

from emetrics.evaluation import dataset_names
from emetrics.evaluation.all_experiments import get_results_file_name


__author__ = 'Emanuele Tamponi'


STYLES = ["bo--", "r^-.", "gs:", "cp-.", "mD--"]
LEGEND = {
    "ab": "AdaBoost",
    "gb": "Gradient Boosting",
    "ba": "Bagging",
    "rf": "Random Forest",
    "et": "Ext. Rand. Trees",
    "wilks": "Wilks' $\eta^2$",
    "deter": "$R^2$",
    "uncer": "Uncertainty"
}


def main():
    configure_matplotlib()
    from matplotlib import pyplot

    datasets = dataset_names()
    subset_sizes = range(1, 11) + range(15, 36, 5)

    all_results = {}

    for dataset, normalize, subset_size in product(datasets, [True, False], subset_sizes):
        results_file_name = get_results_file_name(dataset, normalize, subset_size)
        if not os.path.isfile("results/{}.res".format(results_file_name)):
            continue
        with open("results/{}.res".format(results_file_name)) as f:
            results = cPickle.load(f)
        all_results[(dataset, normalize, subset_size)] = prepare_results(results)

    plot_correlation_to_range(all_results, pyplot)


def configure_matplotlib():
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


def prepare_results(results):
    scorers = sorted(results["scores"].keys())
    classifiers = sorted(results["errors"].keys())

    output = {}
    for scorer, classifier in product(scorers, classifiers):
        scores = results["scores"][scorer]
        errors = results["errors"][classifier]
        corr, p = pearsonr(scores, errors)
        if isnan(corr):
            corr = 0.0
        x = numpy.sort(scores)
        y = errors[scores.argsort()]
        x_time = results["score_times"][scorer].mean()
        y_time = results["classifier_times"][classifier].mean()
        output[(scorer, classifier)] = {
            "x": x,
            "y": y,
            "corr": corr,
            "p": p,
            "x_time": x_time,
            "y_time": y_time
        }
    return output
        #     pyplot.plot(
        #         sorted(scores),
        #         errors[scores.argsort()],
        #         STYLES[i % len(STYLES)],
        #         linewidth=1.5,
        #         markersize=6
        #     )
        # # Plot style
        # x_range = pyplot.xlim()[1] - pyplot.xlim()[0]
        # y_range = pyplot.ylim()[1] - pyplot.ylim()[0]
        # pyplot.ylim(pyplot.ylim()[0], pyplot.ylim()[0]+1.5*y_range)
        # y_range *= 1.5
        # pyplot.axes().set_aspect(x_range / y_range)
        # pyplot.grid()
        # pyplot.xticks(numpy.linspace(*pyplot.xlim(), num=6))
        # pyplot.yticks(numpy.linspace(*pyplot.ylim(), num=6))
        # pyplot.legend([LEGEND[c] for c in classifiers])
        # pyplot.xlabel("$\eta^2_\Lambda$")
        # pyplot.ylabel("error")
        #
        # pyplot.savefig("figures/{}.pdf".format(figure_file_name), bbox_inches="tight")
        # pyplot.close()


def get_figure_file_name(results_file_name, scorer):
    return "{}_{}".format(scorer, results_file_name)


def plot_correlation_to_range(all_results, pyplot):
    n_bins = 5
    width = 0.8/5

    range_to_corr = defaultdict(lambda: defaultdict(list))
    for (dataset, normalize, subset_size), results in all_results.iteritems():
        for (scorer, classifier), comparison in results.iteritems():
            x = comparison["x"]
            if x.min() <= 0 or x.max() >= 1:
                print "Problems with {} {}".format(get_results_file_name(dataset, normalize, subset_size), scorer)
                continue
            x_range = x.max() - x.min()
            if comparison["p"] <= 0.05 and comparison["corr"] < 0:
                range_to_corr[(scorer, classifier, normalize)]["significant"].append(x_range)
            range_to_corr[(scorer, classifier, normalize)]["valid"].append(x_range)

    for (scorer, classifier, normalize), data in range_to_corr.iteritems():
        normalize_format = "norm" if normalize else "orig"
        figure_name = "corr_to_range_{}_{}_{}".format(scorer, classifier, normalize_format)
        significant, bins, _ = pyplot.hist(data["significant"], bins=n_bins, range=(0, 1))
        valid, bins, _ = pyplot.hist(data["valid"], bins=n_bins, range=(0, 1))
        pyplot.close()

        print "{}: {} significant ({} total)".format(figure_name, significant.sum(), valid.sum())

        x_ticks_pos = numpy.asarray([0.5*(bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        left = x_ticks_pos - width/2
        pyplot.bar(left, 100 * (significant / valid), width=width)

        pyplot.ylim(0, 105)
        pyplot.yticks(numpy.linspace(10, 100, 10))
        pyplot.ylabel("Significant rankings (%)".format(LEGEND[classifier]))

        x_ticks = ["{:.1f} to {:.1f}".format(bins[i], bins[i+1]) for i in range(len(bins)-1)]
        pyplot.xlim(0, 1)
        pyplot.xticks(x_ticks_pos, x_ticks)
        pyplot.grid()
        pyplot.xlabel("Range of {}".format(LEGEND[scorer]))

        pyplot.title("{} vs. {} on {} data: {:.0f} significant ({:.0f} total)".format(
            LEGEND[scorer], LEGEND[classifier],
            "normalized" if normalize else "original",
            significant.sum(), valid.sum())
        )

        pyplot.savefig("figures/{}.pdf".format(figure_name))


if __name__ == "__main__":
    main()
