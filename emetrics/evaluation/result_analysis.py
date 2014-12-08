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


SCATTER_COLORS = ["b", "r", "g", "k", "m"]
SCATTER_MARKERS = ["o", "^", "s", "x", "D"]
LINE_STYLES = ["b--", "r-.", "g:", "k-.", "m--"]
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
    # plot_correlation(all_results, pyplot)


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


def prepare_results(raw_results):
    scorers = sorted(raw_results["scores"].keys())
    classifiers = sorted(raw_results["errors"].keys())

    results = {}
    for scorer, classifier in product(scorers, classifiers):
        scores = raw_results["scores"][scorer]
        errors = raw_results["errors"][classifier]
        corr, p = pearsonr(scores, errors)
        if isnan(corr):
            corr = 0.0
        x = numpy.sort(scores)
        y = errors[scores.argsort()]
        x_time = raw_results["score_times"][scorer].mean()
        y_time = raw_results["classifier_times"][classifier].mean()
        results[(scorer, classifier)] = {
            "x": x,
            "y": y,
            "corr": corr,
            "p": p,
            "x_time": x_time,
            "y_time": y_time
        }
    return results


def get_figure_file_name(results_file_name, scorer):
    return "{}_{}".format(scorer, results_file_name)


def plot_correlation_to_range(all_results, pyplot):
    n_bins = 5
    width = 0.6/n_bins

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

        print "{}: {:.0f} significant ({:.0f} total)".format(figure_name, significant.sum(), valid.sum())

        # Plot format
        x_ticks_pos = numpy.asarray([0.5*(bins[i] + bins[i+1]) for i in range(len(bins)-1)])
        left = x_ticks_pos - width/2
        pyplot.bar(left, 100 * (significant / valid), width=width)

        pyplot.ylim(0, 105)
        pyplot.yticks(range(10, 101, 10), ["{}%".format(percent) for percent in range(10, 101, 10)])
        pyplot.ylabel("Significant rankings".format(LEGEND[classifier]))

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
        pyplot.close()


def plot_correlation(all_results, pyplot):
    datasets = dataset_names()
    normalization = [True]
    subset_sizes = [3, 5, 10, 15, 20]

    for dataset, normalize, subset_size in product(datasets, normalization, subset_sizes):
        results = all_results.get((dataset, normalize, subset_size), None)
        if results is None:
            continue
        scorers = ["deter", "wilks", "uncer"]
        classifiers = ["ab", "gb", "ba", "rf", "et"]
        for i, scorer in enumerate(scorers):
            for j, classifier in enumerate(classifiers):
                data = results[(scorer, classifier)]
                x = data["x"]
                y = data["y"]
                if x.min() <= 0 or x.max() >= 1:
                    continue
                if data["p"] > 0.02 or data["corr"] > 0:
                    if numpy.random.choice([True, False], p=[0.95, 0.05]):
                        continue
                    else:
                        wrong = True
                else:
                    wrong = False
                pyplot.scatter(x, y, color=SCATTER_COLORS[j], marker=SCATTER_MARKERS[j], s=20)
                # noinspection PyTupleAssignmentBalance
                m, q = numpy.polyfit(x, y, deg=1)
                x_min, x_max = pyplot.xlim()
                pyplot.xlim(max(0, x_min), min(1, x_max))
                pyplot.plot([x_min, x_max], [q + m*x_min, q + m*x_max], LINE_STYLES[j], linewidth=1.5)
                # Format
                pyplot.ylim(pyplot.ylim()[0], pyplot.ylim()[1])
                y_range = pyplot.ylim()[1] - pyplot.ylim()[0]
                x_range = pyplot.xlim()[1] - pyplot.xlim()[0]
                pyplot.axes().set_aspect(x_range / y_range)
                pyplot.grid()
                pyplot.xticks(numpy.linspace(*pyplot.xlim(), num=6))
                pyplot.yticks(numpy.linspace(*pyplot.ylim(), num=6))
                pyplot.xlabel(LEGEND[scorer])
                pyplot.ylabel("{} error rate".format(LEGEND[classifier]))
                pyplot.title("{}, {} features - correlation: ${:.2f}$".format(
                    dataset, subset_size,
                    results[(scorer, classifier)]["corr"])
                )
                pyplot.savefig("figures/{}plot_{}_{}_{}.pdf".format(
                    "wrong" if wrong else "",
                    scorer, classifier, get_results_file_name(dataset, normalize, subset_size)), bbox_inches="tight"
                )
                pyplot.close()


if __name__ == "__main__":
    main()
