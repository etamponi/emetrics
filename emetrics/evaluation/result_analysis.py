from __future__ import division

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
SCORERS = ["deter", "wilks", "uncer"]
CLASSIFIERS = ["ab", "gb", "ba", "rf", "et"]
SUBSET_SIZES = range(1, 11) + range(15, 36, 5)
DATASETS = dataset_names()
NL = "\n"


def main():
    configure_matplotlib()
    from matplotlib import pyplot

    all_results = {}

    for dataset, normalize, subset_size in product(DATASETS, [True, False], SUBSET_SIZES):
        results_file_name = get_results_file_name(dataset, normalize, subset_size)
        if not os.path.isfile("results/{}.res".format(results_file_name)):
            continue
        with open("results/{}.res".format(results_file_name)) as f:
            results = cPickle.load(f)
        all_results[(dataset, normalize, subset_size)] = prepare_results(results)

    correlation_to_range_plots(all_results, pyplot)
    correlation_plots(all_results, pyplot)
    correlation_tables(all_results)
    synthesis_per_dataset(all_results)
    synthesis_per_subset_size(all_results)


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
    results = {}
    for scorer, classifier in product(SCORERS, CLASSIFIERS):
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


def correlation_to_range_plots(all_results, pyplot):
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
        pyplot.ylabel("Percentage of significant rankings")

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

        pyplot.savefig("figures/{}.pdf".format(figure_name), bbox_inches="tight")
        pyplot.close()


def correlation_plots(all_results, pyplot):
    numpy.random.seed(1)

    datasets = dataset_names()
    normalization = [True, False]
    subset_sizes = [3, 5, 10, 15, 20]

    for dataset, normalize, subset_size in product(datasets, normalization, subset_sizes):
        results = all_results.get((dataset, normalize, subset_size), None)
        if results is None:
            continue
        for scorer, classifier in product(SCORERS, CLASSIFIERS):
            data = results[(scorer, classifier)]
            draw_single_plot(dataset, normalize, subset_size, scorer, classifier, data, pyplot)


def draw_single_plot(dataset, normalize, subset_size, scorer, classifier, data, pyplot):
    x = data["x"]
    y = data["y"]
    if x.min() <= 0 or x.max() >= 1:
        return
    if data["p"] > 0.02 or data["corr"] > 0:
        if numpy.random.choice([True, False], p=[0.95, 0.05]):
            return
        else:
            wrong = True
    else:
        wrong = False
    pyplot.scatter(x, y, color="k", marker="s", s=20)
    x_min, x_max = pyplot.xlim()
    y_min, y_max = pyplot.ylim()
    # noinspection PyTupleAssignmentBalance
    m, q = numpy.polyfit(x, y, deg=1)
    pyplot.plot([0, 1], [q, q + m], "k--", linewidth=1.5)
    # Format
    pyplot.xlim(max(0, x_min), min(1, x_max))
    pyplot.ylim(max(0, y_min), min(1, y_max))
    y_range = pyplot.ylim()[1] - pyplot.ylim()[0]
    x_range = pyplot.xlim()[1] - pyplot.xlim()[0]
    pyplot.axes().set_aspect(x_range / y_range)
    pyplot.grid()
    pyplot.xticks(numpy.linspace(*pyplot.xlim(), num=6))
    pyplot.yticks(numpy.linspace(*pyplot.ylim(), num=6))
    pyplot.xlabel(LEGEND[scorer])
    pyplot.ylabel("{} error rate".format(LEGEND[classifier]))
    pyplot.title("{}, {} {} - ${:.1f}$% of determination".format(
        dataset, subset_size, "feature" if subset_size == 1 else "features", 100*data["corr"]**2
    ))
    figure_name = "{}plot_{}_{}_{}".format(
        "wrong" if wrong else "",
        scorer, classifier, get_results_file_name(dataset, normalize, subset_size)
    )
    try:
        pyplot.savefig("figures/{}.pdf".format(figure_name), bbox_inches="tight")
    except RuntimeError:
        print "Error processing {}. Continuing".format(figure_name)
    pyplot.close()


def correlation_tables(all_results):
    normalization = [True, False]

    for dataset, normalize, classifier in product(DATASETS, normalization, CLASSIFIERS):
        table_data = []
        for subset_size in SUBSET_SIZES:
            if (dataset, normalize, subset_size) not in all_results:
                break
            table_data.append([])
            for scorer in SCORERS:
                table_data[-1].append((
                    all_results[(dataset, normalize, subset_size)][(scorer, classifier)]["corr"],
                    all_results[(dataset, normalize, subset_size)][(scorer, classifier)]["p"],
                ))
        write_table(dataset, normalize, classifier, table_data)


def write_table(dataset, normalize, classifier, table_data):
    table_name = "table_{}_{}_{}".format(classifier, dataset, "norm" if normalize else "orig")
    with open("figures/{}.tex".format(table_name), "w") as f:
        f.writelines((r"\begin{table}\centering", NL, r"\label{tab:{%s}}" % table_name, NL))
        f.writelines((r"\renewcommand{\arraystretch}{1.2}", NL))
        f.writelines((r"\begin{tabularx}{0.65\textwidth}{>{\small}r *3{Y}}", NL))
        f.writelines((r"\toprule", NL))
        f.writelines((r"Features & {} \\".format(" & ".join(LEGEND[scorer] for scorer in SCORERS)), NL))
        f.writelines((r"\midrule", NL))
        for i, row in enumerate(table_data):
            f.write(r"{}".format(SUBSET_SIZES[i]))
            for corr, p in row:
                f.write(" & ")
                if corr > 0:
                    f.write("n/a")
                    continue
                f.write("$")
                if p <= 0.02:
                    f.write(r"\mathbf{")
                corr = 100 * corr**2
                if p <= 0.05:
                    f.write(r"\bullet \ ")
                f.write("{:.1f}\%".format(corr))
                if p <= 0.02:
                    f.write(r"}")
                f.write("$")
            f.writelines((r"\\", NL))
        f.writelines((r"\bottomrule", NL))
        f.writelines((r"\end{tabularx}", NL))
        caption = r"Dataset {}{}. Correlation with {} error rate.".format(
            dataset, " (normalized)" if normalize else "", LEGEND[classifier]
        )
        f.writelines((r"\caption{%s}" % caption, NL))
        f.writelines((r"\end{table}", NL))


def synthesis_per_dataset(all_results):
    for classifier in CLASSIFIERS:
        table_data = []
        for normalize, dataset in product([True, False], DATASETS):
            table_data.append([])
            for scorer in SCORERS:
                significant = 0
                total = 0
                for subset_size in SUBSET_SIZES:
                    if (dataset, normalize, subset_size) not in all_results:
                        continue
                    x = all_results[(dataset, normalize, subset_size)][(scorer, classifier)]["x"]
                    p = all_results[(dataset, normalize, subset_size)][(scorer, classifier)]["p"]
                    corr = all_results[(dataset, normalize, subset_size)][(scorer, classifier)]["corr"]
                    if p <= 0.05 and corr < 0 and (x.min() >= 0 and x.max() <= 1):
                        significant += 1
                    total += 1
                table_data[-1].append((significant, total))
        write_table_synthesis_per_dataset(classifier, table_data)


def write_table_synthesis_per_dataset(classifier, table_data):
    table_name = "synthesis_per_dataset_{}".format(classifier)
    with open("figures/{}.tex".format(table_name), "w") as f:
        f.writelines((r"\begin{table}\centering", NL, r"\label{tab:{%s}}" % table_name, NL))
        f.writelines((r"\renewcommand{\arraystretch}{1.10}", NL))
        # TODO: number of columns depend on number of scorers
        f.writelines((r"\begin{tabularx}{0.85\textwidth}{>{\small}XZZcZZcZZ}", NL))
        f.writelines((r"\toprule", NL))
        f.writelines((r"Dataset (runs) & {} \\".format(" & \phantom{a} & ".join(
            ("\multicolumn{2}{c}{%s}" % LEGEND[scorer]) for scorer in SCORERS
        )), NL))
        f.writelines((r"\midrule", NL))
        f.writelines((r" & ", " & & ".join(len(SCORERS)*[r"\tiny orig & \tiny norm"]), r"\\", NL))
        # TODO: number of columns depend on number of scorers
        f.writelines((r"\cmidrule{2-3} \cmidrule{5-6} \cmidrule{8-9}", NL))
        normalized_data = table_data[:len(DATASETS)]
        original_data = table_data[len(DATASETS):]
        total_total = 0
        total_sign = 2 * len(SCORERS) * [0]
        for i, (normalized_row, original_row) in enumerate(zip(normalized_data, original_data)):
            total = normalized_row[0][1]
            total_total += total
            f.write(r"{} ({})".format(DATASETS[i], total))
            for j, ((norm_sign, _), (orig_sign, _)) in enumerate(zip(normalized_row, original_row)):
                f.write(" & ")
                for k, significant in enumerate([orig_sign, norm_sign]):
                    if significant / total > 0.60:
                        f.write(r"$\mathbf{%d}$" % significant)
                    else:
                        f.write(r"$%d$" % significant)
                    if k == 0:
                        f.write(r" & ")
                    total_sign[2*j + k] += significant
                if j < 2:
                    f.write(r" & ")
            f.writelines((r"\\", NL))
        # TODO: number of columns depend on number of scorers
        f.writelines((r"\cmidrule{2-3} \cmidrule{5-6} \cmidrule{8-9}", NL))
        f.writelines((
            r"\bfseries Total ({}) & {} & {} & & {} & {} & & {} & {} \\".format(total_total, *total_sign),
            NL
        ))
        f.writelines((r"\bottomrule", NL))
        f.writelines((r"\end{tabularx}", NL))
        f.writelines((r"\captionsetup{justification=raggedright,singlelinecheck=false}", NL))
        caption = r"Overall significant results per dataset when compared to {} error rate.".format(
            LEGEND[classifier]
        )
        f.writelines((r"\caption{%s}" % caption, NL))
        f.writelines((r"\end{table}", NL))


def synthesis_per_subset_size(all_results):
    for classifier in CLASSIFIERS:
        table_data = []
        for normalize, subset_size in product([True, False], SUBSET_SIZES):
            table_data.append([])
            for scorer in SCORERS:
                significant = 0
                total = 0
                for dataset in DATASETS:
                    if (dataset, normalize, subset_size) not in all_results:
                        continue
                    x = all_results[(dataset, normalize, subset_size)][(scorer, classifier)]["x"]
                    p = all_results[(dataset, normalize, subset_size)][(scorer, classifier)]["p"]
                    corr = all_results[(dataset, normalize, subset_size)][(scorer, classifier)]["corr"]
                    if p <= 0.05 and corr < 0 and (x.min() >= 0 and x.max() <= 1):
                        significant += 1
                    total += 1
                table_data[-1].append((significant, total))
        write_table_synthesis_per_subset_size(classifier, table_data)


def write_table_synthesis_per_subset_size(classifier, table_data):
    table_name = "synthesis_per_subset_size_{}".format(classifier)
    with open("figures/{}.tex".format(table_name), "w") as f:
        f.writelines((r"\begin{table}\centering", NL, r"\label{tab:{%s}}" % table_name, NL))
        f.writelines((r"\renewcommand{\arraystretch}{1.20}", NL))
        # TODO: number of columns depend on number of scorers
        f.writelines((r"\begin{tabularx}{0.85\textwidth}{>{\small}XZZcZZcZZ}", NL))
        f.writelines((r"\toprule", NL))
        f.writelines((r"Subset size (runs) & {} \\".format(" & \phantom{a} & ".join(
            ("\multicolumn{2}{c}{%s}" % LEGEND[scorer]) for scorer in SCORERS
        )), NL))
        f.writelines((r"\midrule", NL))
        f.writelines((r" & ", " & & ".join(len(SCORERS)*[r"\tiny orig & \tiny norm"]), r"\\", NL))
        # TODO: number of columns depend on number of scorers
        f.writelines((r"\cmidrule{2-3} \cmidrule{5-6} \cmidrule{8-9}", NL))
        normalized_data = table_data[:len(SUBSET_SIZES)]
        original_data = table_data[len(SUBSET_SIZES):]
        total_total = 0
        total_sign = 2 * len(SCORERS) * [0]
        for i, (normalized_row, original_row) in enumerate(zip(normalized_data, original_data)):
            total = normalized_row[0][1]
            total_total += total
            f.write(r"{} {} ({})".format(
                SUBSET_SIZES[i], "feature" if SUBSET_SIZES[i] == 1 else "features", total
            ))
            for j, ((norm_sign, _), (orig_sign, _)) in enumerate(zip(normalized_row, original_row)):
                f.write(" & ")
                for k, significant in enumerate([orig_sign, norm_sign]):
                    if significant / total > 0.50:
                        f.write(r"$\mathbf{%d}$" % significant)
                    else:
                        f.write(r"$%d$" % significant)
                    if k == 0:
                        f.write(r" & ")
                    total_sign[2*j + k] += significant
                if j < 2:
                    f.write(r" & ")
            f.writelines((r"\\", NL))
        # TODO: number of columns depend on number of scorers
        f.writelines((r"\cmidrule{2-3} \cmidrule{5-6} \cmidrule{8-9}", NL))
        f.writelines((
            r"\bfseries Total ({}) & {} & {} & & {} & {} & & {} & {} \\".format(total_total, *total_sign),
            NL
        ))
        f.writelines((r"\bottomrule", NL))
        f.writelines((r"\end{tabularx}", NL))
        f.writelines((r"\captionsetup{justification=raggedright,singlelinecheck=false}", NL))
        caption = r"Overall significant results per feature subset size when compared to {} error rate.".format(
            LEGEND[classifier]
        )
        f.writelines((r"\caption{%s}" % caption, NL))
        f.writelines((r"\end{table}", NL))


if __name__ == "__main__":
    main()
