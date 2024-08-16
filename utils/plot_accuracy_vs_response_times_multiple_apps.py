import os
import sys
import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import warnings
warnings.filterwarnings("ignore")

pcolors = ['#000080', '#008000', '#990000', '#a5669f',  '#db850d',  '#00112d']
markers = ['s', 'o', 'x', '^', 'v', '*', 'p', 'h']
linestyle = ['-', '.', '--']

dir_path = os.path.dirname(os.path.realpath(__file__))
gs_font = fm.FontProperties(fname=dir_path + '/gillsans.ttf', size=20, weight='bold')
light_grey=(0.5,0.5,0.5)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

results_directory = sys.argv[1]
test_name_suffix = sys.argv[2]
output_file_name = sys.argv[3]

def plot_lines(xs, ys, labels, xlabel, ylabel, outfile):
    fig, ax = plt.subplots()
    nlines = len(xs)
    assert (len(ys) == nlines and len(labels) == nlines)
    for i in range(nlines):
        pcolor = pcolors[i]
        ax.plot(xs[i], ys[i], '-', color=pcolor,  lw=2.5,  marker=markers[i], mew = 1.5, fillstyle="full", markersize = 9, markeredgecolor=pcolor, dash_capstyle='round', label=labels[i], zorder=10, clip_on=False)

    label_fontsize=20
    ax.set_xlabel(xlabel, fontproperties=gs_font, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontproperties=gs_font, fontsize=label_fontsize)
    xmax=1.05*max([max(x) for x in xs])
    xmax, ymax = 200, 60
    plt.xlim(xmax=100.0)
    plt.ylim(ymin=0, ymax=100.0)
    ax.grid(linestyle=':', linewidth=1, color='grey')
    ticklabelcolor = 'black'
    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = zip(*[ (handles[i], labels[i]) for i in sorted(range(len(handles)), key=lambda k: list(labels)[k])])
    order = [2, 1, 3, 4, 0]
    leg = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(0.1, 0.55), borderaxespad=0, loc=2, numpoints=2, handlelength=2, prop=gs_font, fontsize=label_fontsize)
    leg.get_frame().set_linewidth(0.0)
    plt.tick_params(labelsize=label_fontsize)
    axcolor='black'
    ax.xaxis.set_tick_params(width=2, length=10)
    ax.yaxis.set_tick_params(width=2, length=15)
    ax.xaxis.set_tick_params(which='both', colors=axcolor)
    ax.yaxis.set_tick_params(which='both', colors=axcolor)
    ax.spines['bottom'].set_color(axcolor)
    ax.spines['top'].set_color(axcolor)
    ax.spines['right'].set_color(axcolor)
    ax.spines['left'].set_color(axcolor)
    ticklabelcolor = 'black'
    ax.tick_params(axis='x', colors=ticklabelcolor, direction="in")
    ax.tick_params(axis='y', colors=ticklabelcolor, direction="in")

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color(light_grey)
    ax.spines['top'].set_linestyle(':')
    ax.spines['right'].set_linestyle(':')

    for label in ax.get_xticklabels() :
        label.set_fontproperties(gs_font)
    for label in ax.get_yticklabels() :
        label.set_fontproperties(gs_font)
    plt.tight_layout()
    plt.savefig(outfile)

xs = []
ys = []
methods = ["MaxScoreBatchSubsetWithSkipsTopK", "MaxScoreBatchSubsetWithSkips", "WAP5", "DeepFlow", "FCFS"]
loads = [25, 50, 75, 100, 125, 150]
apps = ["hotel", "media", "node"]

method_accuracy = {}
for j in range(len(loads)):
    p = []
    for k in range(len(apps)):
        with open(results_directory + "bin_acc_" + str(loads[j]) + "_" + apps[k] + "_" + test_name_suffix + "_0.0.pickle", 'rb') as afile:
            accuracy_percentile_bins = pickle.load(afile)
            for method, acc in accuracy_percentile_bins.items():
                if method not in method_accuracy:
                    method_accuracy[method] = {}
                    if method == "MaxScoreBatchSubsetWithSkips":
                        method_accuracy["MaxScoreBatchSubsetWithSkipsTopK"] = {}
                x = []
                y = []
                for p, a, t in acc:
                    if p not in method_accuracy[method]:
                        method_accuracy[method][p] = []
                    method_accuracy[method][p].append(a * 100)

xs = []
ys = []
for method in methods:
    x = []
    y = []
    for p in method_accuracy[method].keys():
        j = method_accuracy[method][p]
        x.append(p)
        y.append(np.mean(j))
    xs.append(x)
    ys.append(y)

methods = ["TraceWeaver (Top K)", "TraceWeaver", "WAP5", "vPath", "FCFS"]
plot_lines(xs, ys, methods, "Latency Percentile Bins", "Accuracy % (avg. across apps)", output_file_name)

