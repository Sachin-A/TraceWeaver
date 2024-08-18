import os
import pickle
import sys

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

pcolors = ['#0e326d', '#028413', '#a5669f', '#db850d', '#00112d', '#af0505']
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

def plot_box(xs, ys, labels, xlabel, ylabel, outfile):
    ticks = xs[0]
    label_fontsize=20

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    fig, ax = plt.subplots()

    bp1 = plt.boxplot(ys[0], positions=np.array(range(len(ys[0]))) * len(ticks) + 1.6, sym='', widths=0.75)
    bp2 = plt.boxplot(ys[1], positions=np.array(range(len(ys[1]))) * len(ticks) + 0.8, sym='', widths=0.75)
    bp3 = plt.boxplot(ys[2], positions=np.array(range(len(ys[2]))) * len(ticks) + 0.0, sym='', widths=0.75)
    bp4 = plt.boxplot(ys[3], positions=np.array(range(len(ys[3]))) * len(ticks) - 0.8, sym='', widths=0.75)
    bp5 = plt.boxplot(ys[4], positions=np.array(range(len(ys[4]))) * len(ticks) - 1.6, sym='', widths=0.75)
    set_box_color(bp1, pcolors[0])
    set_box_color(bp2, pcolors[1])
    set_box_color(bp3, pcolors[2])
    set_box_color(bp4, pcolors[3])
    set_box_color(bp5, pcolors[4])

    plt.plot([], c = pcolors[0], label = str(labels[0]))
    plt.plot([], c = pcolors[1], label = str(labels[1]))
    plt.plot([], c = pcolors[2], label = str(labels[2]))
    plt.plot([], c = pcolors[3], label = str(labels[3]))
    plt.plot([], c = pcolors[4], label = str(labels[4]))

    ax.set_xlabel(xlabel, fontproperties=gs_font, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontproperties=gs_font, fontsize=label_fontsize)
    plt.xticks([i for i in range(0, len(ys[0]) * len(ticks), len(ticks))], ticks)
    plt.ylim(0, 101)
    plt.tight_layout()
    plt.grid(linestyle=':', linewidth=1, color='grey')
    ticklabelcolor = 'black'
    leg = ax.legend(bbox_to_anchor=(0.0, 0.55), borderaxespad=0, loc=2, numpoints=2, handlelength=2, prop=gs_font, fontsize=label_fontsize)
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
    plt.savefig(output_file_name)

def plot_lines(xs, ys, labels, xlabel, ylabel, outfile):
    fig, ax = plt.subplots()
    nlines = len(xs)
    assert (len(ys) == nlines and len(labels) == nlines)
    for i in range(nlines):
        pcolor = pcolors[i]
        ax.plot(xs[i], ys[i], '-', color=pcolor,  lw=2.5,  marker=markers[i], mew = 1.5, markersize = 10, markerfacecolor='none', markeredgecolor=pcolor, dash_capstyle='round', label=labels[i])

    label_fontsize=20
    ax.set_xlabel(xlabel, fontproperties=gs_font, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontproperties=gs_font, fontsize=label_fontsize)
    xmax=1.05*max([max(x) for x in xs])
    plt.xlim(xmax=xs[0][-1] + 10)
    plt.ylim(ymin=0, ymax=1.05)
    ax.grid(linestyle=':', linewidth=1, color='grey')
    ticklabelcolor = 'black'
    leg = ax.legend(bbox_to_anchor=(0.02, 0.42), borderaxespad=0, loc=2, numpoints=2, handlelength=2, prop=gs_font, fontsize=label_fontsize)
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
methods = ["MaxScoreBatchSubsetWithSkipsTopK", "MaxScoreBatchSubsetWithSkips", "WAP5", "vPath", "FCFS"]
compress_levels = [1, 200, 1000, 4000, 10000, 15000]
call_graphs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
for i in range(len(methods)):
    x = []
    y = []
    for j in range(len(compress_levels)):
        z = []
        for k in range(len(call_graphs)):
            with open(results_directory + "accuracy_alibaba_cg_" + str(call_graphs[k]) + "_" + test_name_suffix + "_1_" + str(compress_levels[j]) + "_1_0.0.pickle", 'rb') as afile:
                accuracy_load = pickle.load(afile)
                z.append(accuracy_load[methods[i]])
        x.append(compress_levels[j])
        y.append(z)
    xs.append(x)
    ys.append(y)

methods = ["TraceWeaver (Top K)", "TraceWeaver", "WAP5", "vPath", "FCFS"]
plot_box(xs, ys, methods, "Load Multiple (Compression Factor)", "Accuracy Dist. (Across Call Graphs)", output_file_name)
