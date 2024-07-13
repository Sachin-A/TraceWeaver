import os
import sys
import pickle
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import warnings
warnings.filterwarnings("ignore")

pcolors = ['#008000', '#990000', '#a5669f',  '#db850d',  '#00112d']
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
    # create plot
    fig, ax = plt.subplots()
    #ax.set_xscale("log", basex=2)
    #ax.set_yscale("log", basey=2)
    nlines = len(xs)
    assert (len(ys) == nlines and len(labels) == nlines)
    for i in range(nlines):
        pcolor = pcolors[i]
        ax.plot(xs[i], ys[i], '-', color=pcolor,  lw=2.5,  marker=markers[i], mew = 1.5, fillstyle="full", markersize = 10, markeredgecolor=pcolor, dash_capstyle='round', label=labels[i], zorder=10, clip_on=False)
        #ax.plot(xs[i], ys[i], '-', color=pcolor,  lw=2.5,  dash_capstyle='round', label=labels[i])

    label_fontsize=20
    ax.set_xlabel(xlabel, fontproperties=gs_font, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontproperties=gs_font, fontsize=label_fontsize)
    xmax=1.05*max([max(x) for x in xs])
    xmax, ymax = 200, 60
    plt.xlim(xmax=76.0)
    # plt.xlim(xmax=46.0)
    plt.ylim(ymin=0, ymax=105)
    # plt.axhline(y = 0.9, color = 'r', linestyle = '--', linewidth = 1)
    #plt.ylim(ymin=-0.1, ymax=1.0) #ymax=1.01*max([max(y) for y in ys]))
    #plt.annotate('correct prediction', xy=(0.45, 0.90), xycoords='axes fraction', fontsize=label_fontsize)
    #ax.axhspan(0.0, xmax, alpha=0.6, color='grey')
    ax.grid(linestyle=':', linewidth=1, color='grey')
    ticklabelcolor = 'black'
    #xticks = np.array([300, 1200, 4800, 19200, 76800])
    #ax.xaxis.set_ticks(xticks)
    #xticks = np.array([string(x) for x in xticks])
    #ax.set_xticklabels(xticks, color=ticklabelcolor)
    leg = ax.legend(bbox_to_anchor=(0.28, 0.6), borderaxespad=0, loc=2, numpoints=2, handlelength=2, prop=gs_font, fontsize=label_fontsize)
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

def string_to_numeric_array(s):
    return [float(x) for x in s.split()]

load = 150
cache_hit_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
xs = []
ys = []
methods = ["MaxScoreBatchSubsetWithSkips", "WAP5", "vPath", "FCFS"]
for i in range(len(methods)):
    x = []
    y = []
    for j in range(len(cache_hit_levels)):
        # if methods[i] == "MaxScoreBatchSubsetWithSkips":
        with open(results_directory + "accuracy_" + str(load) + "_" + test_name_suffix + "_" + str(cache_hit_levels[j]) + ".pickle", 'rb') as afile:
            accuracy_load = pickle.load(afile)
        # else:
        #     with open("accuracy_150__skipTestCameraReady_" + str(cache_hit_levels[j]) + ".pickle", 'rb') as afile:
        #         accuracy_load = pickle.load(afile)
        x.append((j) * 5)
        y.append(accuracy_load[methods[i]])
    xs.append(x)
    ys.append(y)

print(xs)
print(ys)

methods = ["TraceWeaver", "WAP5", "vPath", "FCFS"]
plot_lines(xs, ys, methods, "Cache %", "Accuracy %", output_file_name)
