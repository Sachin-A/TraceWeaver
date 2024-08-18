import os
import pickle
import sys

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.stats import pearsonr

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

def plot_scatter(xs, ys, labels, xlabel, ylabel, outfile):
    plt.figure(figsize=(10,10))
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, color=pcolors[0], marker=markers[1], s=100, alpha=0.5)
    label_fontsize = 20
    ax.set_xlabel(xlabel, fontproperties=gs_font, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontproperties=gs_font, fontsize=label_fontsize)
    plt.ylim(ymin=60, ymax=100)
    plt.xlim(xmin=60, xmax=100)
    ax.grid(linestyle=':', linewidth=1, color='grey')
    ticklabelcolor = 'black'
    plt.tick_params(labelsize=label_fontsize)
    axcolor = 'black'
    ax.xaxis.set_tick_params(width=2, length=10)
    ax.yaxis.set_tick_params(width=2, length=10)
    ax.xaxis.set_tick_params(which='both', colors=axcolor)
    ax.yaxis.set_tick_params(which='both', colors=axcolor)
    ax.spines['bottom'].set_color(axcolor)
    ax.spines['top'].set_color(axcolor)
    ax.spines['right'].set_color(axcolor)
    ax.spines['left'].set_color(axcolor)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color(light_grey)
    ax.spines['top'].set_linestyle(':')
    ax.spines['right'].set_linestyle(':')
    for label in ax.get_xticklabels():
        label.set_fontproperties(gs_font)
    for label in ax.get_yticklabels():
        label.set_fontproperties(gs_font)
    plt.tight_layout()
    plt.savefig(outfile)

xs = []
ys = []
method = "MaxScoreBatchSubsetWithSkips"
compress_level = 15000
call_graphs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

combined_data = {}

for k in range(len(call_graphs)):
    file_path = results_directory + "confidence_scores_alibaba_cg_" + str(call_graphs[k]) + "_" + test_name_suffix + "_1_" + str(compress_level) + "_1_0.0.pickle"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as afile:
            accuracy_confidence = pickle.load(afile)
            for process, values in accuracy_confidence.items():
                if process not in combined_data:
                    combined_data[process] = []
                combined_data[process].append(values)

x = []
y = []
for process, values in combined_data.items():
    for value in values:
        x.append(value[0] * 100)
        y.append((1 - (value[1] / value[2])) * 100)

plot_scatter(x, y, ["Accuracy", "Confidence"], "Accuracy (%)", "Confidence Score", output_file_name)
print("Pearson coefficient:", pearsonr(x,y))
