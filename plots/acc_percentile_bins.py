import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
 

pcolors = ['#0e326d', '#028413', '#a5669f', '#db850d', '#00112d', '#af0505']
pcolors = ['#000080', '#008000', '#990000', '#a5669f',  '#db850d',  '#00112d']
markers = ['s', 'o', 'x', '^', 'v', '*', 'p', 'h']
linestyle = ['-', '.', '--']

gs_font = fm.FontProperties(fname='gillsans.ttf', size=20, weight='bold')
light_grey=(0.5,0.5,0.5)

def string(x):
    if x >= (1024 * 1024 * 1024):
        return str(x/(1024*1024*1024)) + 'B'    
    elif x >= (1024 * 1024):
        return str(x/(1024*1024)) + 'M'
    elif x >= 1024:
        return str(x/1024) + 'K'
    else:
        return str(x)

infile = sys.argv[1]
outfile = sys.argv[2]

def plot_lines(xs, ys, labels, xlabel, ylabel, outfile):
    # create plot
    fig, ax = plt.subplots()
    #ax.set_xscale("log", basex=2)
    #ax.set_yscale("log", basey=2)
    nlines = len(xs)
    assert (len(ys) == nlines and len(labels) == nlines)
    for i in range(nlines):
        pcolor = pcolors[i]
        ax.plot(xs[i], ys[i], '-', color=pcolor,  lw=2.5,  marker=markers[i], mew = 1.5, markersize = 9, markerfacecolor='none', markeredgecolor=pcolor, dash_capstyle='round', label=labels[i])
        #ax.plot(xs[i], ys[i], '-', color=pcolor,  lw=2.5,  dash_capstyle='round', label=labels[i])

    label_fontsize=20
    ax.set_xlabel(xlabel, fontproperties=gs_font, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontproperties=gs_font, fontsize=label_fontsize)
    xmax=1.05*max([max(x) for x in xs])
    xmax, ymax = 200, 60
    plt.xlim(xmax=100.0)
    plt.ylim(ymin=0, ymax=1.0)
    #plt.annotate('correct prediction', xy=(0.45, 0.90), xycoords='axes fraction', fontsize=label_fontsize)
    #ax.axhspan(0.0, xmax, alpha=0.6, color='grey')
    ax.grid(linestyle=':', linewidth=1, color='grey')
    ticklabelcolor = 'black'
    #xticks = np.array([300, 1200, 4800, 19200, 76800])
    #ax.xaxis.set_ticks(xticks)
    #xticks = np.array([string(x) for x in xticks])
    #ax.set_xticklabels(xticks, color=ticklabelcolor)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = zip(*[ (handles[i], labels[i]) for i in sorted(range(len(handles)), key=lambda k: list(labels)[k])])
    print(labels)
    order = [3,2,1,0]
    print([labels[idx] for idx in order])
    leg = ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], bbox_to_anchor=(0.28, 0.45), borderaxespad=0, loc=2, numpoints=2, handlelength=2, prop=gs_font, fontsize=label_fontsize)
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

xs = []
ys = []
labels = []
with open(infile, 'rb') as afile:
    accuracy_percentile_bins = pickle.load(afile)
    for method, acc in accuracy_percentile_bins.items():
        x = []
        y = []
        for p, a, t in acc:
            x.append(p)
            y.append(a)
        labels.append(method.lower())
        xs.append(x)
        ys.append(y)

plot_lines(xs, ys, labels, "Latency percentile", "Accuracy", outfile)



