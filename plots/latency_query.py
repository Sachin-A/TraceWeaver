import pickle5 as pickle
import numpy as np
import seaborn as sns
import statistics as stats
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

methods = ["Greedy++", "Greedy", "FCFS", "FCFS++"]
services = ["Search", "Geo", "Rate", "Reserve", "Profile"]

pcolors = ['#000080', '#008000', '#990000', '#a5669f',  '#db850d',  '#00112d']
gs_font = fm.FontProperties(fname='gillsans.ttf', size=15, weight='bold')
light_grey=(0.5,0.5,0.5)

# for k in range(len(methods)):

nreq = "_na_" #500
percentile = "_na_"


#for l in range(4, 5):
for l in range(0, 1):

    load_level = (l + 1) * 25

    with open('vipul/query_latency_' + str(load_level) + '_version2_before'+str(nreq)+'_p'+str(percentile)+'.pickle', 'rb') as handle:
        query_latency = pickle.load(handle)

    data = [[[] for j in range(5)] for i in range(2)]
    for i in range(2):
        for j in range(5):
                data[i][j] = [x[3]/1000 for x in query_latency[methods[0]][i][j]]

    # data2 = [[[] for j in range(5)] for i in range(2)]
    # for i in range(2):
    #     for j in range(5):
    #             data2[i][j] = [x[3] for x in query_latency[methods[3]][i][j]]

    with open('vipul/query_latency_' + str(load_level) + '_all_version2_before'+str(nreq)+'_p'+str(percentile)+'.pickle', 'rb') as handle:
        query_latency_all = pickle.load(handle)

    data3 = [[[] for j in range(5)] for i in range(2)]
    for i in range(2):
        for j in range(5):
                data3[i][j] = [x[3]/1000 for x in query_latency_all[methods[0]][i][j]]

    fig, ax = plt.subplots()
    boxprops = dict(linewidth=1.5)
    plot1 = plt.boxplot(
        data[0][:],
        positions = np.array(np.arange(len(data[0][:]))) * 3.5 + 0.8,
        widths = 0.6,
        whis = [5, 95],
        showfliers = False,
        boxprops = boxprops,
    )
    plot2 = plt.boxplot(
        data[1][:],
        positions = np.array(np.arange(len(data[1][:]))) * 3.5,
        widths = 0.6,
        whis = [5, 95],
        showfliers = False,
        boxprops = boxprops,
    )
    print(len(data[1][0]))
    # plot3 = plt.boxplot(
    #     data2[1][:],
    #     positions = np.array(np.arange(len(data2[1][:]))) * 3.5 - 0.8,
    #     widths = 0.6
    # )

    # comment out for all spans
    for z in range(5):
        print(len(data3[0][z]))
        data3[0][z].sort()
        #data3[0][z] = data3[0][z][p1:]

    plot3 = plt.boxplot(
        data3[0][:],
        positions = np.array(np.arange(len(data3[0][:]))) * 3.5 - 0.8,
        widths = 0.6,
        whis = [5, 95],
        showfliers = False,
        boxprops = boxprops,
    )

    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)

        plt.plot([], c = color_code, label = label)
        plt.legend()

    '''
    define_box_properties(plot1, '#D7191C', 'Ground Truth')
    define_box_properties(plot2, '#2C7BB6', 'MaxScoreBatch')
    # define_box_properties(plot3, '#f6c42c', 'FCFS++')
    define_box_properties(plot3, '#f6c42c', 'Developer view')
    '''
    define_box_properties(plot3, pcolors[2], 'Developer view \n w/o tracing')
    define_box_properties(plot2, pcolors[1], 'MaxScoreBatch')
    define_box_properties(plot1, pcolors[0], 'Ground Truth')
    # define_box_properties(plot3, '#f6c42c', 'FCFS++')
    plt.xticks(np.arange(0, len(services) * 3.5, 3.5), services)

    #plt.annotate('correct prediction', xy=(0.45, 0.90), xycoords='axes fraction', fontsize=label_fontsize)

    plt.xlim(-2, len(services) * 3.1)
    # plt.ylim(0, 50)

    label_fontsize = 18
    leg = ax.legend(bbox_to_anchor=(0.58, 0.35), borderaxespad=0, loc=2, numpoints=2, handlelength=2, prop=gs_font, fontsize=label_fontsize)
    leg.get_frame().set_linewidth(0.0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color(light_grey)
    ax.spines['top'].set_linestyle(':')
    ax.spines['right'].set_linestyle(':')

    for label in ax.get_xticklabels() :
        label.set_fontproperties(gs_font)
    for label in ax.get_yticklabels() :
        label.set_fontproperties(gs_font)
    ax.grid(linestyle=':', linewidth=1, color='grey', axis='y')
    ax.set_xlabel("Service", fontproperties=gs_font, fontsize=label_fontsize)
    ax.set_ylabel("Response time (ms)", fontproperties=gs_font, fontsize=label_fontsize)
    plt.tight_layout()
    #plt.show()
    plt.savefig("vipul/plot.pdf")
