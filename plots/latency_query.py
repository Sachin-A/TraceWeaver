import pickle
import numpy as np
import seaborn as sns
import statistics as stats
import matplotlib.pyplot as plt

methods = ["Greedy++", "Greedy", "FCFS", "FCFS++"]
services = ["Search", "Geo", "Rate", "Reservation", "Profile"]

# for k in range(len(methods)):

for l in range(4, 5):

    load_level = (l + 1) * 25

    with open('query_latency_' + str(load_level) + '_version2_before100.pickle', 'rb') as handle:
        query_latency = pickle.load(handle)

    data = [[[] for j in range(5)] for i in range(2)]
    for i in range(2):
        for j in range(5):
                data[i][j] = [x[3] for x in query_latency[methods[0]][i][j]]

    # data2 = [[[] for j in range(5)] for i in range(2)]
    # for i in range(2):
    #     for j in range(5):
    #             data2[i][j] = [x[3] for x in query_latency[methods[3]][i][j]]

    with open('query_latency_' + str(load_level) + '_all_version2_before100.pickle', 'rb') as handle:
        query_latency_all = pickle.load(handle)

    data3 = [[[] for j in range(5)] for i in range(2)]
    for i in range(2):
        for j in range(5):
                data3[i][j] = [x[3] for x in query_latency_all[methods[0]][i][j]]

    plot1 = plt.boxplot(
        data[0][:],
        positions = np.array(np.arange(len(data[0][:]))) * 3.0 + 0.8,
        widths = 0.6,
        whis = [5, 95],
        showfliers = False
    )
    plot2 = plt.boxplot(
        data[1][:],
        positions = np.array(np.arange(len(data[1][:]))) * 3.0,
        widths = 0.6,
        whis = [5, 95],
        showfliers = False
    )
    # plot3 = plt.boxplot(
    #     data2[1][:],
    #     positions = np.array(np.arange(len(data2[1][:]))) * 3.0 - 0.8,
    #     widths = 0.6
    # )

    for z in range(5):
        p1 = int(0.95 * len(data3[0][z]))
        data3[0][z].sort()
        data3[0][z] = data3[0][z][p1:]

    plot3 = plt.boxplot(
        data3[0][:],
        positions = np.array(np.arange(len(data3[0][:]))) * 3.0 - 0.8,
        widths = 0.6,
        whis = [5, 95],
        showfliers = False
    )

    def define_box_properties(plot_name, color_code, label):
        for k, v in plot_name.items():
            plt.setp(plot_name.get(k), color=color_code)

        plt.plot([], c = color_code, label = label)
        plt.legend()


    define_box_properties(plot1, '#D7191C', 'Ground Truth')
    define_box_properties(plot2, '#2C7BB6', 'Greedy++')
    # define_box_properties(plot3, '#f6c42c', 'FCFS++')
    define_box_properties(plot3, '#f6c42c', 'All spans')

    plt.xticks(np.arange(0, len(services) * 3, 3), services)

    plt.xlim(-2, len(services) * 3)
    # plt.ylim(0, 50)

    plt.show()
