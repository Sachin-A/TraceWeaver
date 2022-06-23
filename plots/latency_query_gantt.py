import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

with open('query_latency_125all.pickle', 'rb') as handle:
    query_latency = pickle.load(handle)

for b in range(1000):

    methods = ["Greedy++", "Greedy", "FCFS", "FCFS++"]
    services = ["Search", "Geo", "Rate", "Reservation", "Profile"]
    servicesY = [5, 4, 3, 2, 1]

    delays1 = []
    starts1 = []
    delays2 = []
    starts2 = []
    delays3 = []
    starts3 = []

    for i in range(5):
        delays1.append(query_latency[methods[0]][0][i][b][3])
        delays2.append(query_latency[methods[0]][1][i][b][3])
        delays3.append(query_latency[methods[3]][1][i][b][3])
        starts1.append(query_latency[methods[0]][0][i][b][2])
        starts2.append(query_latency[methods[0]][1][i][b][2])
        starts3.append(query_latency[methods[3]][1][i][b][2])

    mindelay = min(min(starts1), min(starts2), min(starts3))
    starts1 = list(np.array(starts1) - mindelay)
    starts2 = list(np.array(starts2) - mindelay)
    starts3 = list(np.array(starts3) - mindelay)

    fig, ax = plt.subplots(1, figsize=(16,6))

    delays1.reverse()
    starts1.reverse()
    delays2.reverse()
    starts2.reverse()
    delays3.reverse()
    starts3.reverse()

    services.reverse()
    servicesY.reverse()

    # color = "#eb4034"
    # color = "#2617ad"

    # c_dict = {
    #     "Search":'#E64646',
    #     "Geo":'#E69646',
    #     "Rate":'#34D05C',
    #     "Recommendations":'#34D0C3',
    #     "Profile":'#3475D0'
    # }

    # c_list = [
    #     '#E64646',
    #     '#E69646',
    #     '#34D05C',
    #     '#34D0C3',
    #     '#3475D0'
    # ]

    c_list = [
        '#E64646',
        '#E69646',
        '#34D05C'
    ]

    c_list.reverse()

    ax.barh(servicesY, delays1, left = starts1, color = c_list[0], height = 0.15, label = "Ground Truth")
    ax.barh(np.array(servicesY) + 0.2, delays2, color = c_list[1], left = starts2, height = 0.15, label = "Greedy++")
    ax.barh(np.array(servicesY) + 0.4, delays3, color = c_list[2], left = starts3, height = 0.15, label = "FCFS++")
    ax.set(yticks = np.array(servicesY) + 0.1, yticklabels = services)

    # legend_elements = [Patch(facecolor = c_dict[i], label = i) for i in c_dict]
    # plt.legend(handles = legend_elements)
    plt.legend()

    # xticks = np.arange(0, df.end_num.max()+1, 3)
    # xticks_labels = pd.date_range(proj_start, end=df.End.max()).strftime("%m/%d")
    # xticks_minor = np.arange(0, df.end_num.max()+1, 1)
    # ax.set_xticks(xticks)

    # ax.set_xticks(xticks_minor, minor=True)
    # ax.set_xticklabels(xticks_labels[::3])
    plt.show()
