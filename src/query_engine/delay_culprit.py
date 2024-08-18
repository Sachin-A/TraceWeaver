import argparse
import os
import pickle
import sys

sys.path.append("..")

from trace_reconstructor.ports.python.helpers.misc import get_project_root
from trace_reconstructor.ports.python.spans import Span

parser = argparse.ArgumentParser(
        description='Execute query: Identify service contributing most\
                     delay to hot-path'
    )

PROJECT_ROOT = get_project_root()
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots/")

'''
FOR
    all end to end requests
WHICH
    were in the top X %ile response latency bracket AND
    were initiated after time Y,
FIND
    the worst performing service AND
    its mean service latency for these requests
'''

def delay_culprit():

    for j in range(4, 5):

        query_latency = {}

        with open(PLOTS_DIR + "e2e_" + str((j + 1) * 25) + "_version2.pickle", 'rb') as afile:
            e2e_traces = pickle.load(afile)

        for method in e2e_traces.keys():

            true_traces = e2e_traces[method][0]

            true_traces = dict(
                sorted(
                    true_traces.items(),
                    key=lambda x: x[1][-1].start_mus + x[1][-1].duration_mus - x[1][0].start_mus
                )
            )

            # p1 = int(0.95 * len(true_traces))
            p1 = int(0 * len(true_traces))
            start_time = list(true_traces.items())[100][1][0].start_mus

            # true_traces = list(
            #     filter(
            #         lambda x: x[1][0].start_mus > start_time,
            #         list(true_traces.items())[p1:]
            #     )
            # )
            true_traces = list(
                filter(
                    lambda x: x[1][0].start_mus < start_time,
                    list(true_traces.items())[p1:]
                )
            )
            # true_traces = list(
            #     filter(
            #         lambda x: x[1][0].start_mus > 0,
            #         list(true_traces.items())[p1:]
            #     )
            # )

            pred_traces = e2e_traces[method][1]
            pred_traces_assigned = []

            for trace in true_traces:
                if not any(x is None for x in pred_traces[trace[0]]):
                    pred_traces_assigned.append((trace[0], pred_traces[trace[0]]))

            latency_per_service_true = [[] for i in range(5)]
            latency_per_service_pred = [[] for i in range(5)]

            for _, trace in true_traces:
                for i, span in enumerate(trace):
                    latency_per_service_true[i].append((span.trace_id, span.sid, span.start_mus, span.duration_mus))
            for _, trace in pred_traces_assigned:
                for i, span in enumerate(trace):
                    latency_per_service_pred[i].append((span.trace_id, span.sid, span.start_mus, span.duration_mus))

            query_latency[method] = [latency_per_service_true, latency_per_service_pred]

        LOAD_LEVEL = (j + 1) * 25

        with open(PLOTS_DIR + 'query_latency_' + str(LOAD_LEVEL) + '_version2_randomSet.pickle', 'wb') as handle:
            pickle.dump(query_latency, handle, protocol = pickle.HIGHEST_PROTOCOL)
        with open(PLOTS_DIR + 'query_latency_' + str(LOAD_LEVEL) + '_all_version2_randomSet.pickle', 'wb') as handle:
            pickle.dump(query_latency, handle, protocol = pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    delay_culprit()
