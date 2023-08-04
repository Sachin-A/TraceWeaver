import os
import sys
import json
import math
import copy
import random
import pickle
import string
import numpy as np
from fcfs import FCFS
from vpath import VPATH
from fcfs2 import FCFS2
from timing import Timing
from timing2 import Timing2
from timing3 import Timing3
from wap5_og import WAP5_OG
from deepdiff import DeepDiff
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats

# np.seterr(all='raise')

VERBOSE = False
random.seed(10)

process_map_1 = {
    "service5": "service3",
    "service4": "service2",
    "service2": "service1",
    "service3": "service1",
    "service1": "init-service"
}

satisfied_bool = {}
satisfied_float = {}
replica_id = {}
new_replica_id = {}

all_spans = dict()
all_processes = dict()

class Span(object):
    def __init__(
        self,
        trace_id,
        sid,
        start_mus,
        duration_mus,
        op_name,
        references,
        process_id,
        span_kind,
        span_tags
    ):
        self.sid = sid
        self.trace_id = trace_id
        self.start_mus = start_mus
        self.duration_mus = duration_mus
        self.op_name = op_name
        self.references = references
        self.process_id = process_id
        self.span_kind = span_kind
        self.tags = span_tags
        self.children_spans = []
        self.taken = False
        self.ep = None

    def AddChild(self, child_span_id):
        self.children_spans.append(child_span_id)

    def GetChildProcess(self):
        assert self.span_kind == "client"
        assert len(self.children_spans) == 1
        return all_processes[self.trace_id][
            all_spans[self.children_spans[0]].process_id
        ]

    def GetParentProcess(self):
        if self.IsRoot():
            return "client_" + self.op_name
        assert len(self.references) == 1
        parent_span_id = self.references[0]
        return all_processes[self.trace_id][all_spans[parent_span_id].process_id]

    def GetId(self):
        return (self.trace_id, self.sid)

    def IsRoot(self):
        return len(self.references) == 0

    def __lt__(self, other):
        return self.start_mus < other.start_mus

    def __repr__(self):
        if self.start_mus == "None":
            return "Span:(%s, %s, %s, %s, %s, %s)" % (
                self.trace_id,
                self.sid,
                self.op_name,
                self.start_mus,
                self.duration_mus,
                self.span_kind,
            )
        else:
            return "Span:(%s, %s, %s, %d, %d, %s)" % (
                self.trace_id,
                self.sid,
                self.op_name,
                self.start_mus,
                self.duration_mus,
                self.span_kind,
            )

    def __str__(self):
        return self.__repr__()

def repeatChangeSpans(in_span_partitions, out_span_partitions, repeats, factor):
    assert len(in_span_partitions) == 1
    in_span_partitions_old = copy.deepcopy(in_span_partitions)
    out_span_partitions_old = copy.deepcopy(out_span_partitions)
    ep_in, in_spans = list(in_span_partitions_old.items())[0]

    span_inds = []
    for ind, in_span in enumerate(in_spans):
        time_order = True
        for ep_out in out_span_partitions.keys():
            out_span = out_span_partitions[ep_out][ind]
            time_order = (
                time_order
                and (float(in_span.start_mus) <= float(out_span.start_mus))
                and (
                    float(out_span.start_mus) + float(out_span.duration_mus)
                    <= float(in_span.start_mus) + float(in_span.duration_mus)
                )
            )
        if time_order:
            span_inds.append(ind)

    in_span_partitions[ep_in] = []
    for ep_out in out_span_partitions_old.keys():
        out_span_partitions[ep_out] = []

    span_inds = span_inds * repeats
    random.shuffle(span_inds)
    min_start_t = min(float(in_span.start_mus) for in_span in in_spans) / factor
    max_start_t = max(float(in_span.start_mus) for in_span in in_spans) / factor
    start_ts = sorted([random.uniform(min_start_t, max_start_t) for _ in span_inds])
    for ind, start_t in zip(span_inds, start_ts):
        # if len(in_span_partitions[ep_in]) > 40:
        #    continue
        trace_id = "".join(
            random.choice(string.ascii_lowercase + string.digits) for _ in range(32)
        )
        in_span = copy.deepcopy(in_spans[ind])
        in_span.start_mus = float(in_span.start_mus)
        offset = start_t - in_span.start_mus
        in_span.trace_id = trace_id
        in_span.start_mus += offset
        in_span_partitions[ep_in].append(in_span)
        for ep_out in out_span_partitions_old.keys():
            out_span = copy.deepcopy(out_span_partitions_old[ep_out][ind])
            out_span.start_mus = float(out_span.start_mus)
            out_span.trace_id = trace_id
            out_span.start_mus += offset
            out_span_partitions[ep_out].append(out_span)
    return in_span_partitions, out_span_partitions

def topological_sort_grouped(G):
    indegree_map = {v: d for v, d in G.in_degree() if d > 0}
    zero_indegree = [v for v, d in G.in_degree() if d == 0]
    grouped_list = []
    while zero_indegree:
        # yield zero_indegree
        grouped_list.append(zero_indegree)
        new_zero_indegree = []
        for v in zero_indegree:
            for _, child in G.edges(v):
                indegree_map[child] -= 1
                if not indegree_map[child]:
                    new_zero_indegree.append(child)
        zero_indegree = new_zero_indegree
    return grouped_list

def FindOrder(in_span_partitions, out_span_partitions, true_assignments):
    assert len(in_span_partitions) == 1

    ep_in, in_spans = list(in_span_partitions.items())[0]
    order = set()
    out_eps = list(out_span_partitions.keys())
    G = nx.DiGraph()
    G1 = nx.DiGraph()
    for i in range(len(out_eps)):
        G.add_node(i)
        G1.add_node(out_eps[i])
    for i in range(len(out_eps)):
        for j in range(len(out_eps)):
            if i != j:
                G.add_edge(i, j)
                G1.add_edge(out_eps[i], out_eps[j])
    for in_span in in_spans:
        outgoing_spans = []
        outgoing_eps = {}
        for out_ep in out_eps:
            span = all_spans[true_assignments[out_ep][in_span.GetId()]]
            outgoing_spans.append([span.start_mus, span.duration_mus, span.GetParentProcess(), span.GetChildProcess()])
        outgoing_spans.sort(key=lambda x: x[0])

        for i, x in enumerate(outgoing_spans):
            outgoing_eps[i] = x[3]

        for i, x in enumerate(outgoing_spans):
            for j, y in enumerate(outgoing_spans):
                if i != j:
                    if x[0] + x[1] > y[0]:
                        if G.has_edge(i, j):
                            G.remove_edge(i, j)
                        if G1.has_edge(x[3], y[3]):
                            G1.remove_edge(x[3], y[3])
                    if y[0] + y[1] > x[0]:
                        if G.has_edge(j, i):
                            G.remove_edge(j, i)
                        if G1.has_edge(y[3], x[3]):
                            G1.remove_edge(y[3], x[3])

    sorted_grouped_order = topological_sort_grouped(G)
    service_order = copy.deepcopy(sorted_grouped_order)
    for i in range(len(sorted_grouped_order)):
        for j, service_id in enumerate(sorted_grouped_order[i]):
            service_order[i][j] = outgoing_eps[sorted_grouped_order[i][j]]

    print(service_order)
    print(topological_sort_grouped(G1))

    return G1

def GetOutEpsInOrder(out_span_partitions):
    eps = []
    for ep, spans in out_span_partitions.items():
        assert len(spans) > 0
        eps.append((ep, spans[0].start_mus))
    eps.sort(key=lambda x: x[1])
    return [x[0] for x in eps]

'''
FOR all e2e requests
WHICH
    were in the top 5% response latency bracket AND
    were initiated after time X,
FIND
    the worst performing service AND
    its mean service latency for these requests
'''

def sampleQuery():

    for j in range(4, 5):

        query_latency = {}

        with open("plots/e2e_" + str((j + 1) * 25) + "_version2.pickle", 'rb') as afile:
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

        load_level = (j + 1) * 25

        with open('plots/query_latency_' + str(load_level) + '_version2_randomSet.pickle', 'wb') as handle:
            pickle.dump(query_latency, handle, protocol = pickle.HIGHEST_PROTOCOL)
        with open('plots/query_latency_' + str(load_level) + '_all_version2_randomSet.pickle', 'wb') as handle:
            pickle.dump(query_latency, handle, protocol = pickle.HIGHEST_PROTOCOL)


def GetAllTracesInDir(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(directory + "/" + f)]
    files = [f for f in files if f.endswith("json")]
    full_path = os.path.abspath(directory)
    files = [full_path + "/" + f for f in files]
    return files


def ParseSpansJson(spans_json):

    spans = {}
    # random_num = random.randint(0, 999)
    # random_num2 = random.randint(0, 999)
    for span in spans_json:
        references = []
        for ref in span["references"]:
            references.append((ref["traceID"], ref["spanID"]))
        trace_id = span["traceID"]
        sid = span["spanID"]
        span_id = (trace_id, sid)
        start_mus = span["startTime"]
        duration_mus = span["duration"]
        op_name = span.get("operationName", None)
        process_id = span["processID"]
        span_kind = None

        for tag in span["tags"]:
            if tag["key"] == "span.kind":
                span_kind = tag["value"]

        # if random_num < int(sys.argv[9]):
        #     replica_id[trace_id] = "recommendation1"
        #     if random_num2 >= 666:
        #         score = random.uniform(0, 4)
        #         satisfied_float[trace_id] = score
        #     elif random_num2 >= 333 and random_num2 < 666:
        #         score = random.uniform(4, 8)
        #         satisfied_float[trace_id] = score
        #     elif random_num2 < 333:
        #         score = random.uniform(8, 10)
        #         satisfied_float[trace_id] = score
        # else:
        #     replica_id[trace_id] = "recommendation2"
        #     if random_num2 >= 550:
        #         score = random.uniform(0, 4)
        #         satisfied_float[trace_id] = score
        #     elif random_num2 >= 333 and random_num2 < 570:
        #         score = random.uniform(4, 8)
        #         satisfied_float[trace_id] = score
        #     elif random_num2 < 333:
        #         score = random.uniform(8, 10)
        #         satisfied_float[trace_id] = score

        # for tag in span["tags"]:
        #     if tag["key"] == "span.kind":
        #         span_kind = tag["value"]
        #     if tag["key"] == "replicaType":
        #         replica_id[trace_id] = tag["value"]
        #         if tag["value"] == "recommendation1":
        #             if random_num2 < 800:
        #                 satisfied_bool[trace_id] = True
        #             else:
        #                 satisfied_bool[trace_id] = False
        #         elif tag["value"] == "recommendation2":
        #             if random_num2 < 700:
        #                 satisfied_bool[trace_id] = True
        #             else:
        #                 satisfied_bool[trace_id] = False
        #     # if tag["key"] == "satisfied":
        #     #     satisfied_bool[trace_id] = tag["value"]

        spans[span_id] = Span(
            trace_id,
            sid,
            start_mus,
            duration_mus,
            op_name,
            references,
            process_id,
            span_kind,
            span["tags"]
        )
    # if random_num2 < 990:
    #     replica_id[trace_id] = "A"
    # else:
    #     replica_id[trace_id] = "B"
    return spans


def ParseProcessesJson(processes_json):
    processes = {}
    for pid in processes_json:
        processes[pid] = processes_json[pid]["serviceName"]
    return processes

def FixSpans(spans, processes):

    process_map_2 = {}

    def GetProcessOfSpan(span_id):
        pid = spans[span_id].process_id
        return processes[pid]

    for span_id, span in spans.items():
        process = GetProcessOfSpan(span_id)
        process_map_2[process] = span.process_id

    new_spans = {}
    for span_id, span in spans.items():
        process = GetProcessOfSpan(span_id)
        if span.span_kind == "client":
            span.span_kind = "server"
        elif span.span_kind == "server":
            span_copy = copy.deepcopy(span)
            copy_ref = copy.deepcopy(span.references)
            span.references[0] = (copy_ref[0][0], span.sid + "_client")
            client_process = process_map_1[process]
            span_copy.sid = span_copy.sid + "_client"
            span_copy.process_id = process_map_2[client_process]
            span_copy.span_kind = "client"
            span_copy.references = copy_ref
            new_spans[(span_copy.trace_id, span_copy.sid)] = span_copy

    spans.update(new_spans)
    return spans

new_process_count = 0
new_process_reverse_map = {}
new_processes = {}
def FixSpans2(spans, processes):

    def FindParentProcess(id):
        return spans[id].process_id

    def FindGrandParentProcess(id):
        for span_id, span in spans.items():
            if span_id == id and len(span.references) != 0:
                return FindParentProcess(span.references[0])
        return None

    def DeleteAncestors(id):
        if len(spans[id].references) != 0:
            DeleteAncestors(spans[id].references[0])
        del new_spans[id]

    def ChangeChildReferences(id):
        for span_id, span in spans.items():
            if len(span.references) != 0:
                if span.references[0] == id:
                    new_ref = (span.trace_id, span.trace_id)
                    new_spans[span_id].references[0] = new_ref

    new_spans = copy.deepcopy(spans)
    for span_id, span in spans.items():
        if span.op_name == "ComposeReview":
            DeleteAncestors(span.references[0])
            ChangeChildReferences(span_id)
            span.sid = span.trace_id
            span.references = []
            new_spans[(span.trace_id, span.sid)] = span
            del new_spans[span_id]

    spans = copy.deepcopy(new_spans)
    for span_id, span in spans.items():
        if len(span.references) != 0:
            parent_process = FindParentProcess(span.references[0])
            if parent_process != None:
                if parent_process == span.process_id:
                    del new_spans[span_id]

    spans = copy.deepcopy(new_spans)
    new_spans2 = {}

    for span_id, span in spans.items():
        span.span_kind = "server"
        if len(span.references) != 0:
            span_copy = copy.deepcopy(span)
            copy_ref = copy.deepcopy(span.references)
            span.references[0] = (copy_ref[0][0], span.sid + "_client")
            span_copy.sid = span_copy.sid + "_client"
            span_copy.process_id = FindParentProcess(copy_ref[0])
            span_copy.span_kind = "client"
            span_copy.references = copy_ref
            new_spans2[(span_copy.trace_id, span_copy.sid)] = span_copy

    spans.update(new_spans2)

    new_process_map = {}
    spans = {k: v for k, v in sorted(spans.items(), key=lambda item: item[1].start_mus)}

    multiple_map = {}

    def UpdateMap():
        nonlocal multiple_map
        multiple_map = {}
        for span_id, span in spans.items():
            if span.span_kind == "server":
                if span.process_id in processes:
                    process_name = processes[span.process_id]
                else:
                    process_name = new_processes[span.process_id]

                if process_name not in multiple_map:
                    multiple_map[process_name] = []
                if len(span.references) != 0:
                    pid = FindParentProcess(span.references[0])
                    if pid != None:
                        if pid in processes:
                            incoming = processes[pid]
                        else:
                            incoming = new_processes[pid]
                        multiple_map[process_name].append(incoming)

    UpdateMap()
    global new_process_count

    # Fix multiple incoming slicing

    # for span_id, span in spans.items():
    #     if len(span.references) != 0:
    #         if span.span_kind == "server":
    #             process_name = processes[span.process_id]
    #             if len(multiple_map[process_name]) > 1:
    #                 pid = FindParentProcess(span.references[0])
    #                 if pid != None:
    #                     parent_process = processes[pid]
    #                     new_process_name = parent_process + "->" + process_name
    #                     if new_process_name in new_process_reverse_map:
    #                         span.process_id = new_process_reverse_map[new_process_name]
    #                     else:
    #                         new_process_id = "np" + str(new_process_count)
    #                         new_processes[new_process_id] = new_process_name
    #                         new_process_reverse_map[new_process_name] = new_process_id
    #                         span.process_id = new_process_id
    #                         new_process_count += 1

    #         elif span.span_kind == "client":
    #             process_name = processes[span.process_id]
    #             if len(multiple_map[process_name]) > 1:
    #                 gpid = FindGrandParentProcess(span.references[0])
    #                 if gpid != None:
    #                     grand_parent_process = processes[gpid]
    #                     new_process_name = grand_parent_process + "->" + process_name
    #                     if new_process_name in new_process_reverse_map:
    #                         span.process_id = new_process_reverse_map[new_process_name]
    #                     else:
    #                         new_process_id = "np" + str(new_process_count)
    #                         new_processes[new_process_id] = new_process_name
    #                         new_process_reverse_map[new_process_name] = new_process_id
    #                         span.process_id = new_process_id
    #                         new_process_count += 1

    #         UpdateMap()

    # print(new_processes)
    # input()

    processes.update(new_processes)
    all_spans.update(spans)

    return spans, processes

def ParseJsonTrace(trace_json):
    if sys.argv[6] == "fix":
        first_span = "init-span"
        # first_span = "ComposeReview"
    elif sys.argv[6] == "no-fix":
        # first_span = "HTTP GET /hotels"
        first_span = "HTTP GET /recommendations"
        # first_span = "[Todo] CompleteTodoCommandHandler"
    ret = []
    processes = None
    with open(trace_json, "r") as tfile:
        json_data = json.load(tfile)
        json_data = json_data["data"]
        for d in json_data:
            trace_id = d["traceID"]
            spans = ParseSpansJson(d["spans"])
            processes = ParseProcessesJson(d["processes"])
            if first_span == "init-span":
                spans = FixSpans(spans, processes)
            if first_span == "ComposeReview":
                spans, processes = FixSpans2(spans, processes)

            root_service = None
            for span_id, span in spans.items():
                # no references
                if len(span.references) == 0:
                    root_service = span.op_name
            if root_service is not None:
                ret.append((trace_id, spans))
    assert len(ret) == 1
    trace_id, spans = ret[0]

    # print(len(spans.keys()))
    # for span_id, span in spans.items():
    #     print(span, processes[span.process_id], span.references)
    #     input()

    return trace_id, spans, processes

in_spans_by_process = dict()
out_spans_by_process = dict()

def ProcessTraceData(data):

    if sys.argv[6] == "fix":
        first_span = "init-span"
        # first_span = "ComposeReview"
    elif sys.argv[6] == "no-fix":
        # first_span = "HTTP GET /hotels"
        first_span = "HTTP GET /recommendations"
        # first_span = "[Todo] CompleteTodoCommandHandler"

    trace_id, spans, processes = data

    def GetProcessOfSpan(span_id):
        pid = spans[span_id].process_id
        return processes[pid]

    def AddSpanToProcess(span_id):
        span = spans[span_id]
        process = GetProcessOfSpan(span_id)
        if span.span_kind == "client":
            if process not in out_spans_by_process:
                out_spans_by_process[process] = []
            out_spans_by_process[process].append(span)
        elif span.span_kind == "server":
            if process not in in_spans_by_process:
                in_spans_by_process[process] = []
            in_spans_by_process[process].append(span)
        else:
            assert False

    root_span_id = None
    # populate children
    for span_id, span in spans.items():
        if len(span.references) == 0:
            root_span_id = span_id
        for par_id in span.references:
            spans[par_id].AddChild(span.GetId())
    for span_id, span in spans.items():
        span.children_spans.sort(
            key=lambda child_span_id: spans[child_span_id].start_mus
        )

    def ExploreSubTree(span_id, depth):
        span = spans[span_id]
        AddSpanToProcess(span_id)
        # if VERBOSE:
        # print(
        #     (4 * depth) * " ",
        #     span.sid,
        #     span.op_name,
        #     span.start_mus,
        #     span.duration_mus,
        # )
        # input()
        for child in span.children_spans:
            ExploreSubTree(child, depth + 1)

    # for span_id, span in spans.items():
    #     print(span_id)
    #     print(span)
    #     print(span.process_id)
    #     for child_id in span.children_spans:
    #         print("Child:", spans[child_id])
    #         print("Process:", spans[child_id].process_id)
    #     input()

    # comment out if condition to consider all microservice kinds
    if spans[root_span_id].op_name == first_span:
        ExploreSubTree(root_span_id, 0)
        all_spans.update(spans)
        all_processes[trace_id] = processes
        return 1
    return 0

traces_dir = sys.argv[1]
traces = GetAllTracesInDir(traces_dir)
traces.sort()
cnt = 0
for trace in traces:
    if VERBOSE:
        print("\n\n\n")
    data = ParseJsonTrace(trace)
    cnt += ProcessTraceData(data)
    if cnt > 10000:  # 10000:
        break

if VERBOSE:
    print("Incoming spans")
    for p, s in in_spans_by_process.items():
        print("  %s: %s" % (p, s))
    print("Outgoing spans")
    for p, s in out_spans_by_process.items():
        print("  %s: %s" % (p, s))
    print("\n\n\n")

def GetGroundTruth(in_span_partitions, out_span_partitions):
    assert len(in_span_partitions) == 1
    _, in_spans = list(in_span_partitions.items())[0]
    true_assignments = {ep: {} for ep in out_span_partitions.keys()}
    for in_span in in_spans:
        for ep in out_span_partitions.keys():
            for span in out_span_partitions[ep]:
                if span.trace_id == in_span.trace_id:
                    true_assignments[ep][in_span.GetId()] = span.GetId()
                    break
    return true_assignments

def AccuracyForSpan(pred_assignments, true_assignments, in_span_id):
    correct = True
    for ep in true_assignments.keys():
        if isinstance(pred_assignments[ep][in_span_id], list):
            if len(pred_assignments[ep][in_span_id]) > 1:
                correct = False
            else:
                pred_assignments[ep][in_span_id] = pred_assignments[ep][in_span_id][0]
        correct = correct and (
            pred_assignments[ep][in_span_id]
            == true_assignments[ep][in_span_id]
        )
    return int(correct)

def TopKAccuracyForSpan(pred_topk_assignments, true_assignments, in_span_id):
    ep0 = list(true_assignments.keys())[0]
    correct = False
    for i in range(len(pred_topk_assignments[ep0][in_span_id])):
        correct = True
        for ep in true_assignments.keys():
            correct = correct and (
                pred_topk_assignments[ep][in_span_id][i]
                == true_assignments[ep][in_span_id]
            )
        if correct:
            break
    return int(correct)

def AccuracyForService(pred_assignments, true_assignments, in_span_partitions):
    assert len(in_span_partitions) == 1
    _, in_spans = list(in_span_partitions.items())[0]
    cnt = 0
    for in_span in in_spans:
        correct = True
        for ep in true_assignments.keys():
            if isinstance(pred_assignments[ep][in_span.GetId()], list):
                if len(pred_assignments[ep][in_span.GetId()]) > 1:
                    correct = False
                else:
                    pred_assignments[ep][in_span.GetId()] = pred_assignments[ep][in_span.GetId()][0]
            correct = correct and (
                pred_assignments[ep][in_span.GetId()]
                == true_assignments[ep][in_span.GetId()]
            )
        cnt += int(correct)
    return float(cnt) / len(in_spans)

def TopKAccuracyForService(pred_topk_assignments, true_assignments, in_span_partitions):
    assert len(in_span_partitions) == 1
    _, in_spans = list(in_span_partitions.items())[0]
    cnt = 0
    ep0 = list(true_assignments.keys())[0]
    for in_span in in_spans:
        for i in range(len(pred_topk_assignments[ep0][in_span.GetId()])):
            correct = True
            for ep in true_assignments.keys():
                correct = correct and (
                    pred_topk_assignments[ep][in_span.GetId()][i]
                    == true_assignments[ep][in_span.GetId()]
                )
            if correct:
                cnt += int(correct)
                break
    return float(cnt) / len(in_spans)

def AccuracyEndToEnd(
    pred_assignments_by_process, true_assignments_by_process, in_spans_by_process
):
    processes = true_assignments_by_process.keys()
    trace_acc = {}
    for process in processes:
        for in_span in in_spans_by_process[process]:
            if in_span.trace_id not in trace_acc:
                trace_acc[in_span.trace_id] = True
            true_assignments = true_assignments_by_process[process]
            pred_assignments = pred_assignments_by_process[process]
            for ep in true_assignments.keys():
                if (
                    true_assignments[ep][in_span.GetId()]
                    != pred_assignments[ep][in_span.GetId()]
                ):
                    trace_acc[in_span.trace_id] = False
    correct = sum(trace_acc[tid] for tid in trace_acc)
    return trace_acc, float(correct) / len(trace_acc)

def TopKAccuracyEndToEnd(
    pred_topk_assignments_by_process, true_assignments_by_process, in_spans_by_process
):
    processes = true_assignments_by_process.keys()
    trace_acc = {}
    for i, process in enumerate(processes):
        true_assignments = true_assignments_by_process[process]
        pred_topk_assignments = pred_topk_assignments_by_process[process]
        ep0 = list(true_assignments.keys())[0]
        for x, in_span in enumerate(in_spans_by_process[process]):
            if i!= 0 and trace_acc[in_span.trace_id] == False:
                continue
            if len(pred_topk_assignments[ep0][in_span.GetId()]) < 1:
                trace_acc[in_span.trace_id] = False
                continue
            for j in range(len(pred_topk_assignments[ep0][in_span.GetId()])):
                trace_acc[in_span.trace_id] = True
                for ep in true_assignments.keys():
                    if (
                        true_assignments[ep][in_span.GetId()]
                        != pred_topk_assignments[ep][in_span.GetId()][j]
                    ):
                        trace_acc[in_span.trace_id] = False
                if trace_acc[in_span.trace_id] == True:
                    break
    correct = sum(trace_acc[tid] for tid in trace_acc)
    return trace_acc, float(correct) / len(trace_acc)

def PrintLatency12(trace_acc):
    all_traces = []
    min_time = 1.0e40
    for _, span in all_spans.items():
        if span.IsRoot():
            child_spans = copy.deepcopy(span.children_spans)
            child_spans.sort(key=lambda s: all_spans[s].start_mus)
            processing_delay = all_spans[child_spans[0]].start_mus - span.start_mus
            correct = trace_acc[span.trace_id]
            all_traces.append((span.start_mus, processing_delay, span.trace_id, correct, 1))
            min_time = min(span.start_mus, min_time)
    all_traces.sort()
    for x in all_traces:
        x = x[0] - min_time, x[1], x[2], x[3], x[4]
        print(x)

def BinAccuracyByServiceTimes(method):

    for j in range(4, 5):

        query_latency = {}

        with open("plots/e2e_" + str((j + 1) * 25) + ".pickle", 'rb') as afile:
            e2e_traces = pickle.load(afile)

        true_traces = e2e_traces[method][0]
        pred_traces = e2e_traces[method][1]

        all_traces = []

        for trace in true_traces.items():
            true_trace = true_traces[trace[0]]
            duration = true_trace[1].start_mus - (true_trace[0].start_mus + true_trace[0].duration_mus)
            if pred_traces[trace[0]][1]:
                correct = true_trace[1].sid == pred_traces[trace[0]][1].sid
            else:
                correct = False
            all_traces.append((duration, correct, 1))

        all_traces.sort()
        for i in range(1, len(all_traces)):
            _, c, n = all_traces[i - 1]
            t0, c0, n0 = all_traces[i]
            all_traces[i] = (t0, c + c0, n + n0)
        nbins = 10
        prev_c, prev_n = 0, 0
        accuracy = []
        for b in range(nbins):
            d, c, n = all_traces[int((len(all_traces) * (b + 1)) / nbins - 1)]
            c, n = c - prev_c, n - prev_n
            prev_c, prev_n = prev_c + c, prev_n + n
            percentile = (b + 1) * 100 / nbins
            acc = c / n
            accuracy.append((percentile, acc, d / 1000.0))
        return accuracy

def BinAccuracyByResponseTimes(trace_acc):
    all_traces = []
    for _, span in all_spans.items():
        if span.IsRoot():
            correct = trace_acc[span.trace_id]
            all_traces.append((span.duration_mus, span.trace_id, correct, 1))
    all_traces.sort()
    # accumulate
    for i in range(1, len(all_traces)):
        _, _, c, n = all_traces[i - 1]
        t0, s0, c0, n0 = all_traces[i]
        all_traces[i] = (t0, s0, c + c0, n + n0)
    nbins = 10
    prev_c, prev_n = 0, 0
    accuracy = []
    for b in range(nbins):
        d, _, c, n = all_traces[int((len(all_traces) * (b + 1)) / nbins - 1)]
        c, n = c - prev_c, n - prev_n
        prev_c, prev_n = prev_c + c, prev_n + n
        percentile = (b + 1) * 100 / nbins
        acc = c / n
        if VERBOSE:
            print(
                "Accuracy of %d-percentile bin: %.3f, response_time (ms): %.1f"
                % (percentile, acc, d / 1000.0)
            )
        accuracy.append((percentile, acc, d / 1000.0))
    return accuracy

def ConstructEndToEndTraces(
    pred_assignments_by_process, true_assignments_by_process, in_spans_by_process
):
    def OrderTraces(end_to_end_traces):
        for trace_id in end_to_end_traces:
            end_to_end_traces[trace_id].sort(
                key=lambda x: float('inf') if x is None else x.start_mus
            )

    processes = true_assignments_by_process.keys()
    true_traces = {}
    pred_traces = {}
    for process in processes:
        for in_span in in_spans_by_process[process]:
            if in_span.trace_id not in pred_traces:
                true_traces[in_span.trace_id] = []
                pred_traces[in_span.trace_id] = []
            true_assignments = true_assignments_by_process[process]
            pred_assignments = pred_assignments_by_process[process]
            for ep in true_assignments.keys():
                true_traces[in_span.trace_id].append(
                    all_spans.get(true_assignments[ep][in_span.GetId()])
                )
                options = pred_assignments[ep].get(in_span.GetId(), None)
                if isinstance(options, list):
                    for option in options:
                        pred_traces[in_span.trace_id].append(
                            all_spans.get(option, None)
                        )
                else:
                    pred_traces[in_span.trace_id].append(
                        all_spans.get(options, None)
                    )
    OrderTraces(true_traces)
    OrderTraces(pred_traces)

    return true_traces, pred_traces

def AddCachingEffect(true_assignments, in_span_partitions, out_span_partitions, cache_rate, exponential = False):

    np.random.seed(10)

    def FindSpan(partition, span_id):
        index = -1
        for i, span in enumerate(partition):
            if span.GetId() == span_id:
                index = i
                break

        if index != -1:
            return partition[index]

    def AdjustSpans(in_span_partitions, out_span_partitions, in_span_id, cache_duration_mus, eps, chosen_ep_number):
        trace_id = in_span_id[0]
        for ep in in_span_partitions.keys():
            for span in in_span_partitions[ep]:
                if span.GetId()[0] == trace_id:
                    span.duration_mus -= cache_duration_mus
        for i, ep in enumerate(eps):
            if i > chosen_ep_number:
                for span in out_span_partitions[ep]:
                    if span.GetId()[0] == trace_id:
                        span.start_mus -= cache_duration_mus

    def DeleteSpan(partition, span_id):
        index = -1
        for i, span in enumerate(partition):
            if span.GetId() == span_id:
                index = i
                break

        if index != -1:
            del partition[index]

    eps = GetOutEpsInOrder(out_span_partitions)
    chosen_ep_number = 1
    chosen_ep = eps[chosen_ep_number]

    exponential = True
    if exponential:
        lambda_parameter = 0.001
        in_ep = list(in_span_partitions.keys())[0]
        num_spans = len(in_span_partitions[in_ep])
        samples = np.random.exponential(scale=1/lambda_parameter, size=int(cache_rate * num_spans))
        indices = [int(sample) % num_spans for sample in samples]
        # unique_indices = np.random.choice(num_spans, size=int(cache_rate * num_spans), replace=False, p=np.exp(-lambda_parameter))
        p = np.asarray(np.exp(-lambda_parameter * np.arange(num_spans))).astype('float64')
        p = p / np.sum(p)
        unique_indices = np.random.choice(np.arange(num_spans), size=int(cache_rate * num_spans), replace=False, p=p)
        # print(samples)
        # print(indices)
        # print(sorted(unique_indices))
        # print(len((unique_indices)))
        # input()

    for i, in_span in enumerate(in_spans):
        random_num = random.randint(0, 999)
        # if random_num < (cache_rate * 1000):
        if i in unique_indices:
            for ep in out_span_partitions.keys():
                if ep == chosen_ep:
                    # print("\n Before:\n")
                    # print(in_span)
                    # for ep1 in out_span_partitions.keys():
                    #     print(all_spans[true_assignments[ep1][in_span.GetId()]])
                    span_ID = true_assignments[ep][in_span.GetId()]
                    span = FindSpan(out_span_partitions[ep], span_ID)
                    true_assignments[ep][in_span.GetId()] = ('Skip', 'Skip')
                    AdjustSpans(in_span_partitions, out_span_partitions, in_span.GetId(), span.duration_mus, eps, chosen_ep_number)
                    DeleteSpan(out_span_partitions[ep], span.GetId())
                    # print("\n After:\n")
                    # print(in_span)
                    # for ep1 in out_span_partitions.keys():
                    #     if true_assignments[ep1][in_span.GetId()] in all_spans:
                    #         x = all_spans[true_assignments[ep1][in_span.GetId()]]
                    #     else:
                    #         x = "Skip"
                    #     print(x)
                    # input()
                    break

    return true_assignments

predictors = [
    ("MaxScoreBatchSubsetWithSkips", Timing3(all_spans, all_processes)),
    # ("MaxScoreBatch", Timing2(all_spans, all_processes)),
    # ("MaxScoreBatchParallel", Timing2(all_spans, all_processes)),
    # ("MaxScore", Timing(all_spans, all_processes)),
    ("WAP5_OG", WAP5_OG(all_spans, all_processes)),
    ("FCFS", FCFS(all_spans, all_processes)),
    # ("ArrivalOrder", FCFS2(all_spans, all_processes)),
    # ("VPath", VPATH(all_spans, all_processes)),
]

accuracy_per_process = {}
accuracy_overall = {}
topk_accuracy_overall = {}
accuracy_percentile_bins = {}
traces_overall = {}
cache_updated = False

for method, predictor in predictors:

    random.seed(10)

    if method == "MaxScoreBatch" or method == "MaxScoreBatchSubsetWithSkips":
        confidence_scores_by_process = {}
    if method == "MaxScoreBatchSubset" or method == "MaxScoreBatchParallel" or method == "MaxScoreBatch":
        candidates_per_process = {}

    true_assignments_by_process = {}
    pred_assignments_by_process = {}
    pred_topk_assignments_by_process = {}
    for process_id, process in enumerate(out_spans_by_process.keys()):
        in_spans = copy.deepcopy(in_spans_by_process[process])
        out_spans = copy.deepcopy(out_spans_by_process[process])

        if len(out_spans) == 0:
            continue

        # partition spans by the other endpoint
        def PartitionSpansByEndPoint(spans, endpoint_lambda):
            partitions = {}
            for span in spans:
                ep = endpoint_lambda(span)
                if ep not in partitions:
                    partitions[ep] = []
                partitions[ep].append(span)
            for ep, part in partitions.items():
                part.sort(key=lambda x: x.start_mus)
            return partitions

        print("Process: ", process)
        # partition spans by subservice at the other end
        in_span_partitions = PartitionSpansByEndPoint(
            in_spans, lambda x: x.GetParentProcess()
        )
        print("Incoming span partitions: ", process, in_span_partitions.keys())
        out_span_partitions = PartitionSpansByEndPoint(
            out_spans, lambda x: x.GetChildProcess()
        )
        print("Outgoing span partitions: ", process, out_span_partitions.keys())

        if len(in_span_partitions.keys()) > 1:
            print("SKIPPING THIS PROCESS:", process)
            continue

        if sys.argv[3] == "parallel" or method == "MaxScoreBatchParallel":
            parallel = True
        else:
            parallel = False

        # if process == "frontend":
        #     continue

        if sys.argv[4] == "instrumented" and process == "search":
            print(process)
            instrumented_hops = []
        else:
            instrumented_hops = []

        true_assignments = GetGroundTruth(in_span_partitions, out_span_partitions)

        copy_x = copy.deepcopy(in_span_partitions)
        copy_y = copy.deepcopy(out_span_partitions)
        copy_z = copy.deepcopy(true_assignments)

        invocation_graph = FindOrder(in_span_partitions, out_span_partitions, true_assignments)

        # repeats = int(sys.argv[7])
        # factor = int(sys.argv[8])
        # in_span_partitions, out_span_partitions = repeatChangeSpans(in_span_partitions, out_span_partitions, repeats=repeats, factor=factor)
        # true_assignments = GetGroundTruth(in_span_partitions, out_span_partitions)

        if process == "frontend" and (method != "MaxScoreBatch" or method != "MaxScoreBatchParallel" or method !=  "FCFS" or method !=  "ArrivalOrder"):
            print("cache %: ", float(sys.argv[5]) * 100)
            true_assignments = AddCachingEffect(true_assignments, in_span_partitions, out_span_partitions, cache_rate=float(sys.argv[5]))

        # print(bool(DeepDiff(copy_x, in_span_partitions)))
        # print(bool(DeepDiff(copy_y, out_span_partitions)))
        # print(bool(DeepDiff(copy_z, true_assignments)))

        # in_ep, _ = list(in_span_partitions.items())[0]
        # for in_span in in_span_partitions[in_ep]:
        #     print(in_span)
        #     input()
        #     for out_ep in out_span_partitions.keys():
        #         out_span_id = true_assignments[out_ep][in_span.GetId()]
        #         print(all_spans[out_span_id])
        #         input()

        if method == "MaxScoreBatch" or method == "MaxScoreBatchParallel":
            if process == "service1":
                parallel = True
            pred_assignments, not_best_count, num_spans, per_span_candidates = predictor.FindAssignments(
                process, in_span_partitions, out_span_partitions, parallel, instrumented_hops, true_assignments
            )
        elif method == "MaxScoreBatchSubset":
            if process == "service1":
                parallel = True
            pred_assignments, not_best_count, num_spans, per_span_candidates = predictor.FindAssignments(
                process, in_span_partitions, out_span_partitions, parallel, instrumented_hops, true_assignments
            )
        elif method == "MaxScore":
            if process == "service1":
                parallel = True
            pred_assignments = predictor.FindAssignments(
                process, in_span_partitions, out_span_partitions, True, instrumented_hops, true_assignments
            )
        elif method == "MaxScoreBatchSubsetWithSkips":
            if process == "service1":
                parallel = True
            pred_assignments, pred_topk_assignments, not_best_count, num_spans, per_span_candidates = predictor.FindAssignments(
                process, in_span_partitions, out_span_partitions, parallel, instrumented_hops, true_assignments, invocation_graph
            )
        elif method == "MaxScoreBatchSubsetWithTrueSkips":
            pred_assignments, pred_topk_assignments, not_best_count, num_spans, per_span_candidates = predictor.FindAssignments(
                process, in_span_partitions, out_span_partitions, parallel, instrumented_hops, true_assignments, invocation_graph, True, False
            )
        elif method == "MaxScoreBatchSubsetWithTrueDist":
            pred_assignments, pred_topk_assignments, not_best_count, num_spans, per_span_candidates = predictor.FindAssignments(
                process, in_span_partitions, out_span_partitions, parallel, instrumented_hops, true_assignments, invocation_graph, False, True
            )
        else:
            pred_assignments = predictor.FindAssignments(
                process, in_span_partitions, out_span_partitions, parallel, instrumented_hops, true_assignments
            )

        acc = AccuracyForService(pred_assignments, true_assignments, in_span_partitions)
        print("Accuracy for service %s: %.3f%%\n" % (process, acc * 100))
        if method == "MaxScoreBatchSubsetWithSkips":
            acc2 = TopKAccuracyForService(pred_topk_assignments, true_assignments, in_span_partitions)
            print("Top K accuracy for service %s: %.3f%%\n" % (process, acc2 * 100))
        true_assignments_by_process[process] = true_assignments
        pred_assignments_by_process[process] = pred_assignments
        if method == "MaxScoreBatchSubsetWithSkips":
            pred_topk_assignments_by_process[process] = pred_topk_assignments

        accuracy_per_process[(method, process_id)] = acc

        if method == "MaxScoreBatch" or method == "MaxScoreBatchSubsetWithSkips":
            print(not_best_count, num_spans)
            confidence_scores_by_process[process] = [acc, not_best_count, num_spans]

        if method == "MaxScoreBatchSubset" or method == "MaxScoreBatchParallel":
            candidates_per_process[process] = per_span_candidates

        # new_count = 0
        # max_count = 0
        # values1 = []
        # values2 = []
        # top5 = []
        # for key, value in per_span_candidates.items():
        #     if process == "init-service":
        #         break
        #     # print(process)
        #     # print(key, value)
        #     if value > max_count and bool(AccuracyForSpan(pred_assignments, true_assignments, key)):
        #         max_count = value
        #     if bool(AccuracyForSpan(pred_assignments, true_assignments, key)):
        #         values1.append(value)
        #     else:
        #         values2.append((key, value))
        #         top5.append(bool(TopKAccuracyForSpan(pred_topk_assignments, true_assignments, key)))
        #     # print(bool(AccuracyForSpan(pred_assignments, true_assignments, key)))
        #     # input()
        #     # new_count += 1
        #     # if new_count == 20:
        #     #     break
        # # print(values2)
        # # print(top5)
        # # input()

    trace_acc, acc_e2e = AccuracyEndToEnd(
        pred_assignments_by_process, true_assignments_by_process, in_spans_by_process
    )
    if method == "MaxScoreBatchSubsetWithSkips":
        trace_acc_2, acc_e2e_2 = TopKAccuracyEndToEnd(
            pred_topk_assignments_by_process, true_assignments_by_process, in_spans_by_process
        )
    true_traces_e2e, pred_traces_e2e = ConstructEndToEndTraces(
        pred_assignments_by_process, true_assignments_by_process, in_spans_by_process
    )
    traces_overall[method] = [true_traces_e2e, pred_traces_e2e]

    print("End-to-end accuracy for method %s: %.3f%%\n\n" % (method, acc_e2e * 100))
    if method == "MaxScoreBatchSubsetWithSkips":
        print("End-to-end top K accuracy for method %s: %.3f%%\n\n" % (method, acc_e2e_2 * 100))
    accuracy_overall[method] = acc_e2e
    if method == "MaxScoreBatchSubsetWithSkips":
        accuracy_overall[method + "TopK"] = acc_e2e_2
    accuracy_percentile_bins[method] = BinAccuracyByResponseTimes(trace_acc)

load_level = sys.argv[2]
name = sys.argv[3]
cache = sys.argv[5]

for key in accuracy_overall.keys():
    print("End-to-end accuracy for method: ", key, accuracy_overall[key])

with open('plots/bin_acc' + "_" + str(load_level) + "_" + name + "_" + cache + '.pickle', 'wb') as handle:
    pickle.dump(accuracy_percentile_bins, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open('plots/accuracy' + "_" + str(load_level) + "_" + name + "_" + cache + '.pickle', 'wb') as handle:
    pickle.dump(accuracy_overall, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open('plots/e2e' + "_" + str(load_level) + "_" + name + "_" + cache + '.pickle', 'wb') as handle:
    pickle.dump(traces_overall, handle, protocol = pickle.HIGHEST_PROTOCOL)
# with open('plots/confidence_scores' + "_" + str(load_level) + "_" + name + "_" + cache + '.pickle', 'wb') as handle:
#     pickle.dump(confidence_scores_by_process, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open('plots/process_acc' + "_" + str(load_level) + "_" + name + "_" + cache + '.pickle', 'wb') as handle:
    pickle.dump(accuracy_per_process, handle, protocol = pickle.HIGHEST_PROTOCOL)
# with open('plots/candidates' + "_" + str(load_level) + "_" + name + "_" + cache + '.pickle', 'wb') as handle:
#     pickle.dump(candidates_per_process, handle, protocol = pickle.HIGHEST_PROTOCOL)

# sampleQuery()

# x = {}
# for method, predictor in predictors:
#     x[method] = BinAccuracyByServiceTimes(method)

# with open('plots/bin_acc_per_service_time_' + str(load_level) + '_service12_version3.pickle', 'wb') as handle:
#     pickle.dump(x, handle, protocol = pickle.HIGHEST_PROTOCOL)
