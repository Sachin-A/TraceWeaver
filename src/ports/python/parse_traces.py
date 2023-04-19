import os
import sys
import json
import math
import pickle5 as pickle
from algorithms.fcfs import FCFS
from algorithms.fcfs2 import FCFS2
from algorithms.timing import Timing
from algorithms.timing2 import Timing2
from algorithms.wap5_og import WAP5_OG
from algorithms.vpath import VPATH
import math
import copy

VERBOSE = False
random.seed(10)

all_spans = dict()
all_processes = dict()

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
    ):
        self.sid = sid
        self.trace_id = trace_id
        self.start_mus = start_mus
        self.duration_mus = duration_mus
        self.op_name = op_name
        self.references = references
        self.process_id = process_id
        self.span_kind = span_kind
        self.children_spans = []

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
        return "Span:(%s, %d, %d, %s)" % (
            self.op_name,
            self.start_mus,
            self.duration_mus,
            self.span_kind,
        )

    def __str__(self):
        return self.__repr__()

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

'''
FOR all e2e requests
WHICH
    were in the top 5% response latency bracket AND
    were initiated after time X,
FIND
    the worst performing service AND
    its mean service latency for these requests
'''
#maindir = "152"
maindir = "10-1"
def sampleQuery():
    rrange = range(4, 5)
    if "25-1" in maindir or "10-1" in maindir:
        rrange = range(0, 1)
    for j in rrange:

        query_latency = {}

        filename = "plots/vipul/" + maindir + "/e2e_" + str((j + 1) * 25) + ".pickle"
        if "153" in sys.argv[1]:
            filename = "plots/vipul/153/e2e_" + str((j + 1) * 25) + ".pickle"
        #elif "25-1" in sys.argv[1]:
        #    filename = "plots/vipul/25-1/e2e_" + str((j + 1) * 25) + ".pickle"

        with open(filename, 'rb') as afile:
            e2e_traces = pickle.load(afile)

        for method in e2e_traces.keys():

            true_traces = e2e_traces[method][0]

            true_traces = list(
                sorted(
                    true_traces.items(),
                    key=lambda x: x[1][0].start_mus
                )
            )
            percentile = 98
            nreq = 0
            nreq2 = nreq + 500

            #p1 = int(0.80 * len(true_traces))
            start_time = true_traces[nreq][1][0].start_mus
            end_time = true_traces[nreq2][1][0].start_mus

            def FilterSpan(span, all_spans=False):
                return (
                    (hash(span.trace_id) % 1 == 0 or all_spans) and
                    span.start_mus > start_time and span.start_mus < end_time
                )

            # true_traces = list(
            #     filter(
            #         lambda x: x[1][0].start_mus > start_time,
            #         list(true_traces.items())[p1:]
            #     )
            # )
            if "all_spans" not in sys.argv[-1]:
                true_traces = list(
                    filter(
                        lambda x: FilterSpan(x[1][0]),
                        true_traces,
                    )
                )
                print(len(true_traces))
                true_traces.sort(
                        key=lambda x: x[1][-1].start_mus + x[1][-1].duration_mus - x[1][0].start_mus
                    )
                p1 = int(percentile * float(len(true_traces)/100))
                true_traces = true_traces[p1:]
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


            else:
                latency_per_service_true = [[] for i in range(5)]
                latency_per_service_pred = [[] for i in range(5)]
                for i in range(5):
                    traces_i = list(
                        filter(
                            lambda x: FilterSpan(x[1][i], all_spans=True),
                            true_traces
                            #list(true_traces.items())[p1:]
                        )
                    )
                    traces_i.sort(
                        key=lambda x: x[1][i].duration_mus
                    )
                    p1 = int(percentile * float(len(traces_i)/100))
                    print("p1", p1, len(traces_i))
                    traces_i = traces_i[p1:]
                    #print(traces_i)
                    for _, trace in traces_i:
                        span = trace[i]
                        latency_per_service_true[i].append((span.trace_id, span.sid, span.start_mus, span.duration_mus))
                        latency_per_service_pred[i].append((span.trace_id, span.sid, span.start_mus, span.duration_mus))

            query_latency[method] = [latency_per_service_true, latency_per_service_pred]
        load_level = (j + 1) * 25
        nreq = "_na_"
        percentile = "_na_"
        if "all_spans" in sys.argv[-1]:
            with open('plots/vipul/query_latency_' + str(load_level) + '_all_version2_before' + str(nreq) + '_p' + str(percentile) + '.pickle', 'wb') as handle:
                pickle.dump(query_latency, handle, protocol = pickle.HIGHEST_PROTOCOL)
        else:
            with open('plots/vipul/query_latency_' + str(load_level) + '_version2_before' + str(nreq) + '_p' + str(percentile) + '.pickle', 'wb') as handle:
                pickle.dump(query_latency, handle, protocol = pickle.HIGHEST_PROTOCOL)


def GetAllTracesInDir(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(directory + "/" + f)]
    files = [f for f in files if f.endswith("json")]
    full_path = os.path.abspath(directory)
    files = [full_path + "/" + f for f in files]
    return files


def ParseSpansJson(spans_json):
    spans = {}
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
        spans[span_id] = Span(
            trace_id,
            sid,
            start_mus,
            duration_mus,
            op_name,
            references,
            process_id,
            span_kind,
        )
    return spans


def ParseProcessessJson(processes_json):
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

def ParseInputPickle(filename):
    with open(filename, 'rb') as pfile:
        data = pickle.load(pfile)
        print (len(data.keys()))
        for k in data.keys():
            print(k, type(data[k]))
            for t in data[k]:
                print("new entry")
                for x in t:
                    print(x)
            sys.exit()


input_pickle_file = sys.argv[1]
ParseInputPickle(input_pickle_file)

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


def AccuracyForService(pred_assignments, true_assignments, in_span_partitions):
    assert len(in_span_partitions) == 1
    _, in_spans = list(in_span_partitions.items())[0]
    cnt = 0
    for in_span in in_spans:
        correct = True
        for ep in true_assignments.keys():
            correct = correct and (
                pred_assignments[ep][in_span.GetId()]
                == true_assignments[ep][in_span.GetId()]
            )
        cnt += int(correct)
    return float(cnt) / len(in_spans)

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
                pred_traces[in_span.trace_id].append(
                    all_spans.get(pred_assignments[ep][in_span.GetId()])
                )
    OrderTraces(true_traces)
    OrderTraces(pred_traces)

    return true_traces, pred_traces

predictors = [
    ("Greedy++", Timing2(all_spans, all_processes)),
    #("Greedy", Timing(all_spans, all_processes)),
    #("FCFS", FCFS(all_spans, all_processes)),
    #("FCFS++", FCFS2(all_spans, all_processes)),
]

accuracy_overall = {}
accuracy_percentile_bins = {}
traces_overall = {}
for method, predictor in predictors:

    true_assignments_by_process = {}
    pred_assignments_by_process = {}
    for process in out_spans_by_process.keys():
        in_spans = in_spans_by_process[process]
        out_spans = out_spans_by_process[process]

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

         # partition spans by subservice at the other end
        in_span_partitions = PartitionSpansByEndPoint(
            in_spans, lambda x: x.GetParentProcess()
        )
        print("Incoming span partitions", process, in_span_partitions.keys())
        out_span_partitions = PartitionSpansByEndPoint(
            out_spans, lambda x: x.GetChildProcess()
        )
        print("Outgoing span partitions", process, out_span_partitions.keys())

        true_assignments = GetGroundTruth(in_span_partitions, out_span_partitions)
        invocation_graph = FindOrder(in_span_partitions, out_span_partitions, true_assignments)
        pred_assignments = predictor.FindAssignments(
            process, in_span_partitions, out_span_partitions, invocation_graph
        )
        acc = AccuracyForService(pred_assignments, true_assignments, in_span_partitions)
        print("Accuracy for service %s: %.3f\n" % (process, acc))
        true_assignments_by_process[process] = true_assignments
        pred_assignments_by_process[process] = pred_assignments

    trace_acc, acc_e2e = AccuracyEndToEnd(
        pred_assignments_by_process, true_assignments_by_process, in_spans_by_process
    )
    true_traces_e2e, pred_traces_e2e = ConstructEndToEndTraces(
        pred_assignments_by_process, true_assignments_by_process, in_spans_by_process
    )
    traces_overall[method] = [true_traces_e2e, pred_traces_e2e]

    print("End-to-end accuracy for method %s: %.3f\n\n" % (method, acc_e2e))
    accuracy_overall[method] = acc_e2e
    accuracy_percentile_bins[method] = BinAccuracyByResponseTimes(trace_acc)
#'''
'''
load_level = sys.argv[2]
with open('plots/vipul/' + maindir + '/e2e_' + str(load_level) + '.pickle', 'wb') as handle:
    pickle.dump(traces_overall, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open('plots/vipul/' + maindir + '/bin_acc_' + str(load_level) + '.pickle', 'wb') as handle:
    pickle.dump(accuracy_percentile_bins, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open('plots/vipul/' + maindir + '/accuracy_' + str(load_level) + '.pickle', 'wb') as handle:
    pickle.dump(accuracy_overall, handle, protocol = pickle.HIGHEST_PROTOCOL)
'''
#sampleQuery()
