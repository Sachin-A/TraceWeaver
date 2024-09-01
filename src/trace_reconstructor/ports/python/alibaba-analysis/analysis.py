import json
import os
import shutil
import sys
import tarfile

import matplotlib.pyplot as plt
from algorithms.fcfs import FCFS
from algorithms.arrival_order import ArrivalOrder
from algorithms.traceweaver_v1 import TraceWeaverV1
from algorithms.traceweaver_v2 import TraceWeaverV2

def uncompress(output_path, source_file):
    tarfile_object = tarfile.open(source_file)
    tarfile_object.extractall(output_path)
    tarfile_object.close()

all_traces = {}
all_cgs = {}

class Trace():
    def __init__(
        self,
        root_span_id,
        spans,
        cg_signature,
        processes
    ):
        self.root_span_id = root_span_id
        self.spans = spans
        self.cg_signature = cg_signature
        self.processes = processes

class Span(object):
    def __init__(
        self,
        trace_id,
        sid,
        start_mus,
        duration_mus,
        caller,
        callee,
        # op_name,
        references,
        process_id,
        span_kind,
    ):
        self.sid = sid
        self.trace_id = trace_id
        self.start_mus = start_mus
        self.duration_mus = duration_mus
        self.caller = caller
        self.callee = callee
        # self.op_name = op_name
        self.references = references
        self.process_id = process_id
        self.span_kind = span_kind
        self.children_spans = []

    def AddChild(self, child_span_id):
        self.children_spans.append(child_span_id)

    # def GetChildProcess(self):
    #     assert self.span_kind == "client"
    #     assert len(self.children_spans) == 1
    #     return all_processes[self.trace_id][
    #         all_spans[self.children_spans[0]].process_id
    #     ]

    # def GetParentProcess(self):
    #     if self.IsRoot():
    #         # return "client_" + self.op_name
    #         return "client_request"
    #     assert len(self.references) == 1
    #     parent_span_id = self.references[0]
    #     return all_processes[self.trace_id][all_spans[parent_span_id].process_id]

    def GetId(self):
        return (self.trace_id, self.sid)

    def IsRoot(self):
        return len(self.references) == 0

    def __lt__(self, other):
        return self.start_mus < other.start_mus

    def __repr__(self):
        return "Span:(%s, %d, %d, %s)" % (
            # self.op_name,
            self.process_id,
            self.start_mus,
            self.duration_mus,
            self.span_kind
        )

    def __str__(self):
        return self.__repr__()

def AssignCGSignature(spans, root_span_id):

    processes = {}
    max_depth = 0
    def ExploreSubTree(span_id, depth):
        nonlocal max_depth
        if depth > max_depth:
            max_depth = depth
        span = spans[span_id]
        if span.span_kind == "server":
            if depth not in processes:
                processes[depth] = []
            if depth == 0:
                processes[depth].append(span.caller)
            processes[depth].append(span.process_id)
            for child in span.children_spans:
                ExploreSubTree(child, depth + 1)

    ExploreSubTree(root_span_id, 0)

    cg_key = ''
    for d in range(0, max_depth + 1):
        processes[d].sort()
        for process in processes[d]:
            cg_key += process

    cg_hash = hash(cg_key) % ((sys.maxsize + 1) * 2)
    return cg_hash

def GetAllTracesInDir(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(directory + "/" + f)]
    files = [f for f in files if f.endswith("json")]
    full_path = os.path.abspath(directory)
    files = [full_path + "/" + f for f in files]
    return files

def ParseSpansJson(spans_json):
    spans = {}
    for span in spans_json:
        span_kind = None
        for tag in span["tags"]:
            if tag["key"] == "span.kind":
                span_kind = tag["value"]
        if span_kind == "server":
            references = []
            for ref in span["references"]:
                if bool(ref):
                    references.append((ref["traceID"], ref["spanID"]))
            trace_id = span["traceID"]
            sid = span["spanID"]
            span_id = (trace_id, sid)
            start_mus = span["startTime"]
            duration_mus = span["duration"]
            caller = span["caller"]
            callee = span["callee"]
            # op_name = span.get("operationName", None)
            process_id = span["processID"]
            spans[span_id] = Span(
                trace_id,
                sid,
                start_mus,
                duration_mus,
                caller,
                callee,
                # op_name,
                references,
                process_id,
                span_kind,
            )
    return spans

def ParseProcessessJson(spans):
    processes = set()
    for span in spans:
        processes.add(span["processID"])
    return list(processes)

def ParseJsonTrace(trace_json):
    ret = []
    processes = []
    with open(trace_json, "r") as tfile:
        json_data = json.load(tfile)
        json_data = json_data["data"]
        for d in json_data:
            trace_id = d["traceID"]
            spans = ParseSpansJson(d["spans"])
            processes = ParseProcessessJson(d["spans"])

            root_service = None
            for span_id, span in spans.items():
                # no references
                if len(span.references) == 0:
                    root_service = "client_request"
            if root_service is not None:
                ret.append((trace_id, spans))
    assert len(ret) == 1
    trace_id, spans = ret[0]
    return trace_id, spans, processes

def ProcessTraceData(data):
    trace_id, spans, processes = data
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

    return spans, root_span_id, trace_id, processes

def FindUniqueCGs():
    global all_traces
    unique_cgs = {}
    for k, v in all_traces.items():
        if v.cg_signature not in unique_cgs:
            unique_cgs[v.cg_signature] = []
        unique_cgs[v.cg_signature].append(k)
    return unique_cgs

def PrintUniqueCGs(unique_cgs):
    num_members = []
    for k, v in unique_cgs.items():
        num_members.append(len(v))

    plt.hist(num_members, bins = 200)
    plt.plot()
    plt.yscale('log')
    plt.savefig("/scratch/sachina3/projects/clusterdata/cluster-trace-microservices-v2021/data/analysis/unique-cgs-distribution-dataset-" + str(dataset_id) + ".svg")

def PreparePerCGData(trace_ids):

    in_spans_by_process = dict()
    out_spans_by_process = dict()
    all_spans = dict()
    all_processes = dict()

    def AddSpanToProcess(span):
        nonlocal in_spans_by_process
        nonlocal out_spans_by_process

        caller = span.caller
        if caller not in out_spans_by_process:
            out_spans_by_process[caller] = []
        out_spans_by_process[caller].append(span)

        callee = span.callee
        if callee not in in_spans_by_process:
            in_spans_by_process[callee] = []
        in_spans_by_process[callee].append(span)

    for trace_id in trace_ids:
        trace = all_traces[trace_id]
        root_span_id = trace.root_span_id
        spans = trace.spans

        for span_id, span in spans.items():
            AddSpanToProcess(span)

        all_spans.update(spans)
        all_processes[trace_id] = trace.processes

    return in_spans_by_process, out_spans_by_process, all_spans, all_processes

num_shards = 0
path = '/scratch/sachina3/projects/clusterdata/cluster-trace-microservices-v2021/data/MSCallGraph/'

for dataset_id in range(0, num_shards + 1):

    uncompress(path + "traces" + str(dataset_id) + "/", path + "traces" + str(dataset_id) + ".tar.lama")
    traces = GetAllTracesInDir(path + "traces" + str(dataset_id) + "/")
    traces.sort()

    for i, trace in enumerate(traces):
        if i % 10000 == 0:
            print("Loading: ", i)

        data = ParseJsonTrace(trace)
        spans, root_span_id, trace_id, processes = ProcessTraceData(data)
        cg_signature = AssignCGSignature(spans, root_span_id)
        all_traces[trace_id] = Trace(root_span_id, spans, cg_signature, processes)

    shutil.rmtree(path + 'traces' + str(dataset_id) + '/')

all_cgs = FindUniqueCGs()

for k, v in all_cgs.items():

    in_spans_by_process, out_spans_by_process, all_spans, all_processes = PreparePerCGData(v)
