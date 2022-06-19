import json
import sys
import os
from fcfs import FCFS
from timing import Timing
from timing2 import Timing2

VERBOSE = False

all_spans = dict()
all_processes = dict()


class Span(object):
    def __init__(
        self,
        trace_id,
        span_id,
        start_mus,
        duration_mus,
        op_name,
        references,
        process_id,
        span_kind,
    ):
        self.span_id = span_id
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
        if len(self.references) == 0:  # root
            return "client"
        assert len(self.references) == 1
        parent_span_id, _ = self.references[0]
        return all_processes[self.trace_id][all_spans[parent_span_id].process_id]

    def GetId(self):
        return (self.trace_id, self.span_id)

    def __repr__(self):
        return "Span:(%s, %d, %d, %s)" % (
            self.op_name,
            self.start_mus,
            self.duration_mus,
            self.span_kind,
        )

    def __str__(self):
        return self.__repr__()


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
            references.append((ref["spanID"], ref["traceID"]))
        trace_id = span["traceID"]
        span_id = span["spanID"]
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
            span_id,
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


def ParseJsonTrace(trace_json):
    ret = []
    processes = None
    with open(trace_json, "r") as tfile:
        json_data = json.load(tfile)
        json_data = json_data["data"]
        for d in json_data:
            trace_id = d["traceID"]
            spans = ParseSpansJson(d["spans"])
            processes = ParseProcessessJson(d["processes"])

            root_service = None
            for span_id, span in spans.items():
                # no references
                if len(span.references) == 0:
                    root_service = span.op_name
            if root_service is not None:
                ret.append((trace_id, spans))
    assert len(ret) == 1
    trace_id, spans = ret[0]
    return trace_id, spans, processes


in_spans_by_process = dict()
out_spans_by_process = dict()


def ProcessTraceData(data):
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
        for par_sid, par_tid in span.references:
            spans[par_sid].AddChild(span_id)
    for span_id, span in spans.items():
        span.children_spans.sort(
            key=lambda child_span_id: spans[child_span_id].start_mus
        )

    def ExploreSubTree(span_id, depth):
        span = spans[span_id]
        AddSpanToProcess(span_id)
        if VERBOSE:
            print(
                (4 * depth) * " ",
                span.span_id,
                span.op_name,
                span.start_mus,
                span.duration_mus,
            )
        for child in span.children_spans:
            ExploreSubTree(child, depth + 1)

    # comment out if condition to consider all microservice kinds
    if spans[root_span_id].op_name == "HTTP GET /hotels":
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
    if cnt > 1000:  # 10000:
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
    return float(correct) / len(trace_acc)


#predictor = FCFS(all_spans, all_processes)
#predictor = Timing(all_spans, all_processes)
predictor = Timing2(all_spans, all_processes)

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
    print("\n")

    true_assignments = GetGroundTruth(in_span_partitions, out_span_partitions)
    pred_assignments = predictor.FindAssignments(
        process, in_span_partitions, out_span_partitions
    )
    acc = AccuracyForService(pred_assignments, true_assignments, in_span_partitions)
    print("Accuracy for service %s: %.3f" % (process, acc))
    true_assignments_by_process[process] = true_assignments
    pred_assignments_by_process[process] = pred_assignments

acc_e2e = AccuracyEndToEnd(
    pred_assignments_by_process, true_assignments_by_process, in_spans_by_process
)
print("End-to-end accuracy: %.3f" % acc_e2e)
