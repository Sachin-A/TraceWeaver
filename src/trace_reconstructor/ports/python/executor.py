import argparse
import concurrent.futures
import copy
import json
import math
import os
import pickle
import random
import shutil
import string
import sys
import time
from pathlib import Path

import _pickle as cPickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from deepdiff import DeepDiff
from matplotlib.table import Table
from scipy import stats

import config
import helpers.misc as misc
import helpers.transforms as transforms
import helpers.utils as utils
from algorithms.arrival_order import ArrivalOrder
from algorithms.fcfs import FCFS
from algorithms.timing import Timing
from algorithms.timing2 import Timing2
from algorithms.timing3 import Timing3
from algorithms.vpath import vPath
from algorithms.vpath_old import vPathOld
from algorithms.wap5 import WAP5
from spans import Span

cg_booleans = []

parser = argparse.ArgumentParser(description='Map incoming and outgoing spans at each service.')
parser.add_argument('--relative_path', type=ascii, required=False, default=None,
                    help='relative location for directory with Jaeger-style spans')
parser.add_argument('--absolute_path', type=ascii, required=False, default=None,
                    help='absolute location for directory with Jaeger-style spans')
parser.add_argument('--compressed', type=int, required=False, default=0, choices=[0, 1],
                    help='is directory compressed?')
# parser.add_argument('--gather_data', type=int, required=True,
#                     help='is the data in multiple locations?')
# parser.add_argument('--call_graph_id', type=int, required=True,
#                     help='if dataset contains multiple CGs provide the CG ID, else pass -1')
parser.add_argument('--load_level', type=int, required=False, default=0,
                    help='provide load level if static test')
parser.add_argument('--test_name', type=ascii, required=False, default="test",
                    help='custom name for tracing test')
parser.add_argument('--parallel', type=int, required=False, default=0, choices=[0, 1],
                    help='treat sibling relationships as parallel?')
parser.add_argument('--instrumented', type=int, required=False, default=0, choices=[0, 1],
                    help='treat some hops as instrumented?')
parser.add_argument('--cache_rate', type=float, required=True, default=0,
                    help='rate of artificial caching to apply if needed')
parser.add_argument('--fix', type=int, required=True, default=0,
                    help='do spans require format fixing?')
parser.add_argument('--repeat_factor', type=int, required=False, default=1,
                    help='factor by which spans are duplicated to increase dataset size')
parser.add_argument('--compress_factor', type=float, required=False, default=1,
                    help='factor by which to reduce spacing between adjacent spans')
parser.add_argument('--execute_parallel', type=int, required=False, default=1,
                    help='should each service tracing be executed in parallel?')
parser.add_argument('--results_directory', type=ascii, required=True, default=None,
                    help='directory to store results')
parser.add_argument('--clear_cache', type=int, required=False, default=0,
                    help='clear cache of processed, time-ordered file names')
parser.add_argument('--predictor_indices', type=str, required=False, default='',
                    help='Comma-separated list of indices of algorithms to run')
args = parser.parse_args()

if args.relative_path is None and args.absolute_path is None:
   parser.error("At least one of --relative_path and --absolute_path is required")

PROJECT_ROOT = misc.get_project_root()
if args.relative_path:
    RELATIVE_PATH = os.path.join(PROJECT_ROOT, args.relative_path.strip('\''))
if args.absolute_path:
    ABSOLUTE_PATH = args.absolute_path.strip('\'')
COMPRESSED = bool(args.compressed)
# GATHER = bool(args.gather_data)
# CG_ID = args.call_graph_id
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots/")
LOAD_LEVEL = args.load_level
TEST_NAME = args.test_name.strip('\'')
PARALLEL = bool(args.parallel)
INSTRUMENTED = bool(args.instrumented)
CACHE_RATE = args.cache_rate
FIX = args.fix
REPEAT_FACTOR = args.repeat_factor
COMPRESS_FACTOR = args.compress_factor
EXECUTE_PARALLEL = args.execute_parallel
RESULTS_DIR = args.results_directory.strip('\'')
CLEAR_CACHE = bool(args.clear_cache)

try:
    PREDICTOR_INDICES = list(map(int, args.predictor_indices.split(',')))
except ValueError as e:
    print(f"Error converting predictor indices: {e}")
    sys.exit(1)

random.seed(10)
# np.seterr(all='raise')

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

def CalculateRate(spans):
    rates = []
    window = 10
    for i in range(len(spans) - window):
        time_diff = (spans[i + window].start_mus - spans[i].start_mus) / 1000000
        if time_diff < 0:
            raise ValueError("Time difference should be 0 or positive.")
        rates.append(window / (time_diff + 0.001))

    return np.percentile(rates, 50)

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

def FindConstraintsUsingFit(in_span_partitions, out_span_partitions, gt_invocation_graph, args):
    global cg_booleans

    predictor = Timing3(all_spans, all_processes)

    def test_fit_dag(candidate_invocation_graph):
        unassigned = predictor.FindAssignments(
                        args[0], in_span_partitions, out_span_partitions,
                        args[1], args[2], args[3], candidate_invocation_graph
                    )[-1]
        return unassigned

    candidate_invocation_graph = nx.DiGraph()

    # candidate_invocation_graph.add_nodes_from(in_span_partitions.keys())

    candidate_invocation_graph.add_nodes_from(out_span_partitions.keys())
    num_eps = len(out_span_partitions.keys())

    # Initialize the best score and best graph
    best_score = 0
    best_graph = candidate_invocation_graph.copy()
    candidate_count = 0

    # Test the fit of the initial DAG
    candidate_count += 1
    print("Trying edge count:", candidate_count, "out of (at most)", num_eps * (num_eps - 1) + 1, "edges.")
    unassigned = test_fit_dag(candidate_invocation_graph)
    print("Unassigned:", unassigned, nx.utils.graphs_equal(candidate_invocation_graph, gt_invocation_graph))

    for osp_1 in out_span_partitions.keys():
        for osp_2 in out_span_partitions.keys():
            if osp_1 != osp_2:
                candidate_count += 1
                if candidate_invocation_graph.has_edge(osp_2, osp_1) == False:

                    # Add the edge
                    candidate_invocation_graph.add_edge(osp_1, osp_2)

                    # Test the fit of the current DAG
                    print("Trying edge count:", candidate_count, "out of (at most)", num_eps * (num_eps - 1) + 1, "edges.")
                    unassigned = test_fit_dag(candidate_invocation_graph)
                    print("Unassigned:", unassigned, nx.utils.graphs_equal(candidate_invocation_graph, gt_invocation_graph))

                    # Keep edge if every span is still satisfied
                    if unassigned <= 0:
                        best_graph = candidate_invocation_graph.copy()
                        best_score = unassigned
                    # Otherwise, remove the edge
                    else:
                        candidate_invocation_graph.remove_edge(osp_1, osp_2)

    # Check if the candidate graph is identical to the ground truth graph
    cg_boolean = nx.utils.graphs_equal(best_graph, gt_invocation_graph)
    cg_booleans.append(cg_boolean)
    if cg_boolean:
        print("The candidate graph is identical to the ground truth graph.")

    # nx.draw(best_graph, with_labels=True, font_weight='bold')
    # plt.show()
    return best_graph

def FindOrder(in_span_partitions, out_span_partitions, true_assignments):
    assert len(in_span_partitions) == 1

    ep_in, in_spans = list(in_span_partitions.items())[0]
    order = set()
    out_eps = list(out_span_partitions.keys())
    G = nx.DiGraph()
    G1 = nx.DiGraph()
    G2 = nx.DiGraph()
    for i in range(len(out_eps)):
        G.add_node(i)
        G1.add_node(out_eps[i])
        G2.add_node(out_eps[i] + "-start")
        G2.add_node(out_eps[i] + "-end")
    for i in range(len(out_eps)):
        for j in range(len(out_eps)):
            if i != j:
                G.add_edge(i, j)
                G1.add_edge(out_eps[i], out_eps[j])
                G2.add_edge(out_eps[i] + "-start", out_eps[j] + "-start")
                G2.add_edge(out_eps[i] + "-start", out_eps[j] + "-end")
                G2.add_edge(out_eps[i] + "-end", out_eps[j] + "-start")
                G2.add_edge(out_eps[i] + "-end", out_eps[j] + "-end")

    for in_span in in_spans:
        outgoing_spans = []
        outgoing_eps = {}
        for out_ep in out_eps:
            span = all_spans[true_assignments[out_ep][in_span.GetId()]]
            outgoing_spans.append([span.start_mus, span.duration_mus, span.GetParentProcess(all_processes, all_spans), span.GetChildProcess(all_processes, all_spans)])
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
                        if G2.has_edge(x[3] + "-end", y[3] + "-start"):
                            G2.remove_edge(x[3] + "-end", y[3] + "-start")
                        if x[0] > y[0]:
                            if G2.has_edge(x[3] + "-start", y[3] + "-start"):
                                G2.remove_edge(x[3] + "-start", y[3] + "-start")
                    if x[0] + x[1] > y[0] + y[1]:
                        if G2.has_edge(x[3] + "-end", y[3] + "-end"):
                            G2.remove_edge(x[3] + "-end", y[3] + "-end")
                    if y[0] + y[1] > x[0]:
                        if G.has_edge(j, i):
                            G.remove_edge(j, i)
                        if G1.has_edge(y[3], x[3]):
                            G1.remove_edge(y[3], x[3])
                        if G2.has_edge(y[3] + "-end", x[3] + "-start"):
                            G2.remove_edge(y[3] + "-end", x[3] + "-start")
                        if y[0] > x[0]:
                            if G2.has_edge(y[3] + "-start", x[3] + "-start"):
                                G2.remove_edge(y[3] + "-start", x[3] + "-start")
                    if y[0] + y[1] > x[0] + x[1]:
                        if G2.has_edge(y[3] + "-end", x[3] + "-end"):
                            G2.remove_edge(y[3] + "-end", x[3] + "-end")

    sorted_grouped_order = topological_sort_grouped(G)
    service_order = copy.deepcopy(sorted_grouped_order)
    for i in range(len(sorted_grouped_order)):
        for j, service_id in enumerate(sorted_grouped_order[i]):
            service_order[i][j] = outgoing_eps[sorted_grouped_order[i][j]]

    return G1

def CalculateTraceStartTime(files):
    start_times = []
    for i, afile in enumerate(files):
        print("Calculating start time for ", i)
        with open(afile, 'r') as f:
            json_data = json.load(f)
            data = json_data.get("data", [])
            if not data:
                start_times.append(float('inf'))
                continue

            trace = data[0]
            spans = trace.get("spans", [])
            if not spans:
                start_times.append(float('inf'))
                continue

            # Identify the root span (span with no references)
            root_span = next((span for span in spans if len(span.get("references", [])) == 0), None)
            if not root_span:
                start_times.append(float('inf'))
                continue

            start_times.append(float(root_span["startTime"]))

    return start_times

def TimeOrder(files):
    start_times = CalculateTraceStartTime(files)
    sorted_indices = np.argsort(start_times)
    sorted_files = [files[i] for i in sorted_indices]
    return sorted_files

def GetAllTracesInDir(directory):
    sorted_filenames_path = Path(directory) / "time_order_filenames.pickle"
    if CLEAR_CACHE:
        if os.path.exists(sorted_filenames_path):
            os.remove(sorted_filenames_path)

    if os.path.exists(sorted_filenames_path):
        with open(sorted_filenames_path, "rb") as f:
            files = pickle.load(f)
    else:
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        files = [f for f in files if f.endswith("json")]
        full_path = os.path.abspath(directory)
        files = [os.path.join(full_path, f) for f in files]
        files = TimeOrder(files)

        with open(sorted_filenames_path, "wb") as f:
            pickle.dump(files, f)

    return files

# Parse the spans JSON and process self-loops
def ParseSpansJson(spans_json, selfLoopMap, serviceLoopMap, first_span):
    spans = {}
    overall_trace_id = None

    # Step 1: Create Span objects without linking
    for span in spans_json:

        span_kind = None

        for tag in span["tags"]:
            if tag["key"] == "span.kind":
                span_kind = tag["value"]

        process_id = span["processID"]
        trace_id = span["traceID"]
        sid = span["spanID"]
        span_id = (trace_id, sid)
        start_mus = span["startTime"]
        duration_mus = span["duration"]
        if "requestType" in span.keys():
            op_name = span.get("requestType", None)
        else:
            op_name = span.get("operationName", None)

        if overall_trace_id == None:
            overall_trace_id = trace_id
        else:
            if trace_id != overall_trace_id:
                print("Different trace ids for spans in the same trace!")
                assert False

        references = []
        for ref in span["references"]:
            references.append((ref["traceID"], ref["spanID"]))

        if first_span is None:
            if span_kind == "client":
                sid = sid + ".client"
                span_id = (trace_id, sid)

            if span_kind == "server":
                if len(references) == 1:
                    references[0] = (references[0][0], sid + '.client')

        if first_span == None:
            if span["caller"] == span["callee"]:
                sanitized_sid = sid
                if sanitized_sid.endswith('.client'):
                    sanitized_sid = sanitized_sid[:-7]
                original_callee = span["callee"]
                if sanitized_sid not in selfLoopMap:
                    new_callee = misc.GenerateRandomID(suffix="-loop")
                    selfLoopMap[sanitized_sid] = [original_callee, new_callee]
                    serviceLoopMap[new_callee] = original_callee
                span["callee"] = selfLoopMap[sanitized_sid][1]
                if span_kind == "server":
                    process_id = selfLoopMap[sanitized_sid][1]
                    span["processID"] = process_id

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

    # Todo: Fix multiple outgoing to same service
    if first_span == None:

        # Step 2: Create a temporary structure to hold child references
        temp_children = {}

        for span_id, span in spans.items():
            if not span.IsRoot():
                parent_id = span.references[0]
                if parent_id not in temp_children:
                    temp_children[parent_id] = []
                temp_children[parent_id].append(span_id)

        # Step 3: Link spans using the temporary structure
        for parent_id, children in temp_children.items():
            if parent_id in spans:
                for child_id in children:
                    spans[parent_id].AddChild(child_id)

        # Check time order constraint
        def check_time_constraints(span):
            for child_id in span.children_spans:
                child = spans[child_id]
                if not (
                    span.start_mus <= child.start_mus and
                    (span.start_mus + span.duration_mus) >= (child.start_mus + child.duration_mus)
                ):
                    print(f"Time constraint violated between span {span.sid} and its child {child.sid}")
                    return False
                if not check_time_constraints(child):
                    return False
            return True

        root_span = next((span for span in spans.values() if span.IsRoot()), None)
        if root_span and not check_time_constraints(root_span):
            return None, selfLoopMap, serviceLoopMap, spans_json

        # Step 4: Update references for descendants of self-loop spans
        def update_references(span):
            for child_id in span.children_spans:
                child = spans[child_id]
                if child.span_kind == "client":
                    child.process_id = spans[(span.trace_id, span.sid)].process_id
                # Recursively update references for all descendants
                update_references(child)

        def traverse_and_update(span):
            sanitized_sid = span.sid
            if sanitized_sid.endswith('.client'):
                sanitized_sid = sanitized_sid[:-7]
            if sanitized_sid in selfLoopMap:
                update_references(span)
            for child_id in span.children_spans:
                child = spans[child_id]
                traverse_and_update(child)

        if root_span:
            traverse_and_update(root_span)

        for span_id, span in spans.items():
            span.children_spans = []

    return spans, selfLoopMap, serviceLoopMap, spans_json


def ParseProcessesJson(processes_json):
    processes = {}
    for pid in processes_json:
        processes[pid] = processes_json[pid]["serviceName"]
    return processes

def ParseProcessesJson2(spans_json):
    processes = {}
    for span in spans_json:
        processes[span["processID"]] = span["processID"]
    return processes

def FixSpans3(spans, processes):

    for span_id, span in spans.items():
        if span.span_kind == "server":
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

    processes.update(new_processes)
    all_spans.update(spans)

    return spans, processes

def VisualizeTraceFromSpans(spans):
    # Create a directed graph
    G = nx.DiGraph()

    # Dictionary to map process names to simple letters
    process_mapping = {}
    current_label = 'A'

    # Build the graph from spans
    for span_id, span in spans.items():
        span_kind = span.span_kind
        if span_kind == 'client':
            continue  # Ignore client spans

        # Callee process is the process_id of the span itself
        callee_process = span.process_id

        # Caller process is the process_id of the parent span if it exists
        if span.references:
            parent_span_id = span.references[0]
            parent_span = spans[parent_span_id]
            caller_process = parent_span.process_id
        else:
            caller_process = 'client'

        # Map caller process to a simple letter
        if caller_process not in process_mapping:
            process_mapping[caller_process] = current_label
            current_label = chr(ord(current_label) + 1)

        # Map callee process to a simple letter
        if callee_process not in process_mapping:
            process_mapping[callee_process] = current_label
            current_label = chr(ord(current_label) + 1)

        caller_label = process_mapping[caller_process]
        callee_label = process_mapping[callee_process]

        # Add nodes for the caller and callee processes
        G.add_node(caller_label)
        G.add_node(callee_label)

        # Add an edge from caller to callee with spanID as the edge name
        G.add_edge(caller_label, callee_label, spanID=span_id[1])

    # Draw the graph
    plt.figure(figsize=(14, 10))

    # Plot the graph
    pos = nx.spring_layout(G, k=1.5, iterations=200)  # Position nodes using the Fruchterman-Reingold force-directed algorithm
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)

    # Draw edge labels
    edge_labels = {(u, v): d['spanID'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # Print process mapping table
    print("Process Mapping Table:")
    print("-----------------------")
    print("{:<20} {:<20}".format('Real Process Name', 'Dummy Name'))
    for process, label in process_mapping.items():
        print("{:<20} {:<20}".format(process, label))

def VisualizeTrace(trace_data):
    # Create a directed graph
    G = nx.DiGraph()

    # Dictionary to map process names to simple letters
    process_mapping = {}
    current_label = 'A'

    # Parse the spans and build the graph
    for trace in trace_data['data']:
        for span in trace['spans']:
            span_kind = next(tag['value'] for tag in span['tags'] if tag['key'] == 'span.kind')
            if span_kind == 'client':
                continue  # Ignore client spans

            caller_process = span['caller']
            callee_process = span['callee']
            span_id = span['spanID']

            # Map caller process to a simple letter
            if caller_process not in process_mapping:
                process_mapping[caller_process] = current_label
                current_label = chr(ord(current_label) + 1)

            # Map callee process to a simple letter
            if callee_process not in process_mapping:
                process_mapping[callee_process] = current_label
                current_label = chr(ord(current_label) + 1)

            caller_label = process_mapping[caller_process]
            callee_label = process_mapping[callee_process]

            # Add nodes for the caller and callee processes
            G.add_node(caller_label)
            G.add_node(callee_label)

            # Add an edge from caller to callee with spanID as the edge name
            G.add_edge(caller_label, callee_label, spanID=span_id)

    # Draw the graph
    plt.figure(figsize=(14, 10))

    # Plot the graph
    pos = nx.spring_layout(G, k=0.5, iterations=100)  # Position nodes using the Fruchterman-Reingold force-directed algorithm
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)

    # Draw edge labels
    edge_labels = {(u, v): d['spanID'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # Print process mapping table
    print("Process Mapping Table:")
    print("-----------------------")
    print("{:<20} {:<20}".format('Real Process Name', 'Dummy Name'))
    for process, label in process_mapping.items():
        print("{:<20} {:<20}".format(process, label))

def ParseJsonTrace(trace_json, selfLoopMap, serviceLoopMap):
    match FIX:
        case 0: first_span = "init-span"
        case 1: first_span = "ComposeReview"
        case 2: first_span = "HTTP GET /hotels"
        case 3: first_span = "HTTP GET /recommendations"
        case 4: first_span = "[Todo] CompleteTodoCommandHandler"
        case 5: first_span = None
    ret = []
    processes = None

    with open(trace_json, "r") as tfile:
        json_data = json.load(tfile)
        json_data = json_data["data"]
        for d in json_data:
            trace_id = d["traceID"]
            spans, selfLoopMap, serviceLoopMap, d["spans"] = ParseSpansJson(d["spans"], selfLoopMap, serviceLoopMap, first_span)
            if spans == None:
                return None, None, None, selfLoopMap, serviceLoopMap
            if "requestType" in d["spans"][0].keys():
                processes = ParseProcessesJson2(d["spans"])
            else:
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

    return trace_id, spans, processes, selfLoopMap, serviceLoopMap

in_spans_by_process = dict()
out_spans_by_process = dict()

def ProcessTraceData(trace_id, spans, processes):
    match FIX:
        case 0: first_span = "init-span"
        case 1: first_span = "ComposeReview"
        case 2: first_span = "HTTP GET /hotels"
        case 3: first_span = "HTTP GET /recommendations"
        case 4: first_span = "[Todo] CompleteTodoCommandHandler"
        case 5: first_span = None

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
        for child in span.children_spans:
            ExploreSubTree(child, depth + 1)

    # comment out if condition to consider all microservice kinds
    if spans[root_span_id].op_name == first_span or first_span == None:
        ExploreSubTree(root_span_id, 0)
        all_spans.update(spans)
        all_processes[trace_id] = processes
        return 1
    return 0

if args.absolute_path:
    if ABSOLUTE_PATH is not None:
        ABSOLUTE_PATH = ABSOLUTE_PATH.rstrip('\\')
        if COMPRESSED:
            misc.uncompress(ABSOLUTE_PATH + "/", ABSOLUTE_PATH + ".tar.lama")
        traces = GetAllTracesInDir(ABSOLUTE_PATH)
else:
    traces = GetAllTracesInDir(RELATIVE_PATH)

# traces.sort()
cnt = 0
selfLoopMap = {}
serviceLoopMap = {}
for i, trace in enumerate(traces):
    if i % 1 == 0:
        print("Loading traces:", i)
    if config.VERBOSE:
        print("\n\n\n")
    trace_id, spans, processes, selfLoopMap, serviceLoopMap = ParseJsonTrace(trace, selfLoopMap, serviceLoopMap)
    if trace_id == None:
        continue
    cnt += ProcessTraceData(trace_id, spans, processes)
    if cnt > 1000:  # 10000:
        break

if COMPRESSED:
    shutil.rmtree(ABSOLUTE_PATH + "/")

if config.VERBOSE:
    print("Incoming spans")
    for p, s in in_spans_by_process.items():
        print("  %s: %s" % (p, s))
    print("Outgoing spans")
    for p, s in out_spans_by_process.items():
        print("  %s: %s" % (p, s))
    print("\n\n\n")

predictors = [
    ("MaxScoreBatch", Timing2(all_spans, all_processes)),
    ("MaxScoreBatchParallel", Timing2(all_spans, all_processes)),
    ("MaxScore", Timing(all_spans, all_processes)),
    ("WAP5", WAP5(all_spans, all_processes)),
    ("FCFS", FCFS(all_spans, all_processes)),
    ("ArrivalOrder", ArrivalOrder(all_spans, all_processes)),
    ("vPathOld", vPathOld(all_spans, all_processes)),
    ("vPath", vPath(all_spans, all_processes)),
    ("MaxScoreBatchParallelWithoutIterations", Timing3(all_spans, all_processes)),
    ("MaxScoreBatchParallel", Timing3(all_spans, all_processes)),
    ("MaxScoreBatchSubsetWithSkips", Timing3(all_spans, all_processes)),
]

predictors = [predictors[i] for i in PREDICTOR_INDICES if i < len(predictors)]

accuracy_per_process = {}
accuracy_overall = {}
topk_accuracy_overall = {}
accuracy_percentile_bins = {}
traces_overall = {}
cache_updated = False
rps_rates = {}

with open(os.path.join(PROJECT_ROOT, "data/misc/service_to_replica_new.pickle"), "rb") as input_file:
    service_to_replica = cPickle.load(input_file)

def process_single_process(method, predictor, process_id, process, in_spans_by_process, out_spans_by_process, all_processes, all_spans, service_to_replica, serviceLoopMap, COMPRESS_FACTOR, REPEAT_FACTOR, CACHE_RATE, INSTRUMENTED, CalculateRate, FindOrder, transforms, utils):
    in_spans = copy.deepcopy(in_spans_by_process[process])
    out_spans = copy.deepcopy(out_spans_by_process[process])

    if len(out_spans) == 0:
        return None, None, None, None, None, None, None, None, None

    if COMPRESS_FACTOR > 0:
        if process in service_to_replica:
            load_factor = max(1, math.ceil(COMPRESS_FACTOR / len(service_to_replica[process])))
        elif process[-5:] == "-loop":
            replicas = len(service_to_replica[serviceLoopMap[process]])
            load_factor = max(1, math.ceil(COMPRESS_FACTOR / replicas))
        else:
            assert False

    def PartitionSpansByEndPoint(spans, endpoint_lambda):
        partitions = {}
        for span in spans:
            ep = endpoint_lambda(span)
            if ep not in partitions:
                partitions[ep] = []
            partitions[ep].append(span)
        for ep, part in partitions.items():
            part.sort(key=lambda x: (x.start_mus, x.start_mus + x.duration_mus))
        return partitions

    in_span_partitions = PartitionSpansByEndPoint(
        in_spans, lambda x: x.GetParentProcess(all_processes, all_spans)
    )
    out_span_partitions = PartitionSpansByEndPoint(
        out_spans, lambda x: x.GetChildProcess(all_processes, all_spans)
    )

    if len(in_span_partitions.keys()) > 1:
        return None, None, None, None, None, None, None, None, None

    PARALLEL = (method == "MaxScoreBatchParallel") or (method == "MaxScoreBatchParallelWithoutIterations")

    instrumented_hops = []

    true_assignments = utils.GetGroundTruth(in_span_partitions, out_span_partitions)
    invocation_graph = FindOrder(in_span_partitions, out_span_partitions, true_assignments)

    if COMPRESS_FACTOR > 0:
        in_span_partitions, out_span_partitions = transforms.repeat_change_spans(in_span_partitions, out_span_partitions, REPEAT_FACTOR, load_factor)
        true_assignments = utils.GetGroundTruth(in_span_partitions, out_span_partitions)

    if process == "frontend" and method not in ["MaxScoreBatch", "MaxScoreBatchParallel", "FCFS", "ArrivalOrder"]:
        true_assignments = transforms.create_cache_hits(true_assignments, in_span_partitions, out_span_partitions, cache_rate=CACHE_RATE)

    start_time = time.time()
    pred_topk_assignments = None
    not_best_count = None
    num_spans = None
    per_span_candidates = None

    if method in ["MaxScoreBatch", "MaxScoreBatchParallel", "MaxScoreBatchSubset", "MaxScoreBatchParallelWithoutIterations"]:
        pred_assignments, not_best_count, num_spans, per_span_candidates = predictor.FindAssignments(
            method, process, in_span_partitions, out_span_partitions, PARALLEL, instrumented_hops, true_assignments, invocation_graph
        )
    elif method == "MaxScoreBatchSubsetWithSkips":
        pred_assignments, pred_topk_assignments, not_best_count, num_spans, per_span_candidates, unassigned = predictor.FindAssignments(
            method, process, in_span_partitions, out_span_partitions, PARALLEL, instrumented_hops, true_assignments, invocation_graph
        )
    elif method == "MaxScoreBatchSubsetWithTrueSkips":
        pred_assignments, pred_topk_assignments, not_best_count, num_spans, per_span_candidates, unassigned = predictor.FindAssignments(
            method, process, in_span_partitions, out_span_partitions, PARALLEL, instrumented_hops, true_assignments, invocation_graph, True, False
        )
    elif method == "MaxScoreBatchSubsetWithTrueDist":
        pred_assignments, pred_topk_assignments, not_best_count, num_spans, per_span_candidates, unassigned = predictor.FindAssignments(
            method, process, in_span_partitions, out_span_partitions, PARALLEL, instrumented_hops, true_assignments, invocation_graph, False, True
        )
    else:
        pred_assignments = predictor.FindAssignments(
            method, process, in_span_partitions, out_span_partitions, PARALLEL, instrumented_hops, true_assignments
        )

    acc = utils.AccuracyForService(pred_assignments, true_assignments, in_span_partitions)
    if method == "MaxScoreBatchSubsetWithSkips":
        acc2 = utils.TopKAccuracyForService(pred_topk_assignments, true_assignments, in_span_partitions)
    else:
        acc2 = None

    return process, true_assignments, pred_assignments, pred_topk_assignments, acc, acc2, not_best_count, num_spans, per_span_candidates

if EXECUTE_PARALLEL:

    for method, predictor in predictors:
        random.seed(10)

        if method in ["MaxScoreBatch", "MaxScoreBatchSubsetWithSkips"]:
            confidence_scores_by_process = {}
        if method in ["MaxScoreBatchSubset", "MaxScoreBatchParallel", "MaxScoreBatch", "MaxScoreBatchSubsetWithSkips"]:
            candidates_per_process = {}

        true_assignments_by_process = {}
        pred_assignments_by_process = {}
        pred_topk_assignments_by_process = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    process_single_process,
                    method, predictor, process_id, process, in_spans_by_process, out_spans_by_process,
                    all_processes, all_spans, service_to_replica, serviceLoopMap, COMPRESS_FACTOR, REPEAT_FACTOR,
                    CACHE_RATE, INSTRUMENTED, CalculateRate, FindOrder, transforms, utils
                ): process_id for process_id, process in enumerate(out_spans_by_process.keys())
            }

            for future in concurrent.futures.as_completed(futures):
                process, true_assignments, pred_assignments, pred_topk_assignments, acc, acc2, not_best_count, num_spans, per_span_candidates = future.result()

                if process is None:
                    continue

                true_assignments_by_process[process] = true_assignments
                pred_assignments_by_process[process] = pred_assignments
                if method == "MaxScoreBatchSubsetWithSkips":
                    pred_topk_assignments_by_process[process] = pred_topk_assignments

                accuracy_per_process[(method, process)] = acc

                if method in ["MaxScoreBatch", "MaxScoreBatchSubsetWithSkips"]:
                    confidence_scores_by_process[process] = [acc, not_best_count, num_spans]

                if method in ["MaxScoreBatchSubset", "MaxScoreBatchParallel", "MaxScoreBatchSubsetWithSkips"]:
                    candidates_per_process[process] = per_span_candidates

        trace_acc, acc_e2e = utils.AccuracyEndToEnd(
            pred_assignments_by_process, true_assignments_by_process, in_spans_by_process
        )
        if method == "MaxScoreBatchSubsetWithSkips":
            trace_acc_2, acc_e2e_2 = utils.TopKAccuracyEndToEnd(
                pred_topk_assignments_by_process, true_assignments_by_process, in_spans_by_process
            )
        true_traces_e2e, pred_traces_e2e = utils.ConstructEndToEndTraces(
            pred_assignments_by_process, true_assignments_by_process, in_spans_by_process, all_spans
        )
        traces_overall[method] = [true_traces_e2e, pred_traces_e2e]

        print("End-to-end accuracy for method %s: %.3f%%\n\n" % (method, acc_e2e * 100))
        if method == "MaxScoreBatchSubsetWithSkips":
            print("End-to-end top K accuracy for method %s: %.3f%%\n\n" % (method, acc_e2e_2 * 100))
        accuracy_overall[method] = acc_e2e * 100
        if method == "MaxScoreBatchSubsetWithSkips":
            accuracy_overall[method + "TopK"] = acc_e2e_2 * 100
        accuracy_percentile_bins[method] = utils.BinAccuracyByResponseTimes(trace_acc, all_spans)
        if method == "MaxScoreBatchSubsetWithSkips":
            accuracy_percentile_bins[method + "TopK"] = utils.BinAccuracyByResponseTimes(trace_acc_2, all_spans)

else:

    for method, predictor in predictors:

        random.seed(10)

        if method == "MaxScoreBatch" or method == "MaxScoreBatchSubsetWithSkips":
            confidence_scores_by_process = {}
        if method == "MaxScoreBatchSubset" or method == "MaxScoreBatchParallel" or method == "MaxScoreBatch" or method == "MaxScoreBatchSubsetWithSkips":
            candidates_per_process = {}

        true_assignments_by_process = {}
        pred_assignments_by_process = {}
        pred_topk_assignments_by_process = {}
        for process_id, process in enumerate(out_spans_by_process.keys()):

            in_spans = copy.deepcopy(in_spans_by_process[process])
            out_spans = copy.deepcopy(out_spans_by_process[process])

            if len(out_spans) == 0:
                continue

            if COMPRESS_FACTOR > 1:
                if process in service_to_replica:
                    print("Number of replicas: ", len(service_to_replica[process]))
                    load_factor = max(1, math.ceil(COMPRESS_FACTOR / len(service_to_replica[process])))
                    print("Dynamic load factor: ", load_factor)
                elif process[-5:] == "-loop":
                    replicas = len(service_to_replica[serviceLoopMap[process]])
                    print("Number of replicas: ", replicas)
                    load_factor = max(1, math.ceil(COMPRESS_FACTOR / replicas))
                    print("Dynamic load factor: ", load_factor)
                else:
                    print("Not found")
                    print(process)
                    assert False

            # partition spans by the other endpoint
            def PartitionSpansByEndPoint(spans, endpoint_lambda):
                partitions = {}
                for span in spans:
                    ep = endpoint_lambda(span)
                    if ep not in partitions:
                        partitions[ep] = []
                    partitions[ep].append(span)
                for ep, part in partitions.items():
                    part.sort(key=lambda x: (x.start_mus, x.start_mus + x.duration_mus))
                return partitions

            print("Process: ", process)
            # partition spans by subservice at the other end
            in_span_partitions = PartitionSpansByEndPoint(
                in_spans, lambda x: x.GetParentProcess(all_processes, all_spans)
            )
            print("Incoming span partitions: ", process, in_span_partitions.keys())
            out_span_partitions = PartitionSpansByEndPoint(
                out_spans, lambda x: x.GetChildProcess(all_processes, all_spans)
            )
            print("Outgoing span partitions: ", process, out_span_partitions.keys())

            if len(in_span_partitions.keys()) > 1:
                print("SKIPPING THIS PROCESS:", process)
                continue

            if method == "MaxScoreBatchParallel":
                PARALLEL = True
            else:
                PARALLEL = False

            instrumented_hops = []

            true_assignments = utils.GetGroundTruth(in_span_partitions, out_span_partitions)

            copy_x = copy.deepcopy(in_span_partitions)
            copy_y = copy.deepcopy(out_span_partitions)
            copy_z = copy.deepcopy(true_assignments)

            invocation_graph = FindOrder(in_span_partitions, out_span_partitions, true_assignments)

            ep_in = list(in_span_partitions.keys())[0]
            if COMPRESS_FACTOR > 1:
                in_span_partitions, out_span_partitions = transforms.repeat_change_spans(in_span_partitions, out_span_partitions, REPEAT_FACTOR, load_factor)
                true_assignments = utils.GetGroundTruth(in_span_partitions, out_span_partitions)

            if process == "frontend" and (method != "MaxScoreBatch" or method != "MaxScoreBatchParallel" or method !=  "FCFS" or method !=  "ArrivalOrder"):
                print("cache %: ", float(CACHE_RATE) * 100)
                true_assignments = transforms.create_cache_hits(true_assignments, in_span_partitions, out_span_partitions, cache_rate=CACHE_RATE)

            start_time = time.time()

            if method == "MaxScoreBatch":
                pred_assignments, not_best_count, num_spans, per_span_candidates = predictor.FindAssignments(
                    method, process, in_span_partitions, out_span_partitions, PARALLEL, instrumented_hops, true_assignments
                )
            elif method == "MaxScoreBatchParallel" or method == "MaxScoreBatchParallelWithoutPerfectCuts" or method == "MaxScoreBatchParallelWithoutIterations":
                pred_assignments, pred_topk_assignments, not_best_count, num_spans, per_span_candidates, unassigned = predictor.FindAssignments(
                    method, process, in_span_partitions, out_span_partitions, True, instrumented_hops, true_assignments, invocation_graph
                )
            elif method == "MaxScoreBatchSubset":
                pred_assignments, not_best_count, num_spans, per_span_candidates = predictor.FindAssignments(
                    method, process, in_span_partitions, out_span_partitions, PARALLEL, instrumented_hops, true_assignments
                )
            elif method == "MaxScore" or method == "MaxScoreBatchParallel2":
                pred_assignments = predictor.FindAssignments(
                    method, process, in_span_partitions, out_span_partitions, True, instrumented_hops, true_assignments
                )
            elif method == "MaxScoreBatchSubsetWithSkips":
                pred_assignments, pred_topk_assignments, not_best_count, num_spans, per_span_candidates, unassigned = predictor.FindAssignments(
                    method, process, in_span_partitions, out_span_partitions, PARALLEL, instrumented_hops, true_assignments, invocation_graph
                )
            elif method == "MaxScoreBatchSubsetWithTrueSkips":
                pred_assignments, pred_topk_assignments, not_best_count, num_spans, per_span_candidates, unassigned = predictor.FindAssignments(
                    method, process, in_span_partitions, out_span_partitions, PARALLEL, instrumented_hops, true_assignments, invocation_graph, True, False
                )
            elif method == "MaxScoreBatchSubsetWithTrueDist":
                pred_assignments, pred_topk_assignments, not_best_count, num_spans, per_span_candidate, unassigned = predictor.FindAssignments(
                    method, process, in_span_partitions, out_span_partitions, PARALLEL, instrumented_hops, true_assignments, invocation_graph, False, True
                )
            else:
                pred_assignments = predictor.FindAssignments(
                    method, process, in_span_partitions, out_span_partitions, PARALLEL, instrumented_hops, true_assignments
                )

            print("--- %s seconds ---" % (time.time() - start_time))

            acc = utils.AccuracyForService(pred_assignments, true_assignments, in_span_partitions)
            print("Accuracy for service %s: %.3f%%\n" % (process, acc * 100))
            if method == "MaxScoreBatchSubsetWithSkips":
                acc2 = utils.TopKAccuracyForService(pred_topk_assignments, true_assignments, in_span_partitions)
                print("Top K accuracy for service %s: %.3f%%\n" % (process, acc2 * 100))
            true_assignments_by_process[process] = true_assignments
            pred_assignments_by_process[process] = pred_assignments
            if method == "MaxScoreBatchSubsetWithSkips":
                pred_topk_assignments_by_process[process] = pred_topk_assignments

            accuracy_per_process[(method, process_id)] = acc

            if method == "MaxScoreBatch" or method == "MaxScoreBatchSubsetWithSkips":
                print(not_best_count, num_spans)
                confidence_scores_by_process[process] = [acc, not_best_count, num_spans]

            if method == "MaxScoreBatchSubset" or method == "MaxScoreBatchParallel" or method == "MaxScoreBatchSubsetWithSkips":
                candidates_per_process[process] = per_span_candidates

        trace_acc, acc_e2e = utils.AccuracyEndToEnd(
            pred_assignments_by_process, true_assignments_by_process, in_spans_by_process
        )
        if method == "MaxScoreBatchSubsetWithSkips":
            trace_acc_2, acc_e2e_2 = utils.TopKAccuracyEndToEnd(
                pred_topk_assignments_by_process, true_assignments_by_process, in_spans_by_process
            )
        true_traces_e2e, pred_traces_e2e = utils.ConstructEndToEndTraces(
            pred_assignments_by_process, true_assignments_by_process, in_spans_by_process, all_spans
        )
        traces_overall[method] = [true_traces_e2e, pred_traces_e2e]

        print("End-to-end accuracy for method %s: %.3f%%\n\n" % (method, acc_e2e * 100))
        if method == "MaxScoreBatchSubsetWithSkips":
            print("End-to-end top K accuracy for method %s: %.3f%%\n\n" % (method, acc_e2e_2 * 100))
        accuracy_overall[method] = acc_e2e * 100
        if method == "MaxScoreBatchSubsetWithSkips":
            accuracy_overall[method + "TopK"] = acc_e2e_2 * 100
        accuracy_percentile_bins[method] = utils.BinAccuracyByResponseTimes(trace_acc, all_spans)
        if method == "MaxScoreBatchSubsetWithSkips":
            accuracy_percentile_bins[method + "TopK"] = utils.BinAccuracyByResponseTimes(trace_acc_2, all_spans)

for key in accuracy_overall.keys():
    print("End-to-end accuracy for method %s: %.3f%%" % (key, accuracy_overall[key]))

with open(RESULTS_DIR + 'bin_acc' + "_" + TEST_NAME + "_" + str(LOAD_LEVEL) + "_" + str(int(COMPRESS_FACTOR)) + "_" + str(int(REPEAT_FACTOR)) + "_" + str(CACHE_RATE) + '.pickle', 'wb') as handle:
    pickle.dump(accuracy_percentile_bins, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open(RESULTS_DIR + 'accuracy' + "_" + TEST_NAME + "_" + str(LOAD_LEVEL) + "_" + str(int(COMPRESS_FACTOR)) + "_" + str(int(REPEAT_FACTOR)) + "_" + str(CACHE_RATE) + '.pickle', 'wb') as handle:
    pickle.dump(accuracy_overall, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open(RESULTS_DIR + 'e2e' + "_" + TEST_NAME + "_" + str(LOAD_LEVEL) + "_" + str(int(COMPRESS_FACTOR)) + "_" + str(int(REPEAT_FACTOR)) + "_" + str(CACHE_RATE) + '.pickle', 'wb') as handle:
    pickle.dump(traces_overall, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open(RESULTS_DIR + 'confidence_scores' + "_" + TEST_NAME + "_" + str(LOAD_LEVEL) + "_" + str(int(COMPRESS_FACTOR)) + "_" + str(int(REPEAT_FACTOR)) + "_" + str(CACHE_RATE) + '.pickle', 'wb') as handle:
    pickle.dump(confidence_scores_by_process, handle, protocol = pickle.HIGHEST_PROTOCOL)
with open(RESULTS_DIR + 'process_acc' + "_" + TEST_NAME + "_" + str(LOAD_LEVEL) + "_" + str(int(COMPRESS_FACTOR)) + "_" + str(int(REPEAT_FACTOR)) + "_" + str(CACHE_RATE) + '.pickle', 'wb') as handle:
    pickle.dump(accuracy_per_process, handle, protocol = pickle.HIGHEST_PROTOCOL)
