import csv
import sys
import copy
import json
import string
import random
import numpy as np
from fcfs import FCFS
from timing import Timing
from timing2 import Timing2
import _pickle as cPickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from distutils.version import LooseVersion

search_strings = ["(?)", "NAN", "nan", ""]

def noSpanWith(trace, rpc_id):
    for span in trace:
        if span[3] == rpc_id:
            return False
    return True

def isRoot(span, trace):
    if ((span[3] == "0.1" and noSpanWith(trace, "0")) or
        (span[3] == "0.1.1" and noSpanWith(trace, "0.1")) or
        (span[3] == "0")):
        return True
    return False

def isLeaf(span, trace):
    for span0 in trace:
        parent_rpc_id = ".".join(span0[3].split(".")[:-1])
        if span[3] == parent_rpc_id:
            return False
    return True

def missingCol(span_entry):
    for search_string in search_strings:
        if search_string == span_entry:
            return True
    return False

def removeDuplicates(span_partitions):

    for ep in list(span_partitions.keys()):
        duplicates = set()
        for i in range(len(span_partitions[ep])):
            if i == len(span_partitions[ep]) - 1:
                break
            pos = i + 1
            for j in range(pos, len(span_partitions[ep])):
                if span_partitions[ep][j][1] == span_partitions[ep][i][1] and span_partitions[ep][j][3] == span_partitions[ep][i][3]:
                    duplicates.add(j)
        for k in sorted(duplicates, reverse=True):
            del span_partitions[ep][k]

def GetGroundTruth2(in_span_partitions, out_span_partitions):
    assert len(in_span_partitions) == 1
    _, in_spans = list(in_span_partitions.items())[0]
    true_assignments = {ep: {} for ep in out_span_partitions.keys()}
    in_spans_new = []
    out_spans_partitions_new = {ep: [] for ep in out_span_partitions.keys()}
    no_answer_count = 0
    for in_span in in_spans:
        no_answer = True
        answer_ep = None
        answer_index = None
        for ep in out_span_partitions.keys():
            for k, out_span in enumerate(out_span_partitions[ep]):
                parent_rpc_id = ".".join(out_span[3].split(".")[:-1])
                if (in_span[1] == out_span[1]) and (in_span[6] == out_span[4]) and (in_span[3] == parent_rpc_id):
                    no_answer = False
                    true_assignments[ep][(in_span[1], in_span[3])] = [out_span[1], out_span[3]]
                    answer_ep = ep
                    answer_index = k
                    in_spans_new.append(list(in_span))
                    out_spans_partitions_new[ep].append(list(out_span))
                    break
            if not no_answer:
                break
            else:
                no_answer_count += 1

        if answer_ep is not None and answer_index is not None:
            del out_span_partitions[answer_ep][answer_index]

    print("No right answer: ", no_answer_count)

    ep = list(in_span_partitions.keys())[0]
    in_span_partitions[ep] = list(in_spans_new)
    return true_assignments, in_span_partitions, out_spans_partitions_new

def GetGroundTruth(in_spans, out_spans):
    true_assignments = {}
    in_spans_new = []
    out_spans_new = []
    no_answer_count = 0
    for in_span in in_spans:
        no_answer = True
        answer_index = None
        for j, out_span in enumerate(out_spans):
            parent_rpc_id = ".".join(out_span[3].split(".")[:-1])
            if (in_span[1] == out_span[1]) and (in_span[6] == out_span[4]) and (in_span[3] == parent_rpc_id):
                true_assignments[(in_span[1], in_span[3])] = [out_span[1], out_span[3]]
                no_answer = False
                in_spans_new.append(list(in_span))
                out_spans_new.append(list(out_span))
                answer_index = j
                break
        if no_answer:
            no_answer_count += 1
        elif answer_index is not None:
            del out_spans[answer_index]

    print("No right answer: ", no_answer_count)

    return true_assignments, in_spans_new, out_spans_new

def AccuracyForService2(pred_assignments, true_assignments, in_span_partitions):
    assert len(in_span_partitions) == 1
    _, in_spans = list(in_span_partitions.items())[0]
    cnt = 0
    for i, in_span in enumerate(in_spans):
        correct = True
        for ep in true_assignments.keys():
            correct = correct and (
                pred_assignments[ep][(in_span[1], in_span[3])]
                == true_assignments[ep][(in_span[1], in_span[3])]
            )
        cnt += int(correct)
    return float(cnt) / len(in_spans)

def AccuracyForService(pred_assignments, true_assignments, in_spans):
    cnt = 0
    for in_span in in_spans:
        print("cnt: ", cnt)
        if pred_assignments[(in_span[1], in_span[3])] == true_assignments[(in_span[1], in_span[3])]:
            cnt += 1
    return cnt / len(in_spans)

def shiftSpans(chosen_spans, in_spans):
    deltas = []
    for i in range(len(in_spans) - 1):
        deltas.append(float(in_spans[i + 1][2]) - float(in_spans[i][2]))

    last = float(in_spans[-1][2])
    for i in range(len(chosen_spans)):
        new_delta = float(random.choices(deltas)[0])
        last = last + new_delta
        chosen_spans[i][2] = last

    return chosen_spans

def repeatSpans(in_span_partitions, out_span_partitions, repeats):
    assert len(in_span_partitions) == 1

    in_span_partitions_copy = copy.deepcopy(in_span_partitions)

    for i in range(repeats):
        if i % 10 == 0:
            print(i)

        ep_in, in_spans = list(in_span_partitions_copy.items())[0]
        repeat_num = max(1, random.randint(len(in_spans)//2, len(in_spans)))
        indices = random.sample(range(len(in_spans)), repeat_num)
        chosen_in_spans = list(np.array(in_spans)[indices])
        chosen_in_spans = shiftSpans(chosen_in_spans, in_spans)

        repeat_ids = []
        for j in range(repeat_num):
            repeat_ids.append(''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(32)))
            chosen_in_spans[j][1] = repeat_ids[j]

        in_span_partitions[ep_in].extend(list(chosen_in_spans))

        for ep_out in list(out_span_partitions.keys()):
            chosen_out_spans = list(np.array(out_span_partitions[ep_out])[indices])

            for ind, out_span in enumerate(chosen_out_spans):
                offset = float(out_span[2]) - float(in_spans[indices[ind]][2])
                chosen_out_spans[ind][2] = float(chosen_in_spans[ind][2]) + offset
                chosen_out_spans[ind][1] = repeat_ids[ind]

            out_span_partitions[ep_out].extend(list(chosen_out_spans))

    return in_span_partitions, out_span_partitions

def changeRateByFactor2(in_span_partitions, out_span_partitions, factor):
    assert len(in_span_partitions) == 1
    ep_in, in_spans = list(in_span_partitions.items())[0]

    for i, in_span in enumerate(in_spans):
        x = int(in_span[2]) / factor
        diff = int(in_span[2]) - x

        for ep_out in out_span_partitions.keys():
            y = int(out_span_partitions[ep_out][i][2]) - diff
            out_span_partitions[ep_out][i][2] = y

        in_span_partitions[ep_in][i][2] = x

def changeRateByFactor(in_spans_new, out_spans_new, factor):
    for i in range(len(in_spans_new)):
        x = int(in_spans_new[i][2]) / factor
        diff = int(in_spans_new[i][2]) - x
        y = int(out_spans_new[i][2]) - diff
        in_spans_new[i][2] = x
        out_spans_new[i][2] = y

def GetEndpoint(span, x):
    return span[x]

def PartitionSpansByEndPoint(spans, x):
    partitions = {}
    for span in spans:
        ep = GetEndpoint(span, x)
        if ep not in partitions:
            partitions[ep] = []
        partitions[ep].append(span)
    for ep, part in partitions.items():
        part.sort(key=lambda x: float((x[2])))
    return partitions

dataset = sys.argv[5]
path = '/scratch/sachina3/projects/clusterdata/cluster-trace-microservices-v2021/data/MSCallGraph/'

if int(sys.argv[1]) == 0:

    traces = {}
    trace_ids = []

    with open(path + 'MSCallGraph_' + dataset + '.csv', mode ='r') as file:
        csv_file = csv.reader(file)
        count = 1
        for row in csv_file:
            if row[1] == 'traceid':
                continue
            if row[1] not in traces:
                trace_ids.append(row[1])
                traces[row[1]] = []
                if count % 10000 == 0:
                    print(count)
                count += 1
            traces[row[1]].append(row)

    with open(r"trace-ids-" + dataset + ".pickle", "wb") as output_file:
        cPickle.dump(trace_ids, output_file)
    with open(r"traces-" + dataset + ".pickle", "wb") as output_file:
        cPickle.dump(traces, output_file)

elif int(sys.argv[1]) == 1:
    print("Loading traces ...")
    with open(r"trace-ids-" + dataset + ".pickle", "rb") as input_file:
        trace_ids = cPickle.load(input_file)
    with open(r"traces-" + dataset + ".pickle", "rb") as input_file:
        traces = cPickle.load(input_file)
    print("Loaded traces")

search_strings = ["(?)", "NAN", "nan", ""]
unique_leaf = {}

if int(sys.argv[2]) == 0:

    process_spans = {}

    for i, x in enumerate(trace_ids):

        if i % 10000 == 0:
            print(i)

        trace = traces[x]
        trace.sort(key = lambda x: LooseVersion(x[3]))

        for j, span in enumerate(trace):

            if not missingCol(span[4]):
                if span[4] not in process_spans:
                    process_spans[span[4]] = [[], []]
                span1 = list(span)
                span1[8] = abs(int(span[8]))
                span1[6] = "leaf"
                process_spans[span[4]][1].append(span1)

            if not missingCol(span[6]):
                if span[6] not in process_spans:
                    process_spans[span[6]] = [[], []]
                span3 = list(span)
                span3[8] = abs(int(span[8]))
                span3[4] = "browser"
                process_spans[span[6]][0].append(span3)

    with open(r"process-spans-" + dataset + ".pickle", "wb") as output_file:
        cPickle.dump(process_spans, output_file)

if int(sys.argv[2]) == 1:
    print("Loading process spans ...")
    with open(r"process-spans-" + dataset + ".pickle", "rb") as input_file:
        process_spans = cPickle.load(input_file)
    print("Loaded process spans")

print("Total processes: ", len(process_spans.keys()))

same_degree_count = 0
for k, v in process_spans.items():
    if len(v[0]) == len(v[1]):
        same_degree_count += 1

print("Total processes satisfying condition: ", same_degree_count)

predictors = [
    ("Greedy++", Timing2()),
    ("Greedy", Timing()),
    ("FCFS", FCFS()),
    # ("FCFS++", FCFS2(all_spans, all_processes)),
]

per_method_accuracy = {}
cgs = {}
for i, process in enumerate(process_spans.keys()):

    # if i != 2539:
    #     continue

    in_spans = process_spans[process][0]
    out_spans = process_spans[process][1]

    if len(in_spans) != len(out_spans) or len(in_spans) < 5:
        continue

    # if len(in_spans) < 5:
    #     continue

    in_span_partitions = PartitionSpansByEndPoint(
        in_spans, 4
    )
    print("Incoming span partitions", process, in_span_partitions.keys())
    out_span_partitions = PartitionSpansByEndPoint(
        out_spans, 6
    )
    print("Outgoing span partitions", process, out_span_partitions.keys())

    ep_in = list(in_span_partitions.keys())[0]
    ep_out = list(out_span_partitions.keys())[0]

    # true_assignments, in_spans, out_spans = GetGroundTruth(in_spans, out_spans)
    true_assignments, in_span_partitions, out_span_partitions = GetGroundTruth2(in_span_partitions, out_span_partitions)

    removeDuplicates(in_span_partitions)
    removeDuplicates(out_span_partitions)

    ep_in = list(in_span_partitions.keys())[0]
    if len(in_span_partitions[ep_in]) <= 1:
        continue
    for ep_out in out_span_partitions.keys():
        if len(in_span_partitions[ep_in]) != len(out_span_partitions[ep_out]):
            continue

    # with open(str(i) + "-raw-data-in-before.csv", 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(in_span_partitions[ep_in])
    # with open(str(i) + "-raw-data-out-before.csv", 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(out_span_partitions[ep_out])

    true_assignments, in_span_partitions, out_span_partitions = GetGroundTruth2(in_span_partitions, out_span_partitions)

    changeRateByFactor2(in_span_partitions, out_span_partitions, int(sys.argv[6]))
    repeatSpans(in_span_partitions, out_span_partitions, int(sys.argv[7]))

    true_assignments, in_span_partitions, out_span_partitions = GetGroundTruth2(in_span_partitions, out_span_partitions)

    # pred_assignments = predictor.FindAssignments(
    #     process, in_spans, out_spans
    # )

    # with open(str(i) + "-raw-data-in.csv", 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(in_span_partitions[ep_in])
    # with open(str(i) + "-raw-data-out.csv", 'w') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerows(out_span_partitions[ep_out])

    for method, predictor in predictors:

        pred_assignments = predictor.FindAssignments(
            process, in_span_partitions, out_span_partitions
        )

        acc = AccuracyForService2(pred_assignments, true_assignments, in_span_partitions)
        if method not in per_method_accuracy:
            per_method_accuracy[method] = []
        per_method_accuracy[method].append(acc)
        if method not in cgs:
            cgs[method] = []
        cgs[method].append([i, acc, len(in_span_partitions[ep_in])])
        print("Accuracy for method %s, service %s: %.3f\n" % (method, i, acc))

for method, _ in predictors:

    print("Accuracy for method: ", method, per_method_accuracy[method])
    print("[index, accuracy, num_spans]: ", cgs[method])

    if len(per_method_accuracy[method]) == 0:
        print("No processes qualify")
        continue

    plt.ylim((0, 1))
    plt.bar([i for i in range(len(per_method_accuracy[method]))], per_method_accuracy[method])
    # plt.xticks([x for x in range(0, len(process_accuracy))])
    plt.savefig("/scratch/sachina3/projects/clusterdata/cluster-trace-microservices-v2021/data/analysis/unique-" + dataset + "-method-" + method + "-factor-" + sys.argv[6] + "-repeat-" + sys.argv[7] + ".svg")
    plt.clf()
    print("Done plotting.")
    print("CG (avg): ", np.mean([i[1] for i in cgs[method]]))
