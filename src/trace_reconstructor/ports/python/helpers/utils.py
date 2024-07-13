import copy
import config
import pickle

def SortPartitionsByTraceId(span_partitions):
    for ep, part in span_partitions.items():
        part.sort(key=lambda x: x.trace_id)

def SortPartitionsByTime(span_partitions):
    for ep, part in span_partitions.items():
        part.sort(key=lambda x: (x.start_mus, x.start_mus + x.duration_mus))

def GetOutEpsInOrder(out_span_partitions):
    eps = []
    for ep, spans in out_span_partitions.items():
        assert len(spans) > 0
        eps.append((ep, spans[0].start_mus))
    eps.sort(key=lambda x: x[1])
    return [x[0] for x in eps]

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

def BinAccuracyByServiceTimes(method, PLOTS_DIR):

    for j in range(4, 5):

        query_latency = {}

        with open(PLOTS_DIR + "e2e_" + str((j + 1) * 25) + ".pickle", 'rb') as afile:
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

def BinAccuracyByResponseTimes(trace_acc, all_spans):
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
        if config.VERBOSE:
            print(
                "Accuracy of %d-percentile bin: %.3f, response_time (ms): %.1f"
                % (percentile, acc, d / 1000.0)
            )
        accuracy.append((percentile, acc, d / 1000.0))
    return accuracy

def ConstructEndToEndTraces(
    pred_assignments_by_process, true_assignments_by_process, in_spans_by_process, all_spans
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

def PrintLatency12(trace_acc, all_spans):
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
        # print(x)
