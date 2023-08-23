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
