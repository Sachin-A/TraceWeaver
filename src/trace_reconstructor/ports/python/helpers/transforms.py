import random
import numpy as np
from spans import Span
from helpers.utils import GetOutEpsInOrder

def repeat_change_spans(in_span_partitions, out_span_partitions, repeat_factor, compress_factor):

    if repeat_factor == 1 and compress_factor == 1:
        return in_span_partitions, out_span_partitions

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

    span_inds = span_inds * repeat_factor
    random.shuffle(span_inds)
    min_start_t = min(float(in_span.start_mus) for in_span in in_spans) / compress_factor
    max_start_t = max(float(in_span.start_mus) for in_span in in_spans) / compress_factor
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

def create_cache_hits(true_assignments, in_span_partitions, out_span_partitions, cache_rate, exponential = False):

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

    ep_in, in_spans = list(in_span_partitions.items())[0]

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
