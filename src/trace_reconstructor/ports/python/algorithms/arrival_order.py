import numpy as np
from helpers.utils import GetOutEpsInOrder

class ArrivalOrder(object):

    def __init__(self, all_spans, all_processes):
        self.all_spans = all_spans
        self.all_processes = all_processes
        self.parallel = False
        self.instrumented_hops = []
        self.true_assignments = None

    def FindAssignments(
        self, method, process, in_span_partitions, out_span_partitions, parallel, instrumented_hops, true_assignments
    ):

        assert len(in_span_partitions) == 1

        all_assignments = { ep: {} for ep in out_span_partitions.keys() }

        # Endpoint keys for incoming and outgoing spans in sequential order
        i_eps = list(in_span_partitions.keys())
        o_eps = GetOutEpsInOrder(out_span_partitions)

        for i in range(1, len(out_span_partitions) + 1):

            # mapping incoming spans to outgoing spans for 1st endpoint
            # in FCFS order
            if i == 1:
                in_spans = in_span_partitions[i_eps[0]]
                out_spans = out_span_partitions[o_eps[0]]
                ep_key = o_eps[0]

            # mapping incoming spans to outgoing spans in subsequent endpoints
            # in the arrival order of spans from previous outgoing endpoint
            else:
                in_spans = in_span_partitions[i_eps[0]]
                out_spans_prev = out_spans

                sort_order = list(np.argsort([x.start_mus + x.duration_mus for x in out_spans_prev]))
                if len(out_spans_prev) <= len(out_span_partitions[o_eps[i - 1]]):
                    sort_order = sort_order[:len(out_span_partitions[o_eps[i - 1]])]
                    sort_order.extend(
                        [*range(
                            len(out_spans_prev),
                            len(out_span_partitions[o_eps[i - 1]]),
                            1
                        )]
                    )
                else:
                    sort_order = list(filter(lambda x:
                        x < len(out_span_partitions[o_eps[i - 1]]),
                        sort_order
                    ))

                out_spans = list(np.array(out_span_partitions[o_eps[i - 1]])[sort_order])
                ep_key = o_eps[i - 1]

            for ind in range(len(in_spans)):
                if ind >= len(out_spans):
                    all_assignments[ep_key][in_spans[ind].GetId()] = ("NA", "NA")
                else:
                    all_assignments[ep_key][in_spans[ind].GetId()] = out_spans[ind].GetId()

        return all_assignments
