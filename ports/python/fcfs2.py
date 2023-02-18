import numpy as np

class FCFS2(object):

    def __init__(self, all_spans, all_processes):
        self.all_spans = all_spans
        self.all_processes = all_processes

    def GetOutEpsInOrder(self, out_span_partitions):
        eps = []
        for ep, spans in out_span_partitions.items():
            assert len(spans) > 0
            eps.append((ep, spans[0].start_mus))
        eps.sort(key=lambda x: x[1])
        return [x[0] for x in eps]

    def FindAssignments(
        self, process, in_span_partitions, out_span_partitions
    ):

        assert len(in_span_partitions) == 1

        all_assignments = { ep: {} for ep in out_span_partitions.keys() }

        # Service names for in and out spans
        i_eps = list(in_span_partitions.keys())
        o_eps = self.GetOutEpsInOrder(out_span_partitions)

        for i in range(1, len(out_span_partitions) + 1):

            # picking client with first service
            if i == 1:
                in_spans = in_span_partitions[i_eps[0]]
                out_spans = out_span_partitions[o_eps[0]]
                ep_key = o_eps[0]

            # picking intermediate services
            else:
                in_spans = in_span_partitions[i_eps[0]]
                # out_spans_0 = out_span_partitions[o_eps[i - 2]]
                out_spans_0 = out_spans

                sort_order = np.argsort([x.start_mus + x.duration_mus for x in out_spans_0])
                out_spans = list(np.array(out_span_partitions[o_eps[i - 1]])[sort_order])

                ep_key = o_eps[i - 1]

            for j in range(len(in_spans)):
                all_assignments[ep_key][in_spans[j].GetId()] = out_spans[j].GetId()

        return all_assignments
