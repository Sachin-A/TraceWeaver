# class FCFS(object):
#     def __init__(self):
#         pass

#     def FindAssignments(self, process, in_spans, out_spans):
#         all_assignments = {}
#         in_spans.sort(key = lambda x: int(x[2]))
#         out_spans.sort(key = lambda x: int(x[2]))
#         for i in range(len(in_spans)):
#             all_assignments[(in_spans[i][1],in_spans[i][3])] = [out_spans[i][1],out_spans[i][3]]
#         return all_assignments

class FCFS(object):
    def __init__(self):
        pass

    def FindAssignments(
        self, process, in_span_partitions, out_span_partitions
    ):
        assert len(in_span_partitions) == 1
        for ep in in_span_partitions.keys():
            in_span_partitions[ep].sort(key = lambda x: float(x[2]))
        for ep in out_span_partitions.keys():
            out_span_partitions[ep].sort(key = lambda x: float(x[2]))

        _, in_spans = list(in_span_partitions.items())[0]
        all_assignments = { ep: {} for ep in out_span_partitions.keys() }
        for ind in range(len(in_spans)):
            for ep, out_spans in out_span_partitions.items():
                all_assignments[ep][(in_spans[ind][1],in_spans[ind][3])] = [out_spans[ind][1], out_spans[ind][3]]
        return all_assignments
