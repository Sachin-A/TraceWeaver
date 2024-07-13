class vPath(object):
    def __init__(self, all_spans, all_processes):
        self.all_spans = all_spans
        self.all_processes = all_processes

    def FindAssignments(
        self, method, process, in_span_partitions, out_span_partitions, parallel, instrumented_hops, true_assignments
    ):
        assert len(in_span_partitions) == 1
        for ep in in_span_partitions.keys():
            in_span_partitions[ep].sort(key = lambda x: float(x.start_mus))
        for ep in out_span_partitions.keys():
            out_span_partitions[ep].sort(key = lambda x: float(x.start_mus))

        _, in_spans = list(in_span_partitions.items())[0]
        all_assignments = { ep: {} for ep in out_span_partitions.keys() }

        for ind in range(len(in_spans)):
            for ep, out_spans in out_span_partitions.items():
                all_assignments[ep][(in_spans[ind].trace_id, in_spans[ind].sid)] = ('NA', 'NA')

        for ep, out_spans in out_span_partitions.items():
            j = 0
            for i in range(len(in_spans)):
                while float(out_spans[j].start_mus) < float(in_spans[i].start_mus):
                    j += 1
                if float(out_spans[j].start_mus) >= float(in_spans[i].start_mus) and ((i == (len(in_spans) - 1)) or (float(out_spans[j].start_mus) < float(in_spans[i + 1].start_mus))):
                    all_assignments[ep][(in_spans[i].trace_id, in_spans[i].sid)] = (out_spans[j].trace_id, out_spans[j].sid)
                    j += 1

        return all_assignments
