class FCFS(object):
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
        self.instrumented_hops = instrumented_hops
        self.true_assignments = true_assignments
        _, in_spans = list(in_span_partitions.items())[0]
        all_assignments = { ep: {} for ep in out_span_partitions.keys() }
        for ind in range(len(in_spans)):
            for j, (ep, out_spans) in enumerate(out_span_partitions.items()):
                if ind >= len(out_spans):
                    all_assignments[ep][in_spans[ind].GetId()] = ("NA", "NA")
                    continue
                if (j + 1) in instrumented_hops:
                    all_assignments[ep][in_spans[ind].GetId()] = self.true_assignments[ep][in_spans[ind].GetId()]
                else:
                    all_assignments[ep][in_spans[ind].GetId()] = out_spans[ind].GetId()
        return all_assignments
