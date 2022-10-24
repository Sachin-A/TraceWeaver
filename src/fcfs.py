class FCFS(object):
    def __init__(self, all_spans, all_processes):
        self.all_spans = all_spans
        self.all_processes = all_processes

    def FindAssignments(
        self, process, in_span_partitions, out_span_partitions
    ):
        assert len(in_span_partitions) == 1
        _, in_spans = list(in_span_partitions.items())[0]
        all_assignments = { ep: {} for ep in out_span_partitions.keys() }
        for ind in range(len(in_spans)):
            for ep, out_spans in out_span_partitions.items():
                all_assignments[ep][in_spans[ind].GetId()] = out_spans[ind].GetId()
        return all_assignments
