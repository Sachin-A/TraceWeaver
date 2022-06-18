class FCFS(object):
    def __init__(self, all_spans, all_processes):
        self.all_spans = all_spans
        self.all_processes = all_processes

    def FCFS_TraceSeq(self, incoming_span_partitions):
        # fcfs doesn't use any info about the subservice
        assert len(incoming_span_partitions) == 1
        ep = list(incoming_span_partitions.keys())[0]
        trace_id_seq = [s.trace_id for s in incoming_span_partitions[ep]]
        return trace_id_seq

    def PredictTraceIdSequences(
        self, process, incoming_span_partitions, outgoing_span_partitions
    ):
        ret = {}
        trace_id_seq_pred = self.FCFS_TraceSeq(incoming_span_partitions)
        # iterate over each subservice
        for ep, part in outgoing_span_partitions.items():
            ret[ep] = trace_id_seq_pred
        return ret
