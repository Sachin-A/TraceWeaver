class Timing(object):
    def __init__(self, all_spans, all_processes):
        self.all_spans = all_spans
        self.all_processes = all_processes

    def GetOutgoingSpanOrder(self, outgoing_span_partitions):
        eps = []
        for ep, spans in outgoing_span_partitions.items():
            assert len(spans) > 0
            eps.append((ep, spans[0].start_mus))
        eps.sort(key=lambda x: x[1])
        return [x[0] for x in eps]
        

    def FindMinCostAssignment(self, span, outgoing_eps, outgoing_span_partitions):
        


    def PredictTraceIdSequences(
        self, process, incoming_span_partitions, outgoing_span_partitions
    ):
        assert len(incoming_span_partitions) == 1
        ep, incoming_spans  = list(incoming_span_partitions.items())[0]

        outgoing_eps = self.GetOutgoingSpanOrder(outgoing_span_partitions)
        for span in incoming_spans:
            # find the minimimum cost label assignment for span
            for out
        
