class Event(object):
    def __init__(
        self,
        trace_id,
        sid,
        event_time_mus,
        span_kind,
        event_kind,
        ep,
        sort_key,
    ):
        self.trace_id = trace_id
        self.sid = sid
        self.event_time_mus = event_time_mus
        self.span_kind = span_kind
        self.event_kind = event_kind
        self.ep = ep
        self.sort_key = sort_key

    def GetId(self):
        return (self.trace_id, self.sid)

    def __repr__(self):
        return "Event:(%s, %s, %d, %s, %s, %s)" % (
            self.trace_id,
            self.sid,
            self.event_time_mus,
            self.span_kind,
            self.event_kind,
            self.ep,
        )

    def __str__(self):
        return self.__repr__()

class DeepFlow(object):
    def __init__(self, all_spans, all_processes):
        self.all_spans = all_spans
        self.all_processes = all_processes

    def ReturnParent(self, trace_id, in_span_partitions):
        for ep in in_span_partitions.keys():
            for span in in_span_partitions[ep]:
                if span.trace_id == trace_id:
                    return (span.trace_id, span.sid)
        return None

    def FindAssignments(
        self, method, process, in_span_partitions, out_span_partitions, parallel, instrumented_hops, true_assignments
    ):
        assert len(in_span_partitions) == 1
        all_events = []
        for ep in in_span_partitions.keys():
            for span in in_span_partitions[ep]:
                start_event = Event(span.trace_id, span.sid, span.start_mus, span.span_kind, "request", ep, 1)
                end_event = Event(span.trace_id, span.sid, span.start_mus + span.duration_mus, span.span_kind, "response", ep, 4)
                all_events.extend([start_event, end_event])
        for ep in out_span_partitions.keys():
            for span in out_span_partitions[ep]:
                start_event = Event(span.trace_id, span.sid, span.start_mus, span.span_kind, "request", ep, 2)
                end_event = Event(span.trace_id, span.sid, span.start_mus + span.duration_mus, span.span_kind, "response", ep, 3)
                all_events.extend([start_event, end_event])

        all_events.sort(key = lambda x: (float(x.event_time_mus), x.sort_key))
        all_assignments = { ep: {} for ep in out_span_partitions.keys() }
        _, in_spans = list(in_span_partitions.items())[0]
        for ind in range(len(in_spans)):
            for ep, out_spans in out_span_partitions.items():
                all_assignments[ep][(in_spans[ind].trace_id, in_spans[ind].sid)] = ('NA', 'NA')

        latest_incoming = None
        for event in all_events:

            if event.span_kind == "server":
                if event.event_kind == "request":
                    latest_incoming = (event.trace_id, event.sid)
                elif event.event_kind == "response":
                    latest_incoming = None

            if event.span_kind == "client":
                if event.event_kind == "request":
                    if latest_incoming != None:
                        all_assignments[event.ep][latest_incoming] = (event.trace_id, event.sid)
                if event.event_kind == "response":
                    parent_id = self.ReturnParent(event.trace_id, in_span_partitions)
                    if parent_id != None:
                        latest_incoming = parent_id

        return all_assignments
