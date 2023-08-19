class Span(object):
    def __init__(
        self,
        trace_id,
        sid,
        start_mus,
        duration_mus,
        op_name,
        references,
        process_id,
        span_kind,
        span_tags
    ):
        self.sid = sid
        self.trace_id = trace_id
        self.start_mus = start_mus
        self.duration_mus = duration_mus
        self.op_name = op_name
        self.references = references
        self.process_id = process_id
        self.span_kind = span_kind
        self.tags = span_tags
        self.children_spans = []
        self.taken = False
        self.ep = None

    def AddChild(self, child_span_id):
        self.children_spans.append(child_span_id)

    def GetChildProcess(self, all_processes, all_spans):
        assert self.span_kind == "client"
        assert len(self.children_spans) == 1
        return all_processes[self.trace_id][
            all_spans[self.children_spans[0]].process_id
        ]

    def GetParentProcess(self, all_processes, all_spans):
        if self.IsRoot():
            return "client_" + self.op_name
        assert len(self.references) == 1
        parent_span_id = self.references[0]
        return all_processes[self.trace_id][all_spans[parent_span_id].process_id]

    def GetId(self):
        return (self.trace_id, self.sid)

    def IsRoot(self):
        return len(self.references) == 0

    def __lt__(self, other):
        return self.start_mus < other.start_mus

    def __repr__(self):
        if self.start_mus == "None":
            return "Span:(%s, %s, %s, %s, %s, %s)" % (
                self.trace_id,
                self.sid,
                self.op_name,
                self.start_mus,
                self.duration_mus,
                self.span_kind,
            )
        else:
            return "Span:(%s, %s, %s, %d, %d, %s)" % (
                self.trace_id,
                self.sid,
                self.op_name,
                self.start_mus,
                self.duration_mus,
                self.span_kind,
            )

    def __str__(self):
        return self.__repr__()
