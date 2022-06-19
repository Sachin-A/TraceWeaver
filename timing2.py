import math
import scipy.stats
import copy
from timing import Timing

VERBOSE = False

class Timing2(Timing):
    def __init__(self, all_spans, all_processes):
        super(Timing2).__init__(all_spans, all_processes)
        self.all_spans = all_spans
        self.all_processes = all_processes
        self.services_times = {}

    def FindTopAssignments(
        self, incoming_span, outgoing_eps, outgoing_span_partitions, K
    ):
        global top_assignments
        top_assignments = []
        def DfsTraverse(stack):
            global top_assignments
            i = len(stack)
            if VERBOSE:
                print("DFSTraverse", i, outgoing_eps, stack)
            last_span = stack[-1]
            if i == len(outgoing_span_partitions) + 1:
                score = self.ScoreAssignment(stack)
                top_assignments.heappush((score, stack))
                if len(top_assignments) > K:
                    top_assignments.heappop()
            else:
                ep = outgoing_eps[i-1]
                for s in outgoing_span_partitions[ep]:
                    if i == 1:
                        # first ep
                        if incoming_span.start_mus < s.start_mus:
                            DfsTraverse(stack + [s])
                    elif i <= len(outgoing_eps):
                        # all other eps 
                        if (
                            last_span.start_mus + last_span.duration_mus < s.start_mus
                            and s.start_mus + s.duration_mus
                            < incoming_span.start_mus + incoming_span.duration_mus
                        ):
                            DfsTraverse(stack + [s])
        DfsTraverse([incoming_span])
        return top_assignments

    def PredictTraceIdSequences(
        self, process, incoming_span_partitions, outgoing_span_partitions
    ):
        assert len(incoming_span_partitions) == 1
        ep, incoming_spans = list(incoming_span_partitions.items())[0]
        outgoing_eps = self.GetOutgoingSpanOrder(outgoing_span_partitions)
        batch_size = 100
        topK = 5
        cnt = 0
        top_assignments = []
        for incoming_span in incoming_spans:
            if cnt % batch_size == 0:
                self.PopulateEpPairDistributions(
                    incoming_span_partitions, outgoing_span_partitions, outgoing_eps, cnt, min(len(incoming_spans), cnt + batch_size)
                )
                print("Finished %d spans, unassigned spans: %d"%(cnt, cnt_na))
            top_assignments.append(self.FindTopAssignments(
                incoming_span, outgoing_eps, outgoing_span_partitions, topK
            ))
            cnt += 1
        self.CreateMaxIndSetInstance(top_assignments, incoming_spans, outgoing_eps, outgoing_span_partitions)


    def CreateMaxIndSetInstance(top_assignments, incoming_spans, outgoing_eps, outgoing_span_partitions):
        #!TODO
        # create max independent set(MIS) based on top_assignments for each incoming span
        # each assignment consists of an ordered list of spans, starting with the incoming span and the subsequent spans are outgoing spans in order of dependence
        # For the MIS instance, add one vertex for each incoming span, and one vertex for each possible assignment
        # For an incoming span s, add edges between the top assignments for s (since only one of them need to be chosen)
        # For an assignment a1 for incoming span1 and an assignment a2 for incoming span2, add an edge between a1 and a2 if the assignments a1 and a2 intersect
        # Once the MIS instance is created, call the algorithm and post process the results
