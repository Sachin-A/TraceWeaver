import math
import scipy.stats
import copy
from timing import Timing
import sys
import networkx as nx
import heapq

VERBOSE = False


class Timing2(Timing):
    def __init__(self, all_spans, all_processes):
        super().__init__(all_spans, all_processes)
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
                # negative score to make it a max heap
                heapq.heappush(top_assignments, (-score, stack))
                if len(top_assignments) > K:
                    heapq.heappop(top_assignments)
            else:
                ep = outgoing_eps[i - 1]
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
        for i in range(len(top_assignments)):
            s, a = top_assignments[i]
            top_assignments[i] = -s, a  # restore original scores
        return top_assignments

    def PredictTraceIdSequences(
        self, process, incoming_span_partitions, outgoing_span_partitions
    ):
        assert len(incoming_span_partitions) == 1
        ep, incoming_spans = list(incoming_span_partitions.items())[0]
        outgoing_eps = self.GetOutgoingSpanOrder(outgoing_span_partitions)
        batch_size = 100
        topK = 10
        cnt = 0
        top_assignments = []
        outgoing_span_partitions_copy = copy.deepcopy(outgoing_span_partitions)
        for incoming_span in incoming_spans:
            if cnt % batch_size == 0:
                self.PopulateEpPairDistributions(
                    incoming_span_partitions,
                    outgoing_span_partitions_copy,
                    outgoing_eps,
                    cnt,
                    min(len(incoming_spans), cnt + batch_size),
                )
                print("Finished %d spans" % (cnt))
            top_assignments.append(
                self.FindTopAssignments(
                    incoming_span, outgoing_eps, outgoing_span_partitions_copy, topK
                )
            )
            cnt += 1
        assignments = self.CreateMaxIndSetInstance(
            top_assignments, incoming_spans, outgoing_eps, outgoing_span_partitions_copy
        )
        assignments_dict = {}
        for ind in range(len(incoming_spans)):
            assignment = {}
            if len(assignments[ind]) > 0:
                assert len(outgoing_eps) == len(assignments[ind]) - 1
                for ii in range(len(outgoing_eps)):
                    assignment[outgoing_eps[ii]] = assignments[ind][ii + 1]
            # print(incoming_spans[ind], assignment)
            # print(outgoing_span_partitions_copy)
            self.AssignSpans(
                incoming_spans[ind],
                assignment,
                assignments_dict,
                outgoing_span_partitions_copy,
                outgoing_eps,
            )
        return assignments_dict

    def CreateMaxIndSetInstance(
        self, top_assignments, incoming_spans, outgoing_eps, outgoing_span_partitions
    ):
        #!TODO
        # create max independent set(MIS) based on top_assignments for each incoming span
        # each assignment consists of an ordered list of spans, starting with the incoming span and the subsequent spans are outgoing spans in order of dependence
        # For the MIS instance, add one vertex for each incoming span, and one vertex for each possible assignment
        # For an incoming span s, add edges between the top assignments for s (since only one of them need to be chosen)
        # For an assignment a1 for incoming span1 and an assignment a2 for incoming span2, add an edge between a1 and a2 if the assignments a1 and a2 intersect
        # Once the MIS instance is created, call the algorithm and post process the results
        def AssignmentIntersect(a1, a2):
            assert len(a1) == len(a2)
            for s1, s2 in zip(a1, a2):
                if s1.span_id == s2.span_id or s1.trace_id == s2.trace_id:
                    return True
            return False

        batch_size = 100
        nbatches = int((len(incoming_spans) + batch_size - 1) / batch_size)
        assignments_list = [[]] * len(incoming_spans)
        for b in range(nbatches):
            start = batch_size * b
            end = min(len(incoming_spans), batch_size * (b + 1))
            G = nx.Graph()
            for ind in range(start, end):
                assignments = top_assignments[ind]
                for i in range(len(assignments)):
                    assert assignments[i][1][0] == incoming_spans[ind]
                    aid = (ind, i)
                    G.add_node(aid)
                    for j in range(0, i):
                        aid2 = (ind, j)
                        G.add_edge(aid2, aid)
                    for ind0 in range(start, ind):
                        assignments0 = top_assignments[ind0]
                        for i0 in range(len(assignments0)):
                            if AssignmentIntersect(
                                assignments0[i0][1], assignments[i][1]
                            ):
                                aid0 = (ind0, i0)
                                G.add_edge(aid0, aid)
            best_mis = None
            for i in range(20000):
                mis = nx.maximal_independent_set(G)
                if best_mis is None or len(mis) > len(best_mis):
                    best_mis = mis
            print("Best MIS %d/%d" % (len(best_mis), end - start))
            for span_ind, a_ind in best_mis:
                # print("MIS", span_ind, a_ind, len(top_assignments), len(top_assignments[span_ind]), top_assignments[span_ind][a_ind])
                assignments_list[span_ind] = top_assignments[span_ind][a_ind][1]
                # assert incoming_spans[span_ind] == assignments_list[span_ind][0]
        return assignments_list
