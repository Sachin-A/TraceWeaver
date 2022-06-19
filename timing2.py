import math
import scipy.stats
import copy
from timing import Timing
import sys
import networkx as nx
import heapq
from networkx.algorithms import approximation

VERBOSE = False


class Timing2(Timing):
    def __init__(self, all_spans, all_processes):
        super().__init__(all_spans, all_processes)
        self.all_spans = all_spans
        self.all_processes = all_processes
        self.services_times = {}

    def FindTopAssignments(
        self, in_span, out_eps, out_span_partitions, K
    ):
        global top_assignments
        top_assignments = []

        def DfsTraverse(stack):
            global top_assignments
            i = len(stack)
            if VERBOSE:
                print("DFSTraverse", i, out_eps, stack)
            last_span = stack[-1]
            if i == len(out_span_partitions) + 1:
                score = self.ScoreAssignment(stack)
                # negative score to make it a max heap
                heapq.heappush(top_assignments, (-score, stack))
                if len(top_assignments) > K:
                    heapq.heappop(top_assignments)
            else:
                ep = out_eps[i - 1]
                for s in out_span_partitions[ep]:
                    if i == 1:
                        # first ep
                        if in_span.start_mus < s.start_mus:
                            DfsTraverse(stack + [s])
                    elif i <= len(out_eps):
                        # all other eps
                        if (
                            last_span.start_mus + last_span.duration_mus < s.start_mus
                            and s.start_mus + s.duration_mus
                            < in_span.start_mus + in_span.duration_mus
                        ):
                            DfsTraverse(stack + [s])

        DfsTraverse([in_span])
        for i in range(len(top_assignments)):
            s, a = top_assignments[i]
            top_assignments[i] = -s, a  # restore original scores
        return top_assignments

    def FindAssignments(
        self, process, in_span_partitions, out_span_partitions
    ):
        assert len(in_span_partitions) == 1
        ep, in_spans = list(in_span_partitions.items())[0]
        out_eps = self.GetOutEpsInOrder(out_span_partitions)
        batch_size = 100
        topK = 5
        cnt = 0
        top_assignments = []
        out_span_partitions_copy = copy.deepcopy(out_span_partitions)
        for in_span in in_spans:
            if cnt % batch_size == 0:
                self.PopulateEpPairDistributions(
                    in_span_partitions,
                    out_span_partitions_copy,
                    out_eps,
                    cnt,
                    min(len(in_spans), cnt + batch_size),
                )
                print("Finished %d spans" % (cnt))
            top_assignments.append(
                self.FindTopAssignments(
                    in_span, out_eps, out_span_partitions_copy, topK
                )
            )
            cnt += 1
        assignments = self.CreateMaxIndSetInstance(
            top_assignments, in_spans, out_eps, out_span_partitions_copy
        )
        all_assignments = {}
        for ind in range(len(in_spans)):
            assignment = {}
            if len(assignments[ind]) > 0:
                assert len(out_eps) == len(assignments[ind]) - 1
                for ii in range(len(out_eps)):
                    assignment[out_eps[ii]] = assignments[ind][ii + 1]
            # print(in_spans[ind], assignment)
            # print(out_span_partitions_copy)
            self.AssignSpans(
                in_spans[ind],
                assignment,
                all_assignments,
                out_span_partitions_copy,
                out_eps,
            )
        return all_assignments

    def CreateMaxIndSetInstance(
        self, top_assignments, in_spans, out_eps, out_span_partitions
    ):
        # create max independent set(MIS) based on top_assignments for each incoming span
        # each assignment consists of an ordered list of spans, starting with the incoming span and the subsequent spans are outgoing spans in order of dependence
        # For the MIS instance, add one vertex for each possible assignment
        # For an incoming span s, add edges between the top assignments for s (since only one of them need to be chosen)
        # For an assignment a1 for incoming span1 and an assignment a2 for incoming span2, add an edge between a1 and a2 if the assignments a1 and a2 intersect
        def AssignmentIntersect(a1, a2):
            assert len(a1) == len(a2)
            for s1, s2 in zip(a1, a2):
                if s1.span_id == s2.span_id or s1.trace_id == s2.trace_id:
                    return True
            return False

        batch_size = 100
        nbatches =  math.ceil(float(len(in_spans))/batch_size)
        mis_assignments = [[]] * len(in_spans)
        for b in range(nbatches):
            start = batch_size * b
            end = min(len(in_spans), batch_size * (b + 1))
            G = nx.Graph()
            for ind in range(start, end):
                assignments = top_assignments[ind]
                for i in range(len(assignments)):
                    assert assignments[i][1][0] == in_spans[ind]
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
            best_mis = approximation.independent_set.maximum_independent_set(G)
            '''
            best_mis = None
            for i in range(20000):
                mis = nx.maximal_independent_set(G)
                if best_mis is None or len(mis) > len(best_mis):
                    best_mis = mis
            '''
            print("Best MIS %d/%d" % (len(best_mis), end - start))
            for span_ind, a_ind in best_mis:
                mis_assignments[span_ind] = top_assignments[span_ind][a_ind][1]
        return mis_assignments
