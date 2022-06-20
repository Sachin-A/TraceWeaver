import math
import scipy.stats
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

    def FindTopKAssignments(self, in_span, out_eps, out_span_partitions, K):
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
                    # first ep
                    if (
                        i == 1
                        and in_span.start_mus < s.start_mus
                        and s.start_mus + s.duration_mus
                        < in_span.start_mus + in_span.duration_mus
                    ):
                        DfsTraverse(stack + [s])
                    # all other eps
                    elif (
                        i <= len(out_eps)
                        and last_span.start_mus + last_span.duration_mus < s.start_mus
                        and s.start_mus + s.duration_mus
                        < in_span.start_mus + in_span.duration_mus
                    ):
                        DfsTraverse(stack + [s])

        DfsTraverse([in_span])
        for i in range(len(top_assignments)):
            s, a = top_assignments[i]
            # undo negative scores
            top_assignments[i] = -s, a
        top_assignments.sort()
        return top_assignments

    def FindAssignments(self, process, in_span_partitions, out_span_partitions):
        assert len(in_span_partitions) == 1
        ep, in_spans = list(in_span_partitions.items())[0]
        out_eps = self.GetOutEpsInOrder(out_span_partitions)
        batch_size = 100
        topK = 10
        cnt = 0
        top_assignments = []
        for in_span in in_spans:
            if cnt % batch_size == 0:
                self.ComputeEpPairDistParams(
                    in_span_partitions,
                    out_span_partitions,
                    out_eps,
                    cnt,
                    min(len(in_spans), cnt + batch_size),
                )
                print("Finished %d spans" % (cnt))
            top_assignments.append(
                self.FindTopKAssignments(in_span, out_eps, out_span_partitions, topK)
            )
            cnt += 1
        assignments = self.GetAssignmentsMIS(top_assignments, in_spans)
        all_assignments = {}
        for ind in range(len(in_spans)):
            assignment = {}
            if len(assignments[ind]) > 0:
                assert len(out_eps) == len(assignments[ind]) - 1
                for ii in range(len(out_eps)):
                    assignment[out_eps[ii]] = assignments[ind][ii + 1]
            self.AddAssignment(
                in_spans[ind],
                assignment,
                all_assignments,
                out_span_partitions,
                out_eps,
            )
        return all_assignments

    # Create max independent set(MIS) based on top_assignments for each incoming span
    # Each assignment consists of an ordered list of spans, starting with the incoming span and the subsequent spans are outgoing spans in order of dependence
    # For the MIS instance
    #  - add one vertex for each possible assignment
    #  - for an incoming span s, add edges between the top assignments for s (since only one of them need to be chosen)
    #  - for an assignment a1 for incoming span1 and an assignment a2 for incoming span2, add an edge between a1 and a2 if the assignments a1 and a2 intersect
    def GetAssignmentsMIS(self, top_assignments, in_spans):
        batch_size = 100
        nbatches = math.ceil(float(len(in_spans)) / batch_size)
        mis_assignments = [[]] * len(in_spans)
        for b in range(nbatches):
            start = batch_size * b
            end = min(len(in_spans), batch_size * (b + 1))
            G = self.BuildMISInstance(top_assignments, start, end)
            mis = self.GetMIS(G)
            #mis = self.GetWeightedMIS(G, "weight")
            print("MIS- num assigned: %d/%d" % (len(mis), end - start))
            for in_span_ind, a_ind in mis:
                score, a = top_assignments[in_span_ind][a_ind]
                mis_assignments[in_span_ind] = a
        return mis_assignments

    def BuildMISInstance(self, top_assignments, start, end):
        G = nx.Graph()
        for ind1 in range(start, end):
            for i1 in range(len(top_assignments[ind1])):
                aid1 = (ind1, i1)
                score = top_assignments[ind1][i1][0]
                G.add_node(aid1, weight=1000-score)
                # add edges from previous assignments for the same incoming span
                for i0 in range(0, i1):
                    aid0 = (ind1, i0)
                    G.add_edge(aid0, aid1)
                # add edges from previous intersecting assignments for previous incoming spans
                for ind0 in range(start, ind1):
                    for i0 in range(len(top_assignments[ind0])):
                        if self.AssignmentIntersect(
                            top_assignments[ind0][i0][1],
                            top_assignments[ind1][i1][1],
                        ):
                            aid0 = (ind0, i0)
                            G.add_edge(aid0, aid1)
        return G

    def AssignmentIntersect(self, a1, a2):
        assert len(a1) == len(a2)
        for s1, s2 in zip(a1, a2):
            if s1.GetId() == s2.GetId():
                return True
        return False

    def GetMIS(self, G):
        '''
        mis = approximation.independent_set.maximum_independent_set(G)
        return mis
        '''
        best_mis = None
        best_score = -math.inf
        for i in range(20000):
            mis = nx.maximal_independent_set(G)
            score = sum([G.nodes[n]['weight'] for n in mis])
            #score = len(mis)
            if best_mis is None or score > best_score:
                best_mis = mis
                best_score = score
        return best_mis

    def GetWeightedMIS(self, G, weight):
        vcover = approximation.min_weighted_vertex_cover(G, weight=weight)
        return set(G.nodes()).difference(set(vcover))
