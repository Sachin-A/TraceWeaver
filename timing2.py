import math
import scipy.stats
from timing import Timing
import copy
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
                # min heap
                heapq.heappush(top_assignments, (score, stack))
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
        top_assignments.sort(reverse=True)
        return top_assignments

    def FindAssignments(self, process, in_span_partitions, out_span_partitions):
        assert len(in_span_partitions) == 1
        ep, in_spans = list(in_span_partitions.items())[0]
        out_eps = self.GetOutEpsInOrder(out_span_partitions)
        out_span_partitions_copy = copy.deepcopy(out_span_partitions)
        #!TODO: make this dynamic
        batch_size = 100
        batch_size_mis = 10
        topK = 5
        cnt = 0
        cnt_unassigned = 0
        all_assignments = {}
        top_assignments = []
        batch_in_spans = []
        for in_span in in_spans:
            if cnt % batch_size == 0:
                self.ComputeEpPairDistParams(
                    in_span_partitions,
                    out_span_partitions,
                    out_eps,
                    cnt,
                    min(len(in_spans), cnt + batch_size),
                )
                print("Finished %d spans, unassigned spans: %d" % (cnt, cnt_unassigned))
            top_assignments.append(
                self.FindTopKAssignments(in_span, out_eps, out_span_partitions_copy, topK)
            )
            batch_in_spans.append(in_span)
            cnt += 1

            if cnt % batch_size_mis == 0:
                assignments = self.GetAssignmentsMIS(top_assignments)
                assert len(assignments) == len(top_assignments) == len(batch_in_spans)
                for ind in range(len(assignments)):
                    assignment = {}
                    if len(assignments[ind]) > 0:
                        assert len(out_eps) == len(assignments[ind]) - 1
                        for ii in range(len(out_eps)):
                            assignment[out_eps[ii]] = assignments[ind][ii + 1]
                    self.AddAssignment(
                        batch_in_spans[ind],
                        assignment,
                        all_assignments,
                        out_span_partitions_copy,
                        out_eps,
                        delete_out_spans=True
                    )
                    cnt_unassigned += int(len(assignment) == 0)
                top_assignments = []
                batch_in_spans = []
        return all_assignments

    # Create max independent set(MIS) based on top_assignments for each incoming span
    # Each assignment consists of an ordered list of spans, starting with the incoming span and the subsequent spans are outgoing spans in order of dependence
    # For the MIS instance
    #  - add one vertex for each possible assignment
    #  - for an incoming span s, add edges between the top assignments for s (since only one of them need to be chosen)
    #  - for an assignment a1 for incoming span1 and an assignment a2 for incoming span2, add an edge between a1 and a2 if the assignments a1 and a2 intersect
    def GetAssignmentsMIS(self, top_assignments):
        mis_assignments = [[]] * len(top_assignments)
        G = self.BuildMISInstance(top_assignments)
        mis = self.GetMIS(G)
        #mis = self.GetWeightedMIS(G, "weight")
        #print("MIS- num assigned: %d/%d" % (len(mis), len(top_assignments)))
        for in_span_ind, a_ind in mis:
            score, a = top_assignments[in_span_ind][a_ind]
            mis_assignments[in_span_ind] = a
        return mis_assignments

    def BuildMISInstance(self, top_assignments):
        G = nx.Graph()
        for ind1 in range(len(top_assignments)):
            for i1 in range(len(top_assignments[ind1])):
                aid1 = (ind1, i1)
                score = top_assignments[ind1][i1][0]
                #!TODO: 10000 is sort of arbitrary to offset negative scores
                G.add_node(aid1, weight=10000.0 + score)
                # add edges from previous assignments for the same incoming span
                for i0 in range(0, i1):
                    aid0 = (ind1, i0)
                    G.add_edge(aid0, aid1)
                # add edges from previous intersecting assignments for previous incoming spans
                for ind0 in range(0, ind1):
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
