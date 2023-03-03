import math
import scipy.stats
from timing import Timing
import copy
import sys
import networkx as nx
import heapq
from networkx.algorithms import approximation

VERBOSE = False
ERROR_WINDOW = False #True

class Timing2(Timing):
    def __init__(self):
        super().__init__()
        self.services_times = {}

    def FindTopKAssignments(self, in_span, out_eps, out_span_partitions, K):
        global top_assignments
        top_assignments = []
        if ERROR_WINDOW:
            error_window = 5
        else:
            error_window = 0

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
                        and float(in_span[2]) <= float(s[2]) + float(error_window)
                        and float(s[2]) + float(s[8])
                        <= float(in_span[2]) + float(in_span[8]) + float(error_window)
                    ):
                        DfsTraverse(stack + [s])
                    # all other eps
                    elif (
                        i <= len(out_eps)
                        and float(last_span[2]) + float(last_span[8]) <= float(s[2]) + float(error_window)
                        and float(s[2]) + float(s[8])
                        <= float(in_span[2]) + float(in_span[8]) + float(error_window)
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
        batch_size_mis = 20 #15
        topK = 4 #5
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

            if cnt % batch_size_mis == 0 or cnt == len(in_spans):
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
        mis = self.GetMIS2(top_assignments)
        #!TODO
        #G = self.BuildMISInstance(top_assignments)
        #mis = self.GetMIS(G)

        #mis = self.GetWeightedMIS(G, "weight")
        #print("MIS- num assigned: %d/%d" % (len(mis), len(top_assignments)))
        if mis is not None:
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
            if (s1[1], s1[3]) == (s2[1], s2[3]):
                return True
        return False

    def GetMIS2(self, top_assignments):
        best_mis = None
        best_score = -math.inf
        nrequests = len(top_assignments)
        dfs_stack = [[]]

        def GetTotalScore(asmts):
            score = sum([10000.0 + top_assignments[ind][i][0] for ind, i in asmts if i!=-1])
            return score

        nexplored = 0
        while len(dfs_stack) > 0:
            asmts = dfs_stack.pop()
            if len(asmts) == nrequests:
                score = GetTotalScore(asmts)
                nexplored += 1
                if score > best_score:
                    best_mis = asmts 
                    best_score = score
                    #print("Best mis till now", best_mis, best_score)
            else:
                ind1 = len(asmts)
                # add one entry for the unassigned (denoted by -1)
                if sum([1 for (ind0, i0) in asmts if i0==-1]) < 3:
                    dfs_stack.append(asmts + [(ind1, -1)])
                for i1 in range(len(top_assignments[ind1])):
                    intersect = False
                    for (ind0, i0) in asmts:
                        intersect = intersect or (i0!=-1 and self.AssignmentIntersect(
                            top_assignments[ind0][i0][1],
                            top_assignments[ind1][i1][1],
                        ))
                    if not intersect:
                        dfs_stack.append(asmts + [(ind1, i1)])
        best_mis = [(ind, i) for (ind, i) in best_mis if i != -1]
        print("best_mis", best_mis, "explored", nexplored)
        return best_mis

    def GetMIS(self, G):
        '''
        #mis = approximation.independent_set.maximum_independent_set(G)
        mis = approximation.maximum_independent_set(G)
        return mis
        '''
        best_mis = None
        best_score = -math.inf
        if len(G.nodes()) > 0:
            for i in range(40000):
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
