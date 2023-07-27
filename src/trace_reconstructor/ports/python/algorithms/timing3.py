import time
import math
import scipy.stats
from timing import Timing
import copy
import sys
import random, string
import networkx as nx
import heapq
import numpy as np
from networkx.algorithms import approximation

VERBOSE = False
EPS = 1e-6

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
    ):
        self.sid = sid
        self.trace_id = trace_id
        self.start_mus = start_mus
        self.duration_mus = duration_mus
        self.op_name = op_name
        self.references = references
        self.process_id = process_id
        self.span_kind = span_kind
        self.children_spans = []
        self.taken = False

    def AddChild(self, child_span_id):
        self.children_spans.append(child_span_id)

    def GetChildProcess(self):
        assert self.span_kind == "client"
        assert len(self.children_spans) == 1
        return all_processes[self.trace_id][
            all_spans[self.children_spans[0]].process_id
        ]

    def GetParentProcess(self):
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

class Timing7(Timing):
    def __init__(self, all_spans, all_processes):
        super().__init__(all_spans, all_processes)
        self.all_spans = all_spans
        self.all_processes = all_processes
        self.process = ''
        self.services_times = {}
        self.start_end = {}
        self.parallel = False
        self.normal = True
        self.instrumented_hops = []
        self.true_assignments = None
        self.distribution_values = {}
        self.distribution_values_true = {}
        self.large_delay = None
        self.per_span_candidates = {}
        self.time_windows = []
        self.span_windows = []
        self.skip_count_per_window = {}
        self.available_skips_per_window = {}
        self.true_skips = False
        self.true_dist = False
        self.overall_skip_budget = {}

    def ContainsSkip2(self, assignment):
        for (ep, i) in assignment:
            if i.trace_id == "None":
                return True
        return False

    def ContainsSkip(self, assignment):
        for i in assignment:
            if i.trace_id == "None":
                return True
        return False

    def BuildTrueDistributions(self, in_span_partitions, out_span_partitions, in_eps, out_eps, true_assignments):
        for in_ep in in_eps:
            for in_span in in_span_partitions[in_ep]:
                per_ep_gt = {}
                prev_index = 0
                prev_span = None
                for depth, out_ep in enumerate(out_eps):
                    out_span_id = true_assignments[out_ep][in_span.GetId()]
                    if out_span_id[0] == "Skip":
                        if depth == len(out_eps) - 1:
                            if prev_span != None and prev_index != 0:
                                if (prev_ep, in_ep) not in self.distribution_values_true:
                                    self.distribution_values_true[(prev_ep, in_ep)] = []
                                self.distribution_values_true[(prev_ep, in_ep)].append((in_span.start_mus + in_span.duration_mus) - (prev_span.start_mus + prev_span.duration_mus))
                    else:
                        for out_span in out_span_partitions[out_ep]:
                            if out_span.GetId() == out_span_id:
                                if prev_index == 0:
                                    if (in_ep, out_ep) not in self.distribution_values_true:
                                        self.distribution_values_true[(in_ep, out_ep)] = []
                                    self.distribution_values_true[(in_ep, out_ep)].append(out_span.start_mus - in_span.start_mus)
                                    prev_span = copy.deepcopy(out_span)
                                    prev_ep = copy.deepcopy(out_ep)
                                    prev_index += 1
                                else:
                                    if (prev_ep, out_ep) not in self.distribution_values_true:
                                        self.distribution_values_true[(prev_ep, out_ep)] = []
                                    self.distribution_values_true[(prev_ep, out_ep)].append(out_span.start_mus - (prev_span.start_mus + prev_span.duration_mus))
                                    prev_span = copy.deepcopy(out_span)
                                    prev_ep = copy.deepcopy(out_ep)
                                    prev_index += 1

                                if depth == len(out_eps) - 1:
                                    if (prev_ep, in_ep) not in self.distribution_values_true:
                                        self.distribution_values_true[(prev_ep, in_ep)] = []
                                    self.distribution_values_true[(prev_ep, in_ep)].append((in_span.start_mus + in_span.duration_mus) - (prev_span.start_mus + prev_span.duration_mus))

                                break

        for key in self.distribution_values_true.keys():
            self.services_times[key] = np.mean(self.distribution_values_true[key]), np.std(self.distribution_values_true[key])

    # def BuildDistributions2(self, process, in_span_partitions, out_span_partitions, in_eps, out_eps):
    #     for ep in in_span_partitions.keys():
    #         in_span_partitions[ep].sort(key = lambda x: float(x.start_mus))
    #     for ep in out_span_partitions.keys():
    #         out_span_partitions[ep].sort(key = lambda x: float(x.start_mus))

    #     in_span_ep, in_spans = list(in_span_partitions.items())[0]
    #     all_assignments = { ep: {} for ep in out_span_partitions.keys() }

    #     for ind in range(len(in_spans)):
    #         for ep, out_spans in out_span_partitions.items():
    #             all_assignments[ep][(in_spans[ind].trace_id, in_spans[ind].sid)] = ['NA', 'NA']

    #     out_eps = self.GetOutEpsInOrder(out_span_partitions)

    #     for ep in out_eps:
    #         out_spans = out_span_partitions[ep]
    #         j = 0
    #         for i in range(len(in_spans)):
    #             while float(out_spans[j].start_mus) < float(in_spans[i].start_mus):
    #                 j += 1
    #             if float(out_spans[j].start_mus) >= float(in_spans[i].start_mus) and ((i == (len(in_spans) - 1)) or (float(out_spans[j].start_mus) < float(in_spans[i + 1].start_mus))):
    #                 all_assignments[ep][(in_spans[i].trace_id, in_spans[i].sid)] = [out_spans[j].trace_id, out_spans[j].sid]
    #                 if (in_span_ep, ep) not in self.distribution_values2:
    #                     self.distribution_values2[(in_span_ep, ep)] = []
    #                 self.distribution_values2[(in_span_ep, ep)].append(out_spans[j].start_mus - in_spans[i].start_mus)
    #                 if
    #                 j += 1

    #     for key in self.distribution_values2.keys():
    #         self.services_times[key] = np.mean(self.distribution_values2[key]), np.std(self.distribution_values2[key])

    #     return all_assignments

    def BuildDistributions(self, process, in_span_partitions, out_span_partitions, in_eps, out_eps):

        spans = []
        for in_ep in in_eps:
            for span in in_span_partitions[in_ep]:
                span.ep = span.GetParentProcess()
            spans.extend(in_span_partitions[in_ep])
        for out_ep in out_eps:
            for span in out_span_partitions[out_ep]:
                span.ep = span.GetChildProcess()
            spans.extend(out_span_partitions[out_ep])
        spans.sort(key=lambda x: x.start_mus)
        self.large_delay = max([span.duration_mus for in_ep in in_eps for span in in_span_partitions[in_ep]])
        out_ep_order = {k: v for v, k in enumerate(out_eps)}

        for i, span in enumerate(spans):
            if span.span_kind == "client":
                # print(i, span)
                sent_mus = span.start_mus
                duration_mus = span.duration_mus
                parent_span = None
                parent_type = None
                for j, preceding_span in reversed(list(enumerate(spans[:i]))):
                    if (sent_mus + duration_mus) - preceding_span.start_mus > self.large_delay:
                        break
                    if preceding_span.span_kind == "server":
                        parent_span = preceding_span
                        parent_type = "server"
                        break
                    if ((preceding_span.span_kind == "client") and
                        (preceding_span.start_mus + preceding_span.duration_mus < span.start_mus) and
                        (out_ep_order[preceding_span.ep] < out_ep_order[span.ep])):
                        parent_span = preceding_span
                        parent_type = "client"
                        break
                if parent_span is not None:
                    if (parent_span.ep, span.ep) not in self.distribution_values:
                        self.distribution_values[(parent_span.ep, span.ep)] = []
                    if parent_type == "server":
                        self.distribution_values[(parent_span.ep, span.ep)].append(sent_mus - parent_span.start_mus)
                    elif parent_type == "client":
                        self.distribution_values[(parent_span.ep, span.ep)].append(sent_mus - (parent_span.start_mus + parent_span.duration_mus))

            elif span.span_kind == "server":
                sent_mus = span.start_mus
                duration_mus = span.duration_mus
                parent_span = None
                for j, preceding_span in reversed(list(enumerate(spans[:i]))):
                    if (sent_mus + duration_mus) - preceding_span.start_mus > self.large_delay:
                        break
                    if ((preceding_span.span_kind == "client") and
                        (preceding_span.start_mus + preceding_span.duration_mus < span.start_mus + span.duration_mus)):
                        parent_span = preceding_span
                        parent_type = "client"
                        break
                if parent_span is not None:
                    if (parent_span.ep, span.ep) not in self.distribution_values:
                        self.distribution_values[(parent_span.ep, span.ep)] = []
                    if parent_type == "client":
                        self.distribution_values[(parent_span.ep, span.ep)].append((sent_mus + duration_mus) - (parent_span.start_mus + parent_span.duration_mus))
                if (span.ep, span.ep) not in self.distribution_values:
                    self.distribution_values[(span.ep, span.ep)] = []
                self.distribution_values[(span.ep, span.ep)].append(duration_mus)

        for key in self.distribution_values.keys():
            self.services_times[key] = np.mean(self.distribution_values[key]), np.std(self.distribution_values[key])

    def GenerateRandomID(self):
        x = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(16))
        return x

    def AddToCandidatesList(self, stack):
        if (stack[0].trace_id, stack[0].sid) not in self.per_span_candidates:
            self.per_span_candidates[(stack[0].trace_id, stack[0].sid)] = 0

        self.per_span_candidates[(stack[0].trace_id, stack[0].sid)] += 1

    def FindTopKAssignments(self, in_eps, in_span, out_eps, out_span_partitions, K, invocation_graph):
        # f = open("dump.txt", "a")

        global top_assignments
        top_assignments = []

        normalized = False
        for ep in out_eps:
            if self.overall_skip_budget[ep] > 0:
                normalized = True
                break

        if self.true_skips == False:
            for ep in out_eps:
                out_span_partitions[ep].append(None)

        # Handle skip spans in this version

        def DfsTraverse2(stack, invocation_graph):
            global top_assignments
            i = len(stack)
            if VERBOSE:
                print("DFSTraverse2", i, out_eps, stack)
            if i == len(out_span_partitions) + 1:
                stack2 = []
                for s in stack:
                    stack2.append(s[1])
                # for st in stack2:
                    # print(st)
                # input()
                self.AddToCandidatesList(stack2)
                # if self.ContainsSkip(stack2):
                #     score = self.ScoreAssignmentWithSkip(stack2, normalized)
                # elif self.parallel:
                #     score = self.ScoreAssignmentParallel(stack2, normalized)
                # else:
                #     score = self.ScoreAssignmentSequential(stack2, normalized)
                score = self.ScoreAssignmentAsPerInvocationGraph(stack, invocation_graph, out_eps, normalized)
                # min heap
                heapq.heappush(top_assignments, (score, stack))
                if len(top_assignments) > K:
                    heapq.heappop(top_assignments)
            # elif i in self.instrumented_hops:
            #     ep = out_eps[i - 1]
            #     span_id = self.true_assignments[ep][in_span.GetId()]
            #     for s in out_span_partitions[ep]:
            #         if s.GetId() == span_id:
            #             DfsTraverse2(stack + [s])
            #             break
            else:
                ep = out_eps[i - 1]
                if self.true_skips == True and self.true_assignments[ep][in_span.GetId()][0] == "Skip":
                    new_span_id = self.GenerateRandomID()
                    skip_span = Span("None", new_span_id, "None", "None", "None", "None", "None", "None")
                    DfsTraverse2(stack + [(ep, skip_span)], invocation_graph)
                else:
                    for x, s in enumerate(out_span_partitions[ep]):
                        if self.true_skips == False and s == None:
                            skip_span = self.FetchSkipFromWindow(ep, in_span.start_mus)
                            if skip_span != None:
                                DfsTraverse2(stack + [(ep, skip_span)], invocation_graph)
                        else:
                            before_eps = invocation_graph.in_edges(ep)
                            candidate = True

                            if (
                                in_span.start_mus > s.start_mus or
                                s.start_mus + s.duration_mus > in_span.start_mus + in_span.duration_mus
                            ):
                                candidate = False
                                continue

                            b_span = "None"
                            for (before_ep, self_ep) in before_eps:
                                # print(before_ep, self_ep)
                                # input()
                                # ep_index = out_eps.index(before_ep)
                                # b_ep = 250[ep_index + 1][0]
                                # b_span = stack[ep_index + 1][1]
                                # assert b_ep == before_ep

                                idx = next((i for i, (v, *_) in enumerate(stack) if v == before_ep), None)
                                # if idx == None:
                                #     print(topological_sort_grouped(invocation_graph))
                                #     print(before_ep, self_ep)
                                assert idx != None
                                b_ep = stack[idx][0]
                                b_span = stack[idx][1]
                                assert b_ep == before_ep

                                if b_span.trace_id == "None":
                                    continue

                                if (
                                    b_span.start_mus + b_span.duration_mus > s.start_mus
                                ):
                                    candidate = False
                                    continue

                            if candidate:
                                # if self.process == "search":
                                if b_span == "None":
                                    b_span = stack[0][1]
                                # f.write(str(b_span) + ", " + str(s) + "\n")
                                DfsTraverse2(stack + [(ep, s)], invocation_graph)

        def DfsTraverse(stack, depth, l_non_skip_depth, l_start, l_duration):
            # print(depth, l_non_skip_depth)
            global top_assignments
            i = len(stack)
            if VERBOSE:
                print("DFSTraverse", i, out_eps, stack)
            last_span = stack[-1]
            if i == len(out_span_partitions) + 1:
                # print("X")
                self.AddToCandidatesList(stack)
                if self.ContainsSkip(stack):
                    score = self.ScoreAssignmentWithSkip(stack, normalized)
                elif self.parallel:
                    score = self.ScoreAssignmentParallel(stack, normalized)
                else:
                    score = self.ScoreAssignmentSequential(stack, normalized)
                # min heap
                # if self.true_skips == True:
                #     print(stack)
                #     print(score)
                #     input()
                heapq.heappush(top_assignments, (score, stack))
                if len(top_assignments) > K:
                    heapq.heappop(top_assignments)
            # elif i in self.instrumented_hops:
            #     ep = out_eps[i - 1]
            #     span_id = self.true_assignments[ep][in_span.GetId()]
            #     for s in out_span_partitions[ep]:
            #         if s.GetId() == span_id:
            #             DfsTraverse(stack + [s])
            #             break
            else:
                # print("J")
                ep = out_eps[i - 1]
                # print(in_span.GetId()[0], self.true_skips)
                # if in_span.GetId()[0] == "1d88e7719f1b3be7" and self.true_skips == True:
                    # print("A")
                    # input()
                if self.true_skips == True and self.true_assignments[ep][in_span.GetId()][0] == "Skip":
                    new_span_id = self.GenerateRandomID()
                    skip_span = Span(
                        "None",
                        new_span_id,
                        "None",
                        "None",
                        "None",
                        "None",
                        "None",
                        "None",
                    )
                    # if in_span.GetId()[0] == "1d88e7719f1b3be7" and self.true_skips == True:
                        # print("B")
                        # input()
                    DfsTraverse(stack + [skip_span], depth + 1, l_non_skip_depth, last_span.start_mus, last_span.duration_mus)
                else:
                    for x, s in enumerate(out_span_partitions[ep]):
                        # print("Span num:", x)
                        if self.true_skips == False and s == None:
                            # if in_span.GetId()[0] == "1d88e7719f1b3be7" and self.true_skips == True:
                                # print("C")
                                # input()
                            skip_span = self.FetchSkipFromWindow(ep, in_span.start_mus)
                            # print("None:", len(stack))
                            if skip_span != None:
                                DfsTraverse(stack + [skip_span], depth + 1, l_non_skip_depth, last_span.start_mus, last_span.duration_mus)
                        else:
                            # if in_span.GetId()[0] == "1d88e7719f1b3be7" and self.true_skips == True:
                                # print([x.sid for x in out_span_partitions[ep][:7]])
                                # print("D")
                                # input()
                            # parallel eps
                            if self.parallel:
                                if (
                                    in_span.start_mus <= s.start_mus
                                    and s.start_mus + s.duration_mus
                                    <= in_span.start_mus + in_span.duration_mus
                                ):
                                    # print("Parallel:", len(stack))
                                    DfsTraverse(stack + [s], depth + 1, l_non_skip_depth + 1, None, None)

                            # Sequential eps
                            else:
                                # if in_span.GetId()[0] == "1d88e7719f1b3be7" and self.true_skips == True:
                                    # print("E")
                                    # print(last_span)
                                    # print(s)
                                    # print(l_non_skip_depth)
                                    # print(in_span.start_mus, s.start_mus, in_span.start_mus < s.start_mus)
                                    # print(s.start_mus + s.duration_mus, in_span.start_mus + in_span.duration_mus, s.start_mus + s.duration_mus < in_span.start_mus + in_span.duration_mus)
                                    # input()
                                if last_span.trace_id == "None":
                                    if (
                                        l_non_skip_depth == 1
                                        and in_span.start_mus <= s.start_mus
                                        and s.start_mus + s.duration_mus
                                        <= in_span.start_mus + in_span.duration_mus
                                    ):
                                        # print("E1")
                                        # print("Seq1-prevnone:", len(stack))
                                        # if self.process == "search":
                                        # f.write(str(in_span) + ", " + str(s) + "\n")
                                        DfsTraverse(stack + [s], depth + 1, l_non_skip_depth + 1, None, None)
                                    # all other eps
                                    elif (
                                        l_non_skip_depth <= len(out_eps)
                                        and l_start + l_duration <= s.start_mus
                                        and s.start_mus + s.duration_mus
                                        <= in_span.start_mus + in_span.duration_mus
                                    ):
                                        # print("E2")
                                        # print("Seq2-prevnone:", len(stack))
                                        # if self.process == "search":
                                        # f.write(str(last_span) + ", " + str(s) + "\n")
                                        DfsTraverse(stack + [s], depth + 1, l_non_skip_depth + 1, None, None)
                                else:
                                    # first ep
                                    if (
                                        i == 1
                                        and in_span.start_mus <= s.start_mus
                                        and s.start_mus + s.duration_mus
                                        <= in_span.start_mus + in_span.duration_mus
                                    ):
                                        # print("E3")
                                        # print("Seq1:", len(stack))
                                        # if self.process == "search":
                                        # f.write(str(in_span) + ", " + str(s) + "\n")
                                        DfsTraverse(stack + [s], depth + 1, l_non_skip_depth + 1, None, None)
                                    # all other eps
                                    elif (
                                        i <= len(out_eps)
                                        and last_span.start_mus + last_span.duration_mus <= s.start_mus
                                        and s.start_mus + s.duration_mus
                                        <= in_span.start_mus + in_span.duration_mus
                                    ):
                                        # print("E4")
                                        # print("Seq2:", len(stack))
                                        # if self.process == "search":
                                        # f.write(str(last_span) + ", " + str(s) + "\n")
                                        DfsTraverse(stack + [s], depth + 1, l_non_skip_depth + 1, None, None)

        # DfsTraverse([in_span], 1, 1, None, None)
        # top_assignments.sort(reverse=True)
        # if self.true_skips == False:
        #     for ep in out_eps:
        #         out_span_partitions[ep].pop()

        # # f.close()

        # return top_assignments

        in_ep = in_eps[0]

        DfsTraverse2([(in_ep, in_span)], invocation_graph)
        top_assignments2 = []
        for assignment in top_assignments:
            s_assignment = (assignment[0], [s[1] for s in assignment[1]])
            top_assignments2.append(s_assignment)
        top_assignments2.sort(reverse=True)
        if self.true_skips == False:
            for ep in out_eps:
                out_span_partitions[ep].pop()

        # f.close()

        return top_assignments2

    def GetSpanIDNotation(self, out_eps, assignment, type1):
        span_id_notation = []

        if type1:
            for i in range(1, len(assignment)):
                span_id_notation.append(assignment[i].GetId())
        else:
            for out_ep in out_eps:
                span_id_notation.append(assignment[out_ep].GetId())
        return span_id_notation

    def ComputeEpPairDistParams(
        self,
        in_span_partitions,
        out_span_partitions,
        out_eps,
        in_span_start,
        in_span_end,
    ):
        def ComputeDistParams(ep1, ep2, t1, t2):
            t1 = t1[in_span_start:in_span_end]
            t2 = t2[in_span_start:in_span_end]
            assert len(t1) == len(t2)
            mean = (sum(t2) - sum(t1)) / len(t1)
            batch_means = []
            nbatches = 10
            batch_size = math.ceil(float(len(t1)) / nbatches)
            for i in range(nbatches):
                start = i * batch_size
                end = min(len(t1), (i + 1) * batch_size)
                if end - start > 0:
                    batch_means.append(
                        (sum(t2[start:end]) - sum(t1[start:end])) / (end - start)
                    )
            std = math.sqrt(batch_size) * scipy.stats.tstd(batch_means)
            if VERBOSE:
                print(
                    "Computing ep pair (%s, %s), distribution params: %f, %f"
                    % (ep1, ep2, mean, std)
                )
            self.services_times[(ep1, ep2)] = mean, std

        if self.parallel:
            for i in range(len(out_eps)):
                ep1 = list(in_span_partitions.keys())[0]
                ep2 = out_eps[i]
                t1 = sorted([s.start_mus for s in in_span_partitions[ep1]])
                t2 = sorted([s.start_mus for s in out_span_partitions[ep2]])
                ComputeDistParams(ep1, ep2, t1, t2)
        else:
            # between incoming -- first outgoing
            ep1 = list(in_span_partitions.keys())[0]
            ep2 = out_eps[0]
            t1 = sorted([s.start_mus for s in in_span_partitions[ep1]])
            t2 = sorted([s.start_mus for s in out_span_partitions[ep2]])
            ComputeDistParams(ep1, ep2, t1, t2)

            # between outgoing -- outgoing
            for i in range(len(out_eps) - 1):
                ep1 = out_eps[i]
                ep2 = out_eps[i + 1]
                t1 = sorted(
                    [s.start_mus + s.duration_mus for s in out_span_partitions[ep1]]
                )
                t2 = sorted([s.start_mus for s in out_span_partitions[ep2]])
                ComputeDistParams(ep1, ep2, t1, t2)

            # between last outgoing -- incoming
            ep1 = out_eps[-1]
            ep2 = list(in_span_partitions.keys())[0]
            t1 = sorted([s.start_mus + s.duration_mus for s in out_span_partitions[ep1]])
            t2 = sorted([s.start_mus + s.duration_mus for s in in_span_partitions[ep2]])
            ComputeDistParams(ep1, ep2, t1, t2)

    def ComputeEpPairDistParams2(
        self,
        in_span_partitions,
        out_span_partitions,
        out_eps,
        in_span_start,
        in_span_end,
    ):

        def SetStartEnd(ep1, ep2, t1, t2):
            if in_span_start == 0:
                out_span_start = 0
            else:
                out_span_start = self.start_stop[(ep1, ep2)][3] + 1

            t1 = t1[in_span_start:in_span_end]
            t1_sorted_finish_times = sorted([s.start_mus + s.duration_mus for s in t1])
            last_span = t1_sorted_finish_times[-1]

            out_span_end = None
            for i, span in enumerate(t2[out_span_start:]):
                if span.start_mus + span.duration_mus > last_span.start_mus + last_span.duration_mus:
                    break
                else:
                    out_span_end = i
            if out_span_end != None:
                t2 = t2[out_span_start:out_span_end]
            else:
                assert(False)

            x = in_span_end - in_span_start
            y = out_span_end - out_span_start
            diff = x - y
            if diff > 0:
                in_span_end -= diff
            else:
                out_span_end += diff
            self.start_end[(ep1, ep2)] = [in_span_start, in_span_end, out_span_start, out_span_end]
            print(self.start_end[(ep1, ep2)])

        def ComputeDistParams(ep1, ep2, t1, t2):
            assert len(t1) == len(t2)
            mean = (sum(t2) - sum(t1)) / len(t1)
            batch_means = []
            nbatches = 10
            batch_size = math.ceil(float(len(t1)) / nbatches)
            for i in range(nbatches):
                start = i * batch_size
                end = min(len(t1), (i + 1) * batch_size)
                if end - start > 0:
                    batch_means.append(
                        (sum(t2[start:end]) - sum(t1[start:end])) / (end - start)
                    )
            std = math.sqrt(batch_size) * scipy.stats.tstd(batch_means)
            if VERBOSE:
                print(
                    "Computing ep pair (%s, %s), distribution params: %f, %f"
                    % (ep1, ep2, mean, std)
                )
            self.services_times[(ep1, ep2)] = mean, std

        if self.parallel:
            for i in range(len(out_eps)):
                ep1 = list(in_span_partitions.keys())[0]
                ep2 = out_eps[i]
                t1 = sorted(in_span_partitions[ep1], key=lambda s: s.start_mus)
                t2 = sorted(out_span_partitions[ep2], key=lambda s: s.start_mus)
                ComputeDistParams(ep1, ep2, t1, t2)
        else:
            # between incoming -- first outgoing
            ep1 = list(in_span_partitions.keys())[0]
            ep2 = out_eps[0]
            t1 = sorted(in_span_partitions[ep1], key=lambda s: s.start_mus)
            t2 = sorted(out_span_partitions[ep2], key=lambda s: s.start_mus)
            SetStartEnd(ep1, ep2, t1, t2)
            t1 = [s.start_mus for s in t1[s1: e1]]
            t2 = [s.start_mus for s in t1[s1: e1]]
            s1, e1, s2, e2 = self.start_stop[(ep1, ep2)]
            ComputeDistParams(ep1, ep2, t1, t2)

            # between outgoing -- outgoing
            for i in range(len(out_eps) - 1):
                ep1 = out_eps[i]
                ep2 = out_eps[i + 1]
                t1 = sorted(out_span_partitions[ep1], key=lambda s: s.start_mus + s.duration_mus)
                t2 = sorted(out_span_partitions[ep2], key=lambda s: s.start_mus)
                SetStartEnd(ep1, ep2, t1, t2)
                s1, e1, s2, e2 = self.start_stop[(ep1, ep2)]
                t1 = [s.start_mus + s.duration_mus for s in t1[s1: e1]]
                t2 = [s.start_mus for s in t2[s2: e2]]
                ComputeDistParams(ep1, ep2, t1, t2)

            # between last outgoing -- incoming
            ep1 = out_eps[-1]
            ep2 = list(in_span_partitions.keys())[0]
            t1 = sorted(out_span_partitions[ep1], key=lambda s: s.start_mus + s.duration_mus)
            t2 = sorted(in_span_partitions[ep2], key=lambda s: s.start_mus + s.duration_mus)
            SetStartEnd(ep1, ep2, t1, t2)
            s1, e1, s2, e2 = self.start_stop[(ep1, ep2)]
            t1 = [s.start_mus + s.duration_mus for s in t1[s1: e1]]
            t2 = [s.start_mus + s.duration_mus for s in t2[s2: e2]]
            ComputeDistParams(ep1, ep2, t1, t2)

    def ComputeEpPairDistParams3(
        self,
        in_span_partitions,
        out_span_partitions,
        out_eps,
        in_span_start,
        in_span_end,
        invocation_graph
    ):

        def ComputeDistParams(ep1, ep2, t1, t2):
            t1 = t1[in_span_start:in_span_end]
            t2 = t2[in_span_start:in_span_end]
            assert len(t1) == len(t2)
            mean = (sum(t2) - sum(t1)) / len(t1)
            if len(t1) == 0:
                print("len(t1)")
                input()
            batch_means = []
            nbatches = 10
            batch_size = math.ceil(float(len(t1)) / nbatches)
            if nbatches == 0:
                print("nbatches")
                input()
            for i in range(nbatches):
                start = i * batch_size
                end = min(len(t1), (i + 1) * batch_size)
                if end - start > 0:
                    batch_means.append(
                        (sum(t2[start:end]) - sum(t1[start:end])) / (end - start)
                    )
            std = math.sqrt(batch_size) * scipy.stats.tstd(batch_means)
            if VERBOSE:
                print(
                    "Computing ep pair (%s, %s), distribution params: %f, %f"
                    % (ep1, ep2, mean, std)
                )
            self.services_times[(ep1, ep2)] = mean, std

        in_ep = list(in_span_partitions.keys())[0]

        for out_ep in out_span_partitions.keys():

            if len(invocation_graph.in_edges(out_ep)) == 0:
                t1 = sorted([s.start_mus for s in in_span_partitions[in_ep]])
                t2 = sorted([s.start_mus for s in out_span_partitions[out_ep]])
                ComputeDistParams(in_ep, out_ep, t1, t2)

            before_eps = invocation_graph.in_edges(out_ep)

            for (before_ep, self_ep) in before_eps:

                if not self.AlsoNonPrimaryAncestor(before_ep, self_ep, invocation_graph):

                    if before_ep == in_ep:
                        t1 = sorted([s.start_mus for s in in_span_partitions[before_ep]])
                        t2 = sorted([s.start_mus for s in out_span_partitions[self_ep]])
                        ComputeDistParams(before_ep, self_ep, t1, t2)

                    else:
                        t1 = sorted([s.start_mus + s.duration_mus for s in out_span_partitions[before_ep]])
                        t2 = sorted([s.start_mus for s in out_span_partitions[self_ep]])
                        ComputeDistParams(before_ep, self_ep, t1, t2)

            t1 = sorted([s.start_mus + s.duration_mus for s in out_span_partitions[out_ep]])
            t2 = sorted([s.start_mus + s.duration_mus for s in in_span_partitions[in_ep]])
            ComputeDistParams(out_ep, in_ep, t1, t2)

    def FetchSkipFromWindow(
        self,
        ep,
        start_mus
    ):

        def FindWindow(key, windows):
            return windows.index(max(i for i in windows if i <= key))

        # def FindWindow(start_mus, windows):
        #     index = None
        #     start = 0
        #     end = len(windows)
        #     while start <= end:
        #         mid = (start + end) // 2
        #         if mid >= len(windows):  # out of bounds
        #             break
        #         if windows[mid] <= start_mus:
        #             index = mid
        #             start = mid + 1
        #         else:
        #             end = mid - 1
        #     return index

        self.time_windows.sort(key = lambda x: x[0])
        index = FindWindow(start_mus, [x[0] for x in self.time_windows])
        if index == None:
            assert False
        window = self.time_windows[index][:2]

        if len(self.available_skips_per_window[ep][window]) <= 0:
            return None

        minval = min(self.available_skips_per_window[ep][window], key = lambda x: x[1])
        pos = self.available_skips_per_window[ep][window].index(minval)
        self.available_skips_per_window[ep][window][pos][1] += 1

        return self.available_skips_per_window[ep][window][pos][0]

    def DetectBoundaries(
        self,
        in_span_partitions,
        out_span_partitions,
        in_eps,
        out_eps
    ):
        pass

    def TallySkipSpans(
        self,
        in_span_partitions,
        out_span_partitions,
        in_eps,
        out_eps,
        batch_size_mis
    ):

        def WaterFill(window_diffs, window_counts, skip_budget, ep):

            if skip_budget <= 0:
                return

            num_windows = len(window_diffs)
            window_keys = copy.deepcopy(sorted(self.time_windows, key = lambda x: x[0]))

            index_to_key = {}
            for i, window_key in enumerate(window_keys):
                index_to_key[i] = window_key[:2]

            existing_spans = np.zeros(len(window_counts))
            expected_spans = np.zeros(len(window_counts))
            for i in range(len(window_counts)):
                existing_spans[i] = copy.deepcopy((window_counts[index_to_key[i]]))
                expected_spans[i] = copy.deepcopy((window_keys[i][2]))

            # Sort the windows in decreasing order of existing spans
            sorted_indices = np.argsort(existing_spans)[::-1]
            sorted_existing_spans = existing_spans[sorted_indices]

            # Initialize resource allocation vector
            skip_allocation = np.zeros(num_windows)

            # Calculate the max window span count (i.e., water level in waterfilling)
            lambda_ = 0

            for i in range(num_windows):
                lambda_ = (skip_budget + np.sum(sorted_existing_spans[:i + 1])) // (i + 1)
                total_remaining = (skip_budget + np.sum(sorted_existing_spans[:i + 1])) % (i + 1)
                if lambda_ <= sorted_existing_spans[i]:
                    break

            # Allocate additional resources to windows based on water-filling
            remaining = 0
            for i in range(num_windows):
                remaining += max(lambda_ - sorted_existing_spans[i], 0) - min(max(lambda_ - sorted_existing_spans[i], 0), expected_spans[i] - sorted_existing_spans[i])
                skip_allocation[sorted_indices[i]] = min(max(lambda_ - sorted_existing_spans[i], 0), expected_spans[i] - sorted_existing_spans[i])
            total_remaining += remaining

            while total_remaining > 0:
                no_change = True
                for i in reversed(range(num_windows)):
                    if total_remaining > 0 and skip_allocation[sorted_indices[i]] < (expected_spans[i] - sorted_existing_spans[i]):
                        skip_allocation[sorted_indices[i]] += 1
                        no_change = False
                        total_remaining -= 1
                if no_change:
                    break

            for i in range(len(window_counts)):
                self.skip_count_per_window[ep][index_to_key[i]] = skip_allocation[i]

            return

        def TackleMismatch(ep):

            skip_budget = self.overall_skip_budget[ep]

            self.skip_count_per_window[ep] = {}
            self.available_skips_per_window[ep] = {}
            window_counts = {}
            window_diffs = {}

            for (window_start, window_end, expected_count) in self.time_windows:

                if (window_start, window_end) not in self.skip_count_per_window[ep]:
                    self.skip_count_per_window[ep][(window_start, window_end)] = 0

                count = 0
                # TODO: calculate mean_span_time per window
                mean_span_time = np.mean([i.duration_mus for i in out_span_partitions[ep]])
                for span in out_span_partitions[ep]:
                    if span.start_mus > window_start and span.start_mus <= window_end:
                        count += 1
                window_diffs[(window_start, window_end)] = max(expected_count - count, 0)
                window_counts[(window_start, window_end)] = count

            skip_budget_copy = skip_budget
            # while skip_budget > 0:
            #     no_change = True
            #     for (window_start, window_end, expected_count) in self.time_windows:
            #         if window_diffs[(window_start, window_end)] > self.skip_count_per_window[ep][(window_start, window_end)]:
            #             self.skip_count_per_window[ep][(window_start, window_end)] += 1
            #             skip_budget -= 1
            #             no_change = False
            #         if skip_budget == 0:
            #             break
            #     if no_change:
            #         break

            # x = copy.deepcopy([v for k, v in self.skip_count_per_window[ep].items()])

            WaterFill(window_diffs, window_counts, skip_budget_copy, ep)

            # y = copy.deepcopy([v for k, v in self.skip_count_per_window[ep].items()])

            # if skip_budget_copy > 0:
            #     print(x)
            #     print(y)
            #     print(x == y)
            #     input()

            for (window_start, window_end, _) in self.time_windows:

                if (window_start, window_end) not in self.available_skips_per_window[ep]:
                    self.available_skips_per_window[ep][(window_start, window_end)] = []

                for i in range(int(self.skip_count_per_window[ep][(window_start, window_end)])):

                    new_span_id = self.GenerateRandomID()
                    skip_span = Span(
                            "None",
                            new_span_id,
                            "None",
                            "None",
                            "None",
                            "None",
                            "None",
                            "None",
                        )
                    self.available_skips_per_window[ep][(window_start, window_end)].append([skip_span, 0])

        self.skip_count_per_window = {}

        for ep in in_eps:
            in_span_partitions[ep].sort(key = lambda x: float(x.start_mus))
        for ep in out_eps:
            out_span_partitions[ep].sort(key = lambda x: float(x.start_mus))
            self.overall_skip_budget[ep] = len(in_span_partitions[in_eps[0]]) - len(out_span_partitions[ep])

        window_start = in_span_partitions[in_eps[0]][0].start_mus
        final_span = sorted(in_span_partitions[in_eps[0]], key = lambda x: float(x.start_mus) + float(x.duration_mus))[-1]
        final_window_end = final_span.start_mus + final_span.duration_mus

        len_spans = len(in_span_partitions[in_eps[0]])
        for i in range(0, len_spans):
            if (i != 0 and i != len_spans - 1 and i % batch_size_mis == 0):
                # print(i, batch_size_mis)
                window_end = in_span_partitions[in_eps[0]][i].start_mus + in_span_partitions[in_eps[0]][i].duration_mus
                self.time_windows.append((window_start, window_end, batch_size_mis))
                window_start = window_end
            elif i == len_spans - 1:
                # window_end = in_span_partitions[in_eps[0]][i].start_mus + in_span_partitions[in_eps[0]][i].duration_mus
                window_end = final_window_end
                self.time_windows.append((window_start, window_end, batch_size_mis))

        for ep in out_eps:
            TackleMismatch(ep)

    def CreateWindows(self, in_span_partitions, in_eps, max_size, threshold):

        windows = []
        current_count = 1

        for i, span in enumerate(in_span_partitions[in_eps[0]]):

            if i != 0:
                if i == len(in_span_partitions[in_eps[0]]) - 1:
                    current_count = 0
                    window_end = i
                    windows.append((window_start, window_end))
                elif (in_span_partitions[in_eps[0]][i + 1].start_mus - span.start_mus) > threshold:
                    current_count = 0
                    window_end = i
                    windows.append((window_start, window_end))
                    window_start = i + 1
                elif current_count == max_size:
                    current_count = 0
                    window_end = i
                    windows.append((window_start, window_end))
                    window_start = i + 1
            else:
                window_start = i

            current_count += 1

        return windows

    def FindAssignments(self, process, in_span_partitions, out_span_partitions, parallel, instrumented_hops, true_assignments, invocation_graph, true_skips = False, true_dist = False):
        assert len(in_span_partitions) == 1
        self.process = process
        self.parallel = parallel
        self.instrumented_hops = instrumented_hops
        self.true_assignments = true_assignments
        self.per_span_candidates = {}
        self.true_skips = true_skips
        self.true_dist = true_dist
        for ep in out_span_partitions.keys():
            for key in true_assignments[ep].keys():
                self.per_span_candidates[key] = 0
        span_to_top_assignments = {}
        in_eps, in_spans = list(in_span_partitions.items())[0]
        in_eps = [in_eps] if isinstance(in_eps, str) else in_eps
        out_eps = self.GetOutEpsInOrder(out_span_partitions)
        out_span_partitions_copy = copy.deepcopy(out_span_partitions)
        # TODO: make this dynamic
        sorted_durations = [i.duration_mus for i in sorted(in_span_partitions[in_eps[0]], key=lambda s: s.duration_mus)]
        batch_size = 100
        batch_size_mis = 10
        topK = 5
        self.normal = True
        self.span_windows = self.CreateWindows(in_span_partitions, in_eps, batch_size_mis, np.percentile(sorted_durations, 50))
        window_ends = [i[1] for i in self.span_windows]
        cnt = 0
        cnt_unassigned = 0
        not_best_count = 0
        all_assignments = {}
        all_topk_assignments = {}
        top_assignments = []
        batch_in_spans = []

        # print(len(self.span_windows))
        # print(self.span_windows)
        # print([(i[1] - i[0] + 1) for i in self.span_windows])

        count = 0
        for span in in_span_partitions[in_eps[0]]:
            for ep in out_eps:
                if self.true_assignments[ep][span.GetId()][0] == "Skip":
                    count += 1
        print("True skips: ", count, "\n")

        self.TallySkipSpans(in_span_partitions, out_span_partitions, in_eps, out_eps, batch_size_mis)

        equal_eps = []
        for ep in out_eps:
            print("Endpoint:", ep + ", ", "Num spans:", len(out_span_partitions[ep]))
            if self.overall_skip_budget[ep] == 0:
                equal_eps.append(ep)


        if self.true_dist:
            self.BuildTrueDistributions(in_span_partitions, out_span_partitions, in_eps, out_eps, true_assignments)
        else:
            self.BuildDistributions(process, in_span_partitions, out_span_partitions, in_eps, out_eps)
            # self.BuildDistributions2(process, in_span_partitions, out_span_partitions, in_eps, out_eps)

        for in_span in in_spans:
            if cnt % batch_size == 0:
                # self.ComputeEpPairDistParams(in_span_partitions, out_span_partitions, out_eps, cnt, min(len(in_spans), cnt + batch_size))
                # self.ComputeEpPairDistParams3(in_span_partitions, out_span_partitions, out_eps, cnt, min(len(in_spans), cnt + batch_size), invocation_graph)
                print("Finished %d spans, unassigned spans: %d" % (cnt, cnt_unassigned))

            top_k = self.FindTopKAssignments(in_eps, in_span, out_eps, out_span_partitions_copy, topK, invocation_graph)
            top_k_2 = self.FindTopKAssignments(in_eps, in_span, out_eps, out_span_partitions, topK, invocation_graph)
            self.AddTopKAssignments(in_span, top_k_2, all_topk_assignments, out_span_partitions_copy, out_eps, skips=True)
            span_to_top_assignments[in_span] = top_k
            top_assignments.append(top_k)
            batch_in_spans.append(in_span)
            cnt += 1

            if cnt % batch_size_mis == 0:
            # if (cnt - 1) in window_ends:
                assignments = self.GetAssignmentsMIS(top_assignments)
                assert len(assignments) == len(top_assignments) == len(batch_in_spans)
                for ind in range(len(assignments)):
                    assignment = {}
                    if len(assignments[ind]) > 0:
                        assert len(out_eps) == len(assignments[ind]) - 1
                        for ii in range(len(out_eps)):
                            assignment[out_eps[ii]] = assignments[ind][ii + 1]
                    if len(span_to_top_assignments[batch_in_spans[ind]]) < 1 or not assignment:
                        not_best_count += 1
                    else:
                        best = self.GetSpanIDNotation(out_eps, span_to_top_assignments[batch_in_spans[ind]][0][1], type1 = True)
                        chosen = self.GetSpanIDNotation(out_eps, assignment, type1 = False)
                        if best != chosen:
                            not_best_count += 1
                    self.AddAssignment(
                        batch_in_spans[ind],
                        assignment,
                        all_assignments,
                        out_span_partitions_copy,
                        out_eps,
                        delete_out_spans=True,
                        skips=True
                    )
                    cnt_unassigned += int(len(assignment) == 0)
                top_assignments = []
                batch_in_spans = []

        return all_assignments, all_topk_assignments, not_best_count, len(in_spans), self.per_span_candidates

    # Create max independent set(MIS) based on top_assignments for each incoming span
    # Each assignment consists of an ordered list of spans, starting with the incoming span and the subsequent spans are outgoing spans in order of dependence
    # For the MIS instance
    #  - add one vertex for each possible assignment
    #  - for an incoming span s, add edges between the top assignments for s (since only one of them need to be chosen)
    #  - for an assignment a1 for incoming span1 and an assignment a2 for incoming span2, add an edge between a1 and a2 if the assignments a1 and a2 intersect
    def GetAssignmentsMIS(self, top_assignments):
        mis_assignments = [[]] * len(top_assignments)
        G = self.BuildMISInstance(top_assignments)
        if len(G.nodes) != 0:
            mis = self.GetMIS(G)
            # mis = self.GetWeightedMIS(G, "weight")
            # pi = dict(zip(G.nodes(), (G.nodes[n]['weight'] for n in G.nodes)))
            # mis, w = self.exact_MWIS(G, pi, 0)
            # mis = self.exact_MWIS(G, pi, 0)
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
                # TODO: 10000 is sort of arbitrary to offset negative scores
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
            try:
                # start_time = time.time()
                mis = nx.maximal_independent_set(G) # 20000 iterations
                # mis = approximation.maximum_independent_set(G) # 350 iterations
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"Elapsed time: {elapsed_time:.6f} seconds")
            except:
                assert False
            score = sum([G.nodes[n]['weight'] for n in mis])
            # score = len(mis)
            if best_mis is None or score > best_score:
                best_mis = mis
                best_score = score
        return best_mis

    def GetWeightedMIS(self, G, weight):
        vcover = approximation.min_weighted_vertex_cover(G, weight=weight)
        return set(G.nodes()).difference(set(vcover))

    def exact_MWIS(self, graph, pi, b_score=0):
        ''' compute maximum weighted independent set (recursively) using python
        networkx package. Input items are:
        - graph, a networkx graph
        - pi, a dictionary of dual values attached to node (primal constraints)
        - b_score, a bestscore (if non 0, it pruned some final branches)
        It returns:
        - mwis_set, a MWIS as a sorted tuple of nodes
        - mwis_weight, the sum over n in mwis_set of pi[n]'''
        global best_score
        # assert sum(pi) > 0
        graph_copy = graph.copy()
        # mwis weight is stored as a 'score' graph attribute
        graph_copy.graph['score'] = 0
        best_score = b_score

        def get_mwis(G):
            '''
            Based on "A column generation approach for graph coloring" from
            Mehrotra and Trick, 1995'''
            global best_score
            # score stores the best score along the path explored so far
            key = tuple(sorted(G.nodes()))
            ub = sum(pi[n] for n in G.nodes())
            score = G.graph['score']
            # if graph is composed of singletons, leave now
            if G.number_of_edges == 0:
                if score + ub > best_score + EPS:
                    best_score = score + ub
                return key, ub
            # compute highest priority node (used in recursion to choose {i})
            node_iter = ((n, deg*pi[n]) for (n, deg) in G.degree())
            node_chosen, _ = max(node_iter, key=lambda x: x[1])
            pi_chosen = pi[node_chosen]
            node_chosen_neighbors = list(G[node_chosen])
            pi_neighbors = sum(pi[n] for n in node_chosen_neighbors)
            G.remove_node(node_chosen)
            # Gh = G - {node_chosen} union {anti-neighbors{node-chosen}}
            # For Gh, ub decreases by value of pi over neighbors of {node_chosen}
            # and value of pi over {node_chosen} as node_chosen is disconnected
            # For Gh, score increases by value of pi over {node_chosen}
            Gh = G.copy()
            Gh.remove_nodes_from(node_chosen_neighbors)
            mwis_set_h, mwis_weight_h = tuple(), 0
            if Gh:
                ubh = ub - pi_neighbors - pi_chosen
                if score + pi_chosen + ubh > best_score + EPS:
                    Gh.graph['score'] += pi_chosen
                    mwis_set_h, mwis_weight_h = get_mwis(Gh)
                del Gh
            mwis_set_h += (node_chosen, )
            mwis_weight_h += pi_chosen
            # Gp = G - {node_chosen}
            # For Gp, ub decreases by value of pi over {node_chosen}
            # For Gh, score does not increase
            mwis_set_p, mwis_weight_p = tuple(), 0
            if G:
                ubp = ub - pi_chosen
                if score + ubp > best_score + EPS:
                    mwis_set_p, mwis_weight_p = get_mwis(G)
                del G
            # select case with maximum score
            if mwis_set_p and mwis_weight_p > mwis_weight_h + EPS:
                mwis_set, mwis_weight = mwis_set_p, mwis_weight_p
            else:
                mwis_set, mwis_weight = mwis_set_h, mwis_weight_h
            # increase score
            score += mwis_weight
            if score > best_score + EPS:
                best_score = score
            # return set and weight
            key = tuple(sorted(mwis_set))
            return key, mwis_weight

        best_mis = None
        best_score_1 = -math.inf

        for i in range(1000):
            try:
                mis, w = get_mwis(copy.deepcopy(graph_copy))
            except:
                assert False
            score = sum([graph_copy.nodes[n]['weight'] for n in mis])
            # score = len(mis)
            if best_mis is None or score > best_score_1:
                best_mis = mis
                best_score_1 = score

        return best_mis
        # return get_mwis(graph_copy)