import math
import scipy.stats
import copy
import networkx as nx

VERBOSE = False

class Timing(object):
    def __init__(self, all_spans, all_processes):
        self.all_spans = all_spans
        self.all_processes = all_processes
        self.services_times = {}
        self.parallel = False
        self.instrumented_hops = []
        self.true_assignments = None
        self.normal = True

    # verify that all outgoing request dependencies are serial
    def VerifySerialDependency(self, in_spans, out_eps, out_span_partitions):
        def FindSpanTraceId(trace_id, spans):
            for s in spans:
                if s.trace_id == trace_id:
                    return s
            return None

        for s in in_spans:
            trace_id = s.trace_id
            prev_time = s.start_mus
            for ep in out_eps:
                out_span = FindSpanTraceId(trace_id, out_span_partitions[ep])
                assert out_span.start_mus > prev_time
                prev_time = out_span.start_mus + out_span.duration_mus

    def GetOutEpsInOrder(self, out_span_partitions):
        eps = []
        for ep, spans in out_span_partitions.items():
            assert len(spans) > 0
            eps.append((ep, spans[0].start_mus))
        eps.sort(key=lambda x: x[1])
        return [x[0] for x in eps]

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

    def GetExponentialPDF(self, t, mean, std):
        if mean < 1.0e-10 or std < 1.0e-10:
            return 1
        scale = mean
        p = scipy.stats.expon.logpdf(t, scale=scale)
        return p

    def GetEpPairCost(self, ep1, ep2, t1, t2, normalized = False):
        mean, std = self.services_times[(ep1, ep2)]
        if std < 1.0e-12:
            std = 0.001
        if self.normal:
            if not normalized:
                p = scipy.stats.norm.logpdf(t2 - t1, loc=mean, scale=std)
            else:
                p = scipy.stats.norm.pdf(t2 - t1, loc=mean, scale=std)
        else:
            p = scipy.stats.expon.logpdf(t2 - t1, scale=mean)
        return p
        """
        # CDF
        x = scipy.stats.norm.cdf(t2 - t1, loc=mean, scale=std)
        cp = 2 * min(x, 1-x)
        if cp==0:
            return -math.inf
        else:
            return math.log(cp)
        """

    def AllSkip(self, assignment):
        for i in assignment[1:]:
            if i.trace_id != "None":
                return False
        return True

    def AllSkip2(self, assignment):
        for i in assignment[1:]:
            if i[1].trace_id != "None":
                return False
        return True

    def ScoreAssignmentWithSkip(self, assignment, normalized = False):
        cost = 0
        num_mappings = 0

        if self.AllSkip(assignment):
            # print(assignment)
            # print(0)
            # input()
            return 0

        for i in range(len(assignment) + 1):

            if i == len(assignment):
                curr_ep = assignment[0].GetParentProcess(self.all_processes, self.all_spans)
                curr_time = assignment[0].start_mus + assignment[0].duration_mus
                cost += self.GetEpPairCost(prev_ep, curr_ep, prev_time, curr_time, normalized)

            else:
                if assignment[i].trace_id != "None":
                    num_mappings += 1
                    if i != 0:
                        curr_ep = assignment[i].GetChildProcess(self.all_processes, self.all_spans)
                        curr_time = assignment[i].start_mus
                        cost += self.GetEpPairCost(prev_ep, curr_ep, prev_time, curr_time, normalized)

                    prev_ep = (
                        assignment[i].GetParentProcess(self.all_processes, self.all_spans)
                        if i == 0
                        else assignment[i].GetChildProcess(self.all_processes, self.all_spans)
                    )
                    prev_time = (
                        assignment[i].start_mus
                        if i == 0
                        else assignment[i].start_mus + assignment[i].duration_mus
                    )

        # print(assignment)
        # print(cost)
        # input()
        # return cost / len(assignment)
        return cost / (num_mappings)

    def ScoreAssignmentSequential(self, assignment, normalized = False):
        cost = 0
        for i in range(len(assignment)):
            curr_ep = (
                assignment[i].GetParentProcess(self.all_processes, self.all_spans)
                if i == 0
                else assignment[i].GetChildProcess(self.all_processes, self.all_spans)
            )
            curr_time = (
                assignment[i].start_mus
                if i == 0
                else assignment[i].start_mus + assignment[i].duration_mus
            )

            next_i = (i + 1) % len(assignment)
            next_ep = (
                assignment[next_i].GetParentProcess(self.all_processes, self.all_spans)
                if next_i == 0
                else assignment[next_i].GetChildProcess(self.all_processes, self.all_spans)
            )
            next_time = (
                assignment[next_i].start_mus + assignment[next_i].duration_mus
                if next_i == 0
                else assignment[next_i].start_mus
            )
            if VERBOSE:
                print("Computing cost between", curr_ep, next_ep)
            cost += self.GetEpPairCost(curr_ep, next_ep, curr_time, next_time, normalized)

        if normalized:
            return cost / len(assignment)
        return cost

    def ScoreAssignmentParallel(self, assignment, normalized = False):
        cost = 0
        for i in range(1, len(assignment)):
            curr_ep = assignment[0].GetParentProcess(self.all_processes, self.all_spans)
            curr_time = float(assignment[0].start_mus)

            next_ep = assignment[i].GetChildProcess(self.all_processes, self.all_spans)
            next_time = float(assignment[i].start_mus)
            if VERBOSE:
                print("Computing cost between", curr_ep, next_ep)
            cost += self.GetEpPairCost(curr_ep, next_ep, curr_time, next_time, normalized)

        # latest_ep = ""
        # latest_time = -1
        # curr_ep = assignment[0][4]
        # curr_time = float(assignment[0][2])

        # for i in range(1, len(assignment)):
        #     if float(assignment[i][2]) + float(assignment[i][8]) > latest_time:
        #         latest_ep = assignment[i][6]
        #         latest_time = float(assignment[i][2]) + float(assignment[i][8])

        # if latest_ep != "" and latest_time != -1:
        #     cost += self.GetEpPairCost(curr_ep, latest_ep, curr_time, latest_time)

        if normalized:
            return cost / len(assignment)
        return cost

    def AlsoNonPrimaryAncestor(self, before_ep, current_ep, invocation_graph):
        all_paths = list(nx.all_simple_paths(invocation_graph, source=before_ep, target=current_ep, cutoff=2))
        if not all_paths:
            assert False
        else:
            for i, path in enumerate(all_paths):
                path_length = len(path) - 1
                if path_length > 1:
                    return True
        return False

    def ScoreAssignmentAsPerInvocationGraph(self, assignment, invocation_graph, out_eps, normalized = False):

        # print("New")

        if self.AllSkip2(assignment):
            return 0

        def FindValidAncestor(ep):

            before_eps = invocation_graph.in_edges(ep)
            if len(before_eps) == 0:
                return None

            valid_spans = []
            invalid_spans = []
            for (before_ep, self_ep) in before_eps:

                ep_index = out_eps.index(before_ep)
                b_ep = assignment[ep_index + 1][0]
                b_span = assignment[ep_index + 1][1]
                assert b_ep == before_ep

                if b_span.trace_id != "None":
                    valid_spans.append((b_ep, b_span))
                else:
                    invalid_spans.append((b_ep, b_span))

            if len(valid_spans) > 0:
                return valid_spans
                # return max(valid_spans, key = lambda x: x[1].start_mus + x[1].duration_mus)
            else:
                next_layer_spans = []
                for (ep, span) in invalid_spans:
                    x = FindValidAncestor(ep)
                    if x != None:
                        next_layer_spans.append(x)
                return next_layer_spans

        def AlsoNonPrimaryAncestor(before_ep, current_ep):
            all_paths = list(nx.all_simple_paths(invocation_graph, source=before_ep, target=current_ep, cutoff=2))
            if not all_paths:
                assert False
            else:
                for i, path in enumerate(all_paths):
                    path_length = len(path) - 1
                    if path_length > 1:
                        return True
            return False

        cost = 0
        num_mappings = 0
        first_ep, first_span = assignment[0]

        assignment_without_skips = []
        for a in assignment:
            if a[1].trace_id != "None":
                assignment_without_skips.append(a)

        last_ep, last_span = max(assignment_without_skips[1:], key = lambda x: x[1].start_mus + x[1].duration_mus)

        for (current_ep, current_span) in assignment[1:]:
            before_eps = invocation_graph.in_edges(current_ep)

            if current_span.trace_id == "None":
                continue

            for (before_ep, self_ep) in before_eps:

                ep_index = out_eps.index(before_ep)
                b_ep = assignment[ep_index + 1][0]
                b_span = assignment[ep_index + 1][1]
                assert b_ep == before_ep

                if not AlsoNonPrimaryAncestor(before_ep, current_ep):

                    if b_span.trace_id == "None":
                        valid_spans = FindValidAncestor(b_ep)
                        if valid_spans == None:
                            cost += self.GetEpPairCost(first_ep, current_ep, first_span.start_mus, current_span.start_mus, normalized)
                            num_mappings += 1
                            # print(first_ep, current_ep)
                            # print(num_mappings)
                            # input()
                        else:
                            latest = max(valid_spans, key = lambda x: x[1].start_mus + x[1].duration_mus)
                            cost += self.GetEpPairCost(latest[0], current_ep, latest[1].start_mus, current_span.start_mus, normalized)
                            num_mappings += 1
                            # print(latest[0], current_ep)
                            # print(num_mappings)
                            # input()

                        continue

                    # print(before_ep, current_ep)
                    cost += self.GetEpPairCost(before_ep, current_ep, b_span.start_mus + b_span.duration_mus, current_span.start_mus, normalized)
                    num_mappings += 1
                    # print(num_mappings)

            if len(invocation_graph.in_edges(current_ep)) == 0:
                # print(first_ep, current_ep)
                cost += self.GetEpPairCost(first_ep, current_ep, first_span.start_mus, current_span.start_mus, normalized)
                num_mappings += 1
                # print(num_mappings)

            if current_ep == last_ep:
                # print(current_ep, first_ep)
                cost += self.GetEpPairCost(current_ep, first_ep, current_span.start_mus + current_span.duration_mus, first_span.start_mus + first_span.duration_mus, normalized)
                num_mappings += 1
                # print(num_mappings)

        # if len(assignment_without_skips) < len(assignment):
        #     print(num_mappings)
        #     input()

        # print(num_mappings, normalized)
        if normalized:
            return cost / num_mappings
        return cost

    def FindMinCostAssignment(self, in_span, out_eps, out_span_partitions):
        global best_assignment
        global best_score
        best_assignment = None
        best_score = -1000000.0
        score_list = []

        def DfsTraverse(stack):
            global best_assignment
            global best_score
            i = len(stack)
            if VERBOSE:
                print("DFSTraverse", i, out_eps, stack)
            last_span = stack[-1]
            if i == len(out_span_partitions) + 1:
                if self.parallel:
                    score = self.ScoreAssignmentParallel(stack)
                else:
                    score = self.ScoreAssignmentSequential(stack)
                score_list.append(score)
                if best_score < score:
                    best_assignment = stack
                    best_score = score
            elif i in self.instrumented_hops:
                ep = out_eps[i - 1]
                span_id = self.true_assignments[ep][in_span.GetId()]
                for s in out_span_partitions[ep]:
                    if s.GetId() == span_id:
                        DfsTraverse(stack + [s])
                        break
            else:
                #!TODO: filter out branches that have high cost
                ep = out_eps[i - 1]
                for s in out_span_partitions[ep]:
                    # parallel eps
                    if self.parallel:
                        if (
                            in_span.start_mus < s.start_mus
                            and s.start_mus + s.duration_mus
                            < in_span.start_mus + in_span.duration_mus
                        ):
                            DfsTraverse(stack + [s])

                    # Sequential eps
                    else:
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
        # return a dictionary of {ep: span}
        ret = {}
        if best_assignment is not None:
            assert len(out_eps) == len(best_assignment) - 1
            ret = {out_eps[i]: best_assignment[i + 1] for i in range(len(out_eps))}

        # print(score_list)
        # print(scipy.stats.zscore(score_list))
        # input()
        return ret

    def AddAssignment(
        self,
        in_span,
        assignment,
        all_assignments,
        out_span_partitions,
        out_eps,
        delete_out_spans=False,
        skips=False
    ):
        # add assignment to all_assignments
        for ep in out_eps:
            if ep not in all_assignments:
                all_assignments[ep] = {}
            out_span = assignment.get(ep, None)
            if skips:
                if out_span is None:
                    out_span_id = ("NA", "NA")
                else:
                    out_span_id = out_span.GetId() if out_span.trace_id != "None" else ('Skip', 'Skip')
            else:
                out_span_id = out_span.GetId() if out_span is not None else ("NA", "NA")
            all_assignments[ep][in_span.GetId()] = out_span_id

        if delete_out_spans:
            # remove spans of this assignment so they can't be assigned again
            #!TODO: this implementation is not efficient
            for ep, span in assignment.items():
                if span.trace_id != "None":
                    # print(ep, in_span, span)
                    out_span_partitions[ep].remove(span)

    def AddTopKAssignments(
        self,
        in_span,
        topk_assignments,
        all_topk_assignments,
        out_span_partitions,
        out_eps,
        skips=False
    ):
        for i, ep in enumerate(out_eps):
            if ep not in all_topk_assignments:
                all_topk_assignments[ep] = {}
            all_topk_assignments[ep][in_span.GetId()] = []
            for assignment in topk_assignments:
                assignment = assignment[1]
                out_span = assignment[i + 1]
                if skips:
                    if out_span is None:
                        out_span_id = ("NA", "NA")
                    else:
                        out_span_id = out_span.GetId() if out_span.trace_id != "None" else ('Skip', 'Skip')
                else:
                    out_span_id = out_span.GetId() if out_span is not None else ("NA", "NA")
                all_topk_assignments[ep][in_span.GetId()].append(out_span_id)

    def FindAssignments(self, process, in_span_partitions, out_span_partitions, parallel, instrumented_hops, true_assignments):
        assert len(in_span_partitions) == 1
        self.parallel = parallel
        self.instrumented_hops = instrumented_hops
        self.true_assignments = true_assignments
        _, in_spans = list(in_span_partitions.items())[0]
        out_eps = self.GetOutEpsInOrder(out_span_partitions)
        out_span_partitions_copy = copy.deepcopy(out_span_partitions)
        all_assignments = {}
        cnt = 0
        cnt_unassigned = 0
        batch_size = 100
        for in_span in in_spans:
            if cnt % batch_size == 0:
                self.ComputeEpPairDistParams(
                    in_span_partitions,
                    out_span_partitions,
                    out_eps,
                    in_span_start=cnt,
                    in_span_end=min(len(in_spans), cnt + batch_size),
                )
            # find the min-cost assignment for in_span
            min_cost_assignment = self.FindMinCostAssignment(
                in_span, out_eps, out_span_partitions_copy
            )
            self.AddAssignment(
                in_span,
                min_cost_assignment,
                all_assignments,
                out_span_partitions_copy,
                out_eps,
                delete_out_spans=True
            )
            cnt += 1
            cnt_unassigned += len(min_cost_assignment) == 0
            #!TODO: update mean, std of service times using EWMA
        print("Finished %d spans, unassigned spans: %d" % (cnt, cnt_unassigned))
        return all_assignments
