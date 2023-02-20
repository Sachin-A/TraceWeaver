import math
import scipy.stats
import copy

VERBOSE = False
ERROR_WINDOW = True

class Timing(object):
    def __init__(self):
        self.services_times = {}

    # verify that all outgoing request dependencies are serial
    def VerifySerialDependency(self, in_spans, out_eps, out_span_partitions):
        def FindSpanTraceId(trace_id, spans):
            for s in spans:
                if s[1] == trace_id:
                    return s
            return None

        for s in in_spans:
            trace_id = s[1]
            prev_time = float(s[2])
            for ep in out_eps:
                out_span = FindSpanTraceId(trace_id, out_span_partitions[ep])
                assert float(out_span[2]) > prev_time
                prev_time = float(out_span[2]) + float(out_span[8])

    def GetOutEpsInOrder(self, out_span_partitions):
        eps = []
        for ep, spans in out_span_partitions.items():
            assert len(spans) > 0
            eps.append((ep, float(spans[0][2])))
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
            if ERROR_WINDOW:
                mean = max(0, (sum(t2) - sum(t1)) / len(t1))
            else:
                mean = (sum(t2) - sum(t1)) / len(t1)
            batch_means = []
            nbatches = 10
            batch_size = math.ceil(float(len(t1)) / nbatches)
            for i in range(nbatches):
                start = i * batch_size
                end = min(len(t1), (i + 1) * batch_size)
                if end - start > 0:
                    if ERROR_WINDOW:
                        batch_means.append(
                            max(0, (sum(t2[start:end]) - sum(t1[start:end])) / (end - start))
                        )
                    else:
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

        # between incoming -- first outgoing
        ep1 = list(in_span_partitions.keys())[0]
        ep2 = out_eps[0]
        t1 = sorted([float(s[2]) for s in in_span_partitions[ep1]])
        t2 = sorted([float(s[2]) for s in out_span_partitions[ep2]])
        ComputeDistParams(ep1, ep2, t1, t2)

        # between outgoing -- outgoing
        for i in range(len(out_eps) - 1):
            ep1 = out_eps[i]
            ep2 = out_eps[i + 1]
            t1 = sorted(
                [float(s[2]) + float(s[8]) for s in out_span_partitions[ep1]]
            )
            t2 = sorted([float(s[2]) for s in out_span_partitions[ep2]])
            ComputeDistParams(ep1, ep2, t1, t2)

        # between last outgoing -- incoming
        ep1 = out_eps[-1]
        ep2 = list(in_span_partitions.keys())[0]
        t1 = sorted([float(s[2]) + float(s[8]) for s in out_span_partitions[ep1]])
        t2 = sorted([float(s[2]) + float(s[8]) for s in in_span_partitions[ep2]])
        ComputeDistParams(ep1, ep2, t1, t2)

    def GetEpPairCost(self, ep1, ep2, t1, t2):
        mean, std = self.services_times[(ep1, ep2)]
        # print("mean, std:", mean, std)
        if std == 0:
            std = 0.001
        p = scipy.stats.norm.logpdf(t2 - t1, loc=mean, scale=std)
        # print(t2, t1)
        # print("x:", t2-t1)
        # print("p:", p)
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

    def ScoreAssignment(self, assignment):
        cost = 0
        for i in range(len(assignment)):
            curr_ep = (
                assignment[i][4]
                if i == 0
                else assignment[i][6]
            )
            curr_time = (
                float(assignment[i][2])
                if i == 0
                else float(assignment[i][2]) + float(assignment[i][8])
            )

            next_i = (i + 1) % len(assignment)
            next_ep = (
                assignment[next_i][4]
                if next_i == 0
                else assignment[next_i][6]
            )
            next_time = (
                float(assignment[next_i][2]) + float(assignment[next_i][8])
                if next_i == 0
                else float(assignment[next_i][2])
            )
            if VERBOSE:
                print("Computing cost between", curr_ep, next_ep)
            cost += self.GetEpPairCost(curr_ep, next_ep, curr_time, next_time)
        return cost

    def FindMinCostAssignment(self, in_span, out_eps, out_span_partitions):
        global best_assignment
        global best_score
        best_assignment = None
        if ERROR_WINDOW:
            error_window = 5
            best_score = -1000000000.0
        else:
            error_window = 0
            best_score = -1000000.0

        def DfsTraverse(stack):
            global best_assignment
            global best_score
            i = len(stack)
            if VERBOSE:
                print("DFSTraverse", i, out_eps, stack)
            last_span = stack[-1]
            if i == len(out_span_partitions) + 1:
                score = self.ScoreAssignment(stack)
                # print([i[2] for i in stack])
                # print(score)
                # input()
                if best_score < score:
                    best_assignment = stack
                    best_score = score
                    # print("Picked:", [i[2] for i in stack])
                    # print(score)
                    # input()
            else:
                #!TODO: filter out branches that have high cost
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
        # return a dictionary of {ep: span}
        ret = {}
        if best_assignment is not None:
            assert len(out_eps) == len(best_assignment) - 1
            ret = {out_eps[i]: best_assignment[i + 1] for i in range(len(out_eps))}
        return ret

    def AddAssignment(
        self,
        in_span,
        assignment,
        all_assignments,
        out_span_partitions,
        out_eps,
        delete_out_spans=False
    ):
        # add assignment to all_assignments
        for ep in out_eps:
            if ep not in all_assignments:
                all_assignments[ep] = {}
            out_span = assignment.get(ep, None)
            out_span_id = [out_span[1], out_span[3]] if out_span is not None else ("NA", "NA")
            all_assignments[ep][(in_span[1], in_span[3])] = out_span_id

        if delete_out_spans:
            # remove spans of this assignment so they can't be assigned again
            #!TODO: this implementation is not efficient
            for ep, span in assignment.items():
                # print(ep, in_span, span)
                out_span_partitions[ep].remove(span)

    def FindAssignments(self, process, in_span_partitions, out_span_partitions):
        assert len(in_span_partitions) == 1
        in_eps, in_spans = list(in_span_partitions.items())[0]
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
