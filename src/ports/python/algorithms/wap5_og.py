import math
import statistics
import scipy.stats
import copy

VERBOSE = False
ERROR_WINDOW = False #True

already_picked = {}

class WAP5_OG(object):
    def __init__(self, all_spans, all_processes):
        self.all_spans = all_spans
        self.all_processes = all_processes
        self.services_times = {}
        self.samples = {}
        self.parallel = True
        self.distribution_values = {}
        self.large_delay = None
        self.magic_delay = 4
        self.all_assignments = {}

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

        def FindPreviousDelay(out_span, in_spans):
            for x, in_span in enumerate(in_spans):
                if float(in_span) > float(out_span):
                    return float(in_spans[x - 1])
            return -1

        def ComputeDistParams(ep1, ep2, t1, t2):
            t1 = t1[in_span_start:in_span_end]
            t2 = t2[in_span_start:in_span_end]
            assert len(t1) == len(t2)
            new_samples = []
            for i in range((in_span_end - in_span_start)):
                x = FindPreviousDelay(t2[i], t1)
                if x != -1:
                    new_samples.append(float(t2[i]) - x)
            # mean = (sum(t2) - sum(t1)) / len(t1)
            if (ep1, ep2) not in self.samples:
                self.samples[(ep1, ep2)] = []
            self.samples[(ep1, ep2)].extend(new_samples)
            mean = statistics.mean(self.samples[(ep1, ep2)])

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

    def GetExponentialPDF(self, t, mean, std):
        if mean < 1.0e-10 or std < 1.0e-10:
            return 1
        scale = mean
        p = scipy.stats.expon.logpdf(t, scale=scale)
        return p

    def GetLogNormalPDF(self, t, mean, std):
        if mean < 1.0e-10 or std < 1.0e-10:
            return 1
        scale = mean
        # s = standard deviation of log x
        s = (1 + math.sqrt(1 + 4 * ((mean/std)**2)))/2
        s = math.sqrt(math.log(s))
        p = scipy.stats.lognorm.logpdf(t, s=s, scale=scale)
        return p

    def GetParetoPDF(self, t, mean, std):
        assert (std > 0.0)
        alpha = 1 + math.sqrt(1 + (mean / (std**2)))
        scale = (mean * (alpha - 1)) / alpha
        loc = -1
        p = scipy.stats.pareto.logcdf(t, alpha, loc=loc, scale=scale)
        return p


    def GetEpPairCost(self, ep1, ep2, t1, t2):
        mean, std = self.services_times[(ep1, ep2)]
        # print("mean, std:", mean, std)
        if std < 1.0e-12:
            std = 0.001
        # !TODO uncomment next line
        # p = scipy.stats.norm.logpdf(t2 - t1, loc=mean, scale=std)
        # p = self.GetParetoPDF(t2 - t1, mean, std)
        # p = self.GetLogNormalPDF(t2 - t1, mean, std)
        p = self.GetExponentialPDF(t2 - t1, mean, std)
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
        for i in range(len(assignment) - 1):
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
            if assignment[i][1] in ["eg9d6u4jwelh6he6acx2fai6385q8br0", "3m6q1llsb1w49543sallcg07eyllj05n"]:
                mean, std = self.services_times[(curr_ep, next_ep)]
                print("Service times", curr_ep, next_ep, mean, std)
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
                if in_span[1] in ["eg9d6u4jwelh6he6acx2fai6385q8br0", "3m6q1llsb1w49543sallcg07eyllj05n"]:
                    print("scores: ", [ii[1] for ii in stack], [ii[2] for ii in stack], score)
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

    def BuildDistributions(self, incoming_spans, outgoing_spans_per_ep, out_ep):
        spans = incoming_spans + outgoing_spans_per_ep
        spans.sort(key=lambda x: x.start_mus)

        for i, span in enumerate(spans):
            if span.span_kind == "client":
                # print(i, span)
                sent_mus = span.start_mus
                parent_span = None
                for j, preceding_span in reversed(list(enumerate(spans[:i]))):
                    if sent_mus - preceding_span.start_mus > self.large_delay:
                        break
                    if preceding_span.span_kind == "server":
                        parent_span = preceding_span
                        break
                if parent_span is not None:
                    if out_ep not in self.distribution_values:
                        self.distribution_values[out_ep] = []
                    self.distribution_values[out_ep].append(sent_mus - parent_span.start_mus)

    def CalculateProbability(self, t, mean):
        scale = mean
        p = scipy.stats.expon.logpdf(t, scale=scale)
        return p

    def ScoreParents(self, incoming_spans, outgoing_spans_per_ep, out_ep):
        spans = incoming_spans + outgoing_spans_per_ep
        spans.sort(key=lambda x: x.start_mus)

        for span in spans:
            already_picked[span.GetId()] = False

        for i, span in enumerate(spans):
            if span.span_kind == "client":
                # print(i, span)
                sent_mus = span.start_mus
                candidate_parent_spans = []
                for j, preceding_span in reversed(list(enumerate(spans[:i]))):
                    if sent_mus - preceding_span.start_mus > self.magic_delay * statistics.mean(self.distribution_values[out_ep]):
                        p = self.CalculateProbability(self.magic_delay * statistics.mean(self.distribution_values[out_ep]), statistics.mean(self.distribution_values[out_ep]))
                        candidate_parent_spans.append(("Spontaneous", p))
                        break
                    if preceding_span.span_kind == "server":
                        if not already_picked[preceding_span.GetId()]:
                            p = self.CalculateProbability(sent_mus - preceding_span.start_mus, statistics.mean(self.distribution_values[out_ep]))
                            candidate_parent_spans.append((preceding_span, p))
                            already_picked[preceding_span.GetId()] = True

                candidate_parent_spans.sort(key=lambda x: x[1])
                if len(candidate_parent_spans) != 0:
                    if candidate_parent_spans[-1][0] != "Spontaneous":
                        parent = candidate_parent_spans[-1][0]

                        if out_ep not in self.all_assignments:
                            self.all_assignments[out_ep] = {}
                        if parent.GetId() not in self.all_assignments[out_ep]:
                            self.all_assignments[out_ep][parent.GetId()] = []

                        self.all_assignments[out_ep][parent.GetId()].append(span.GetId())

    def CombineSpans(self, span_partitions):
        spans = []
        for ep in span_partitions.keys():
            spans.extend(span_partitions[ep])
        return spans

    def FindAssignments(self, process, in_span_partitions, out_span_partitions, parallel, instrumented_hops, true_assignments):
        # Todo: consider responses to outgoing requests in incoming_spans
        incoming_spans = self.CombineSpans(in_span_partitions)
        outcoming_spans = self.CombineSpans(out_span_partitions)

        self.large_delay = max([i.duration_mus for i in incoming_spans])

        for out_ep in out_span_partitions.keys():
            self.BuildDistributions(incoming_spans, out_span_partitions[out_ep], out_ep)
            self.ScoreParents(incoming_spans, out_span_partitions[out_ep], out_ep)

        for out_ep in out_span_partitions.keys():
            for in_span in incoming_spans:
                if in_span.GetId() not in self.all_assignments[out_ep]:
                    self.all_assignments[out_ep][in_span.GetId()] = [("NA", "NA")]

        return self.all_assignments
