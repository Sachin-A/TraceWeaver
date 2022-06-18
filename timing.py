import math
from scipy.stats import norm


class Timing(object):
    def __init__(self, all_spans, all_processes):
        self.all_spans = all_spans
        self.all_processes = all_processes
        self.services_times = {}

    # verify that all outgoing request dependencies are serial
    def VerifySerialDependency(self, outgoing_span_partitions):
        #!TODO
        pass

    def GetOutgoingSpanOrder(self, outgoing_span_partitions):
        # ep: endpoint
        eps = []
        for ep, spans in outgoing_span_partitions.items():
            assert len(spans) > 0
            eps.append((ep, spans[0].start_mus))
        eps.sort(key=lambda x: x[1])
        return [x[0] for x in eps]

    def PopulateEpPairDistributions(
        incoming_span_partitions, outgoing_span_partitions, outgoing_eps
    ):
        for i in range(len(outgoing_eps) - 1):
            ep1 = outgoing_eps[i]
            ep2 = outgoing_eps[i + 1]
            t1 = sorted(
                [s.start_mus + s.duration_mus for s in outgoing_span_partitions[ep1]]
            )
            t2 = sorted([s.start_mus for s in outgoing_span_partitions[ep2]])
            assert len(t1) == len(t2)
            mean = (sum(t2) - sum(t1)) / len(t1)
            batch_means = []
            nbatches = 10
            for i in range(nbatches):
                batch_size = int((len(t1) + nbatches - 1) / nbatches)
                start = min(len(t1), i * batch_size)
                end = min(len(t1), (i + 1) * batch_size)
                if end - start > 0:
                    batch_mean = (sum(t2[start:end]) - sum(t1[start:end])) / (
                        end - start
                    )
                    batch_means.append(batch_mean)
            std = scipy.stats.std(batch_means)
            self.services_times = mean, std

    def GetEpPairCost(ep1, ep2, t1, t2):
        mean, std = self.services_times[(ep1, ep2)]
        p = scipy.stats.norm(mean, std).pdf(t2 - t1)
        return math.log(p)

    def ScoreAssignment(self, stack):
        cost = 0
        for i in range(len(stack)):
            curr_ep = (
                stack[i].GetParentProcess() if i == 0 else stack[i].GetChildProcess()
            )
            curr_time = (
                stack[i].start_mus
                if i == 0
                else stack[i].start_mus + stack[i].duration_mus
            )

            next_i = (i + 1) % len(stack)
            next_ep = (
                stack[next_i].GetParentProcess()
                if next_i == 0
                else stack[next_i].GetChildProcess()
            )
            next_time = (
                stack[next_i].start_mus + stack[next_i].duration_mus
                if next_i == 0
                else stack[next_i].start_mus
            )
            cost += GetEpPairCost(curr_ep, next_ep, curr_time, next_time)
        return cost

    def FindMinCostAssignment(
        self, incoming_span, outgoing_eps, outgoing_span_partitions
    ):
        best_assignment = None
        best_score = -math.inf

        def DfsTraverse(stack):
            i = len(stack)
            last_span = stack[-1]
            ep = outgoing_eps[i - 1]
            if i == len(outgoing_span_partitions) + 1:
                score = ScoreAssignment(stack)
                if best_score < score:
                    best_assignment = stack
                    best_score = score
            else:
                for s in outgoing_span_partitions[ep]:
                    if i == 1:
                        # first ep
                        if incoming_span.start_mus < s.start_mus:
                            DfsTraverse(stack + [s])
                    elif i < len(outgoing_eps):
                        # intermediate ep
                        if last_span.start_mus + last_span.duration_mus < s.start_mus:
                            DfsTraverse(stack + [s])
                    elif i == len(outgoing_eps):
                        # last ep
                        if (
                            last_span.start_mus + last_span.duration_mus < s.start_mus
                            and s.start_mus + s.duration_mus
                            < incoming_span.start_mus + incoming.duration_mus
                        ):
                            DfsTraverse(stack + [s])

        DfsTraverse([incoming_span])
        # return a dictionary of {ep: trace_id}
        ret = {}
        for ep, span in zip(outgoing_eps):
            ret[ep] = span
        return ret

    def AssignSpans(
        self, incoming_span, assignment, assignments_dict, outgoing_span_partitions
    ):
        # update assignments dict
        for ep, trace_id in min_cost_assignments:
            if ep not in assignments_dict:
                assignments_dict[ep] = []
            assignments_dict[ep].append(trace_id)

        #!TODO: remove assignment spans so that they can't be assigned again

    def PredictTraceIdSequences(
        self, process, incoming_span_partitions, outgoing_span_partitions
    ):
        assert len(incoming_span_partitions) == 1
        ep, incoming_spans = list(incoming_span_partitions.items())[0]
        outgoing_eps = self.GetOutgoingSpanOrder(outgoing_span_partitions)
        ret = {}
        for span in incoming_spans:
            # find the minimimum cost label assignment for span
            min_cost_assignments = FindMinCostAssignment(
                span, outgoing_eps, outgoing_span_partitions
            )
            AssignSpans(
                incoming_span, assignment, assignments_dict, outgoing_span_partitions
            )
            #!TODO: update mean, std of service times using EWMA
        return ret
