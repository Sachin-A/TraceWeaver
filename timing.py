import math
import scipy.stats
import copy

VERBOSE = False

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
        self, incoming_span_partitions, outgoing_span_partitions, outgoing_eps
    ):
        def ComputeDistParams(ep1, ep2, t1, t2):
            assert len(t1) == len(t2)
            mean = (sum(t2) - sum(t1)) / len(t1)
            batch_means = []
            nbatches = 25
            for i in range(nbatches):
                batch_size = int((len(t1) + nbatches - 1) / nbatches)
                start = min(len(t1), i * batch_size)
                end = min(len(t1), (i + 1) * batch_size)
                if end - start > 0:
                    batch_mean = (sum(t2[start:end]) - sum(t1[start:end])) / (
                        end - start
                    )
                    batch_means.append(batch_mean)
            std = math.sqrt(len(batch_means)) * scipy.stats.tstd(batch_means)
            #std = 1
            print("Assigning ep pair (%s, %s), distribution params: %f, %f" % (ep1, ep2, mean, std))
            self.services_times[(ep1, ep2)] = mean, std

        # between incoming -- first outgoing
        ep1 = list(incoming_span_partitions.keys())[0]
        ep2 = outgoing_eps[0]
        t1 = sorted([s.start_mus for s in incoming_span_partitions[ep1]])
        t2 = sorted([s.start_mus for s in outgoing_span_partitions[ep2]])
        ComputeDistParams(ep1, ep2, t1, t2)

        # between outgoing -- outgoing
        for i in range(len(outgoing_eps) - 1):
            ep1 = outgoing_eps[i]
            ep2 = outgoing_eps[i + 1]
            t1 = sorted(
                [s.start_mus + s.duration_mus for s in outgoing_span_partitions[ep1]]
            )
            t2 = sorted([s.start_mus for s in outgoing_span_partitions[ep2]])
            ComputeDistParams(ep1, ep2, t1, t2)

        # between last outgoing -- incoming
        ep1 = outgoing_eps[-1]
        ep2 = list(incoming_span_partitions.keys())[0]
        t1 = sorted([s.start_mus + s.duration_mus for s in outgoing_span_partitions[ep1]])
        t2 = sorted([s.start_mus + s.duration_mus for s in incoming_span_partitions[ep2]])
        ComputeDistParams(ep1, ep2, t1, t2)

    def GetEpPairCost(self, ep1, ep2, t1, t2):
        mean, std = self.services_times[(ep1, ep2)]
        p = scipy.stats.norm.logpdf(t2 - t1, loc=mean, scale=std)
        return p
        '''
        if p==0:
            return -math.inf
        else:
            return math.log(p)
        '''

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
            if VERBOSE:
                print("Computing cost between", curr_ep, next_ep)
            cost += self.GetEpPairCost(curr_ep, next_ep, curr_time, next_time)
        return cost

    def FindMinCostAssignment(
        self, incoming_span, outgoing_eps, outgoing_span_partitions
    ):
        global best_assignment
        global best_score
        best_assignment = None
        best_score = -1000000.0
        def DfsTraverse(stack):
            global best_assignment
            global best_score
            i = len(stack)
            if VERBOSE:
                print("DFSTraverse", i, outgoing_eps, stack)
            last_span = stack[-1]
            if i == len(outgoing_span_partitions) + 1:
                score = self.ScoreAssignment(stack)
                if best_score < score:
                    best_assignment = stack
                    best_score = score
            else:
                #!TODO: filter out branches that have high cost
                ep = outgoing_eps[i-1]
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
        ret = {}
        if best_assignment is not None:
            # return a dictionary of {ep: trace_id}
            assert len(outgoing_eps) == len(best_assignment) - 1
            for i in range(len(outgoing_eps)):
                ret[outgoing_eps[i]] = best_assignment[i + 1]
        return ret

    def AssignSpans(
        self,
        incoming_span,
        assignment,
        assignments_dict,
        outgoing_span_partitions,
        outgoing_eps,
    ):
        # update assignments dict
        for ep in outgoing_eps:
            if ep not in assignments_dict:
                assignments_dict[ep] = []
            span = assignment.get(ep, None)
            trace_id = span.trace_id if span is not None else "NA"
            assignments_dict[ep].append(trace_id)

        # remove assignment spans so that they can't be assigned again
        #!TODO: this implementation is not efficient
        for ep, span in assignment.items():
            outgoing_span_partitions[ep].remove(span)

    def PredictTraceIdSequences(
        self, process, incoming_span_partitions, outgoing_span_partitions
    ):
        assert len(incoming_span_partitions) == 1
        ep, incoming_spans = list(incoming_span_partitions.items())[0]
        outgoing_eps = self.GetOutgoingSpanOrder(outgoing_span_partitions)
        self.PopulateEpPairDistributions(
            incoming_span_partitions, outgoing_span_partitions, outgoing_eps
        )
        outgoing_span_partitions_copy = copy.deepcopy(outgoing_span_partitions)
        assignments_dict = {}
        cnt = 0
        cnt_na = 0
        for incoming_span in incoming_spans:
            # find the minimimum cost label assignment for span
            min_cost_assignment = self.FindMinCostAssignment(
                incoming_span, outgoing_eps, outgoing_span_partitions_copy
            )
            self.AssignSpans(
                incoming_span,
                min_cost_assignment,
                assignments_dict,
                outgoing_span_partitions_copy,
                outgoing_eps,
            )
            cnt += 1
            cnt_na += (len(min_cost_assignment) == 0)
            if cnt % 50 == 0:
                print("Finished %d spans, unassigned spans: %d"%(cnt, cnt_na))
            #!TODO: update mean, std of service times using EWMA
        #print("Assignment_dict", assignments_dict)
        return assignments_dict
