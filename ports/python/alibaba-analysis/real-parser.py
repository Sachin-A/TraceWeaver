import os
import csv
import random
import string
import tarfile
from pathlib import Path
from pprint import pprint
from distutils.version import LooseVersion

search_strings = ["(?)", "NAN", "nan", ""]

def compress(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

def uncompress(output_path, source_file):
    tarfile_object = tarfile.open(source_file)
    tarfile_object.extractall(output_path)
    tarfile_object.close()

def returnOutputPath(trace_id):
    original_shard = original_dataset[trace_id]
    output_path = path + "shard" + str(original_shard) + "/"
    return output_path

def getAllTracesInDir(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(directory + "/" + f)]
    files = [f for f in files if f.endswith("tar.gz")]
    return files

def fixDuplicates(trace):
    true_duplicate_count = 0
    rpc_id_duplicates = {}
    for span in trace:
        if span[3] not in rpc_id_duplicates:
            rpc_id_duplicates[span[3]] = []
        rpc_id_duplicates[span[3]].append(list(span))

    fake_duplicates = set()
    true_duplicates_a = set()
    true_duplicates_b = set()
    for i in range(len(trace)):
        pos = i + 1
        for j in range(pos, len(trace)):
            if trace[i][3] == trace[j][3]:
                if len(rpc_id_duplicates[trace[i][3]]) == 2:
                    if checkCopyRelaxed(trace[i], trace[j], -1):
                        fake_duplicates.add(j)
                    else:
                        true_duplicates_a.add(j)
                elif len(rpc_id_duplicates[trace[i][3]]) > 2:
                    true_duplicates_b.add(j)

    for k in sorted(fake_duplicates, reverse = True):
        del trace[k]

    return len(true_duplicates_a), len(true_duplicates_b)

def noSpanWith(trace, rpc_id):
    for span in trace:
        if span[3] == rpc_id:
            return False
    return True

def isRoot(span, trace):
    if ((span[3] == "0") or
        (span[3] == "0.1" and noSpanWith(trace, "0")) or
        (span[3] == "0.1.1" and noSpanWith(trace, "0.1") and noSpanWith(trace, "0"))
        ):
        return True
    return False

def isRoot2(span, len_root):
    if len(span[3].split(".")) == len_root:
        return True
    return False

def isLeaf(span, trace):
    for span0 in trace:
        parent_rpc_id = ".".join(span0[3].split(".")[:-1])
        if span[3] == parent_rpc_id:
            return False
    return True

def checkCopyRelaxed(span1, span2, col):
    if col == 4:
        indices = [1, 3, 5, 6]
    elif col == 6:
        indices = [1, 3, 4, 5]
    elif col == -1:
        indices = [1, 3, 4, 5, 6]

    sublist1 = [span1[x] for x in indices]
    sublist2 = [span2[x] for x in indices]
    if sublist1 == sublist2:
        if ((int(span1[8]) > 0 and int(span2[8]) < 0) or
            (int(span1[8]) < 0 and int(span2[8]) > 0) or
            (int(span1[8]) == 0 and int(span2[8]) == 0)):
            return True

    return False

def checkCopyStrict(span1, span2, col):
    if col == 4:
        indices = [1, 2, 3, 5, 6]
    elif col == 6:
        indices = [1, 2, 3, 4, 5]
    elif col == -1:
        indices = [1, 2, 3, 4, 5, 6]

    sublist1 = [span1[x] for x in indices]
    sublist2 = [span2[x] for x in indices]
    if sublist1 == sublist2:
        if abs(int(span1[8])) == abs(int(span2[8])):
            if ((int(span1[8]) > 0 and int(span2[8]) < 0) or
                (int(span1[8]) < 0 and int(span2[8]) > 0)):
                return True

    return False

def hasSameCallee(trace, parent_rpc_id):
    callees = set()
    for span in trace:
        if span[3] == parent_rpc_id:
            callees.add(span[6])
    if len(callees) == 1:
        return True
    return False

# Not efficient, fix if needed
def checkNeighbours(trace, row, col, rpc_id_count):

    if col != 4 and col != 6:
        print("New column missing:", col)
        return False

    rpc_id = trace[row][3]
    parent_rpc_id = ".".join(rpc_id.split(".")[:-1])

    for i, span in enumerate(trace):

        # trace[row] missing caller information
        if col == 4:

            # trace[row]'s parent
            if span[3] == parent_rpc_id and not missingCol(span[col + 2]):
                if rpc_id_count.get(span[3], 0) == 1 or hasSameCallee(trace, span[3]):
                    trace[row][col] = str(span[col + 2])
                    return True

            # trace[row]'s sibling
            if ".".join(span[3].split(".")[:-1]) == parent_rpc_id and not missingCol(span[col]):
                if rpc_id_count.get(span[3], 0) == 1 or hasSameCallee(trace, span[3]):
                    trace[row][col] = str(span[col])
                    return True

        # trace[row] missing callee information
        elif col == 6:

            # trace[row]'s child
            if ".".join(span[3].split(".")[:-1]) == rpc_id and not missingCol(span[col]):
                if rpc_id_count.get(span[3], 0) == 1 or hasSameCallee(trace, span[3]):
                    trace[row][col] = str(span[col - 2])
                    return True

        if (i != row and
            span[3] == rpc_id and
            not missingCol(span[col]) and
            rpc_id_count.get(span[3], 0) == 2 and
            checkCopyStrict(trace[row], span, col)):
                trace[row][col] = str(span[col])
                return True

    return False

def fixMissingInSpan(trace, row, rpc_id_count):
    for col, span_entry in enumerate(trace[row]):
        if col == 7:
            continue
        for search_string in search_strings:
            if search_string == span_entry:
                if not checkNeighbours(trace, row, col, rpc_id_count):
                    return False
    return True

def missingInfo(span):
    for i, span_entry in enumerate(span):
        for search_string in search_strings:
            if search_string == span_entry:
                return True
    return False

def missingCol(span_entry):
    for search_string in search_strings:
        if search_string == span_entry:
            return True
    return False

def fixRoot(trace, i, collapse = False):
    if missingCol(trace[i][4]):
        if collapse:
            trace[i][4] = "client"
        else:
            unique_key = ''.join(random.choices(string.digits, k = 16))
            trace[i][4] = "client_" + unique_key

def fixLeaf(trace, i, collapse = False):
    if missingCol(trace[i][6]):
        if collapse:
            trace[i][6] = "leaf"
        else:
            unique_key = ''.join(random.choices(string.digits, k = 16))
            trace[i][6] = "leaf_" + unique_key

def fixMissingInTrace(trace):

    missing_count = 0
    rpc_id_count = {}

    len_root = len(trace[0][3].split("."))

    for span in trace:
        if span[3] not in rpc_id_count:
            rpc_id_count[span[3]] = 1
        else:
            rpc_id_count[span[3]] += 1

    for i, span in enumerate(trace):

        sublist0 = span[:7] + span[7+1:]

        if missingInfo(sublist0):

            if fixMissingInSpan(trace, i, rpc_id_count):
                continue

            sublist1 = sublist0[:4] + sublist0[4+1:]
            if isRoot2(span, len_root) and not missingInfo(sublist1):
                fixRoot(trace, i, collapse = True)
                continue

            sublist2 = sublist0[:6] + sublist0[6+1:]
            if isLeaf(span, trace) and not missingInfo(sublist2):
                fixLeaf(trace, i, collapse = True)
                continue

            missing_count += 1

    return missing_count

def checkMultiRoot(trace):
    multi_root = 0
    len_root = len(trace[0][3].split("."))
    for i, span in enumerate(trace):
        if i != 0 and len(span[3].split(".")) == len_root:
            multi_root += 1

    return multi_root

def checkOrphans(trace):
    num_orphans = 0
    rpc_id_count = {}
    len_root = len(trace[0][3].split("."))

    for span in trace:
        if span[3] not in rpc_id_count:
            rpc_id_count[span[3]] = 0
        rpc_id_count[span[3]] += 1

    for span in trace:
        if isRoot2(span, len_root):
            continue
        elif rpc_id_count.get(".".join(span[3].split(".")[:-1]), 0) > 0:
            continue
        else:
            num_orphans += 1

    return num_orphans

def buildCallGraph(trace):
    trace_cg = {}
    trace_cg['root'] = trace[0][3]
    len_root = len(trace[0][3].split("."))

    for i, span in enumerate(trace):
        if i != 0 and len(span[3].split(".")) == len_root:
            print(trace[0])
            print(span)
            print("A")
            input()
            return {}, False

        if span[3] not in trace_cg:
            trace_cg[span[3]] = []
        else:
            print("B")
            return {}, False

        if i != 0:
            parent_rpc_id = ".".join(span[3].split(".")[:-1])
            if parent_rpc_id in trace_cg:
                trace_cg[parent_rpc_id].append(span[3])
            else:
                print("C")
                return {}, False

    return trace_cg, True

path = "/scratch/sachina3/projects/clusterdata/cluster-trace-microservices-v2021/data/MSCallGraph/"

# num_shards = 144
num_shards = 0
overall_trace_ids = [[] for i in range(num_shards + 1)]
original_dataset = {}
overall_missing = 0
overall_duplicates = 0
overall_orphans = 0
both = 0
neither = 0
no_errors = 0
overall_root_missing = 0
overall_multi_root = 0
distinct_roots = set()
overall_not_valid = 0

for dataset_id in range(0, num_shards + 1):

    shard_directory = path + "shard" + str(dataset_id) + "/"
    trace_paths = getAllTracesInDir(shard_directory)

    trace_count = 0

    for trace_path in trace_paths:

        trace_count += 1
        if trace_count % 1000 == 0:
            print("Reading: ", dataset_id, trace_count,
                  "\nMissing (a): ", overall_missing, (overall_missing * 100) / trace_count,
                  "\nDuplicates (b): ", overall_duplicates, (overall_duplicates * 100) / trace_count,
                  "\nOrphans (c): ", overall_orphans, (overall_orphans * 100) / trace_count,
                  "\nBoth (a, b): ", (both * 100) / trace_count,
                  "\nNeither (a, b): ", neither, (neither * 100) / trace_count,
                  "\nRoot missing: ", overall_root_missing, (overall_root_missing * 100) / trace_count,
                  "\nMulti-root (d): ", overall_multi_root, (overall_multi_root * 100) / trace_count,
                  "\nNot valid: ", overall_not_valid, (overall_not_valid * 100) / trace_count,
                  "\nDistinct roots: ", len(distinct_roots),
                  "\nNo errors (a, b, c, d): ", no_errors, (no_errors * 100) / trace_count)

        trace_id = ".".join(trace_path.split(".")[:-2])
        uncompress(shard_directory, shard_directory + trace_path)

        with open(shard_directory + trace_id + ".csv", mode = "r") as file:
            trace = []
            csv_file = csv.reader(file)
            for span in csv_file:
                trace.append(span)

        trace.sort(key = lambda x: LooseVersion(x[3]))
        os.remove(shard_directory + trace_id + ".csv")

        if len(trace) > 200:
            continue

        a = fixMissingInTrace(trace)
        b1, b2 = fixDuplicates(trace)
        c = checkOrphans(trace)
        d = checkMultiRoot(trace)

        if a > 0:
            overall_missing += 1

        if (b1 + b2) > 0:
            overall_duplicates += 1

        if c > 0:
            overall_orphans += 1

        if a > 0 and (b1 + b2) > 0:
            both += 1

        if a == 0 and (b1 + b2) == 0:
            neither += 1

        if d > 0:
            overall_multi_root += 1

        if a == 0 and (b1 + b2) == 0 and c == 0 and d == 0:
            no_errors += 1

            trace_cg, valid = buildCallGraph(trace)
            if not valid:
                print("Not valid")
                # input()
                overall_not_valid += 1
