import copy
import csv
import json
import os
import random
import shutil
import string
import tarfile
from distutils.version import LooseVersion
from pathlib import Path
from pprint import pprint

search_strings = ["(?)", "NAN", "nan", ""]

def compress(output_filename, source_dir, format = "w:gz"):
    with tarfile.open(output_filename, format) as tar:
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

# TODO: preserve original server record
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
            (int(span1[8]) < 0 and int(span2[8]) > 0)):
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
            checkCopyRelaxed(trace[row], span, col)):
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
    span_id_to_index = {}
    trace_cg['root'] = trace[0][3]
    len_root = len(trace[0][3].split("."))

    for i, span in enumerate(trace):
        span_id_to_index[span[3]] = i
        if i != 0 and len(span[3].split(".")) == len_root:
            return {}, {}, False

        if span[3] not in trace_cg:
            trace_cg[span[3]] = []
        else:
            return {}, {}, False

        if i != 0:
            parent_rpc_id = ".".join(span[3].split(".")[:-1])
            if parent_rpc_id in trace_cg:
                trace_cg[parent_rpc_id].append(span[3])
            else:
                return {}, {}, False

    return trace_cg, span_id_to_index, True

def convertToJaegerFormat(trace, trace_cg, span_id_to_index, dataset_id):
    trace_dict = {}
    trace_dict['data'] = []

    trace_data = {}
    trace_data['traceID'] = trace[0][1]
    trace_data['spans'] = []

    for j, span in enumerate(trace):
        if span[3] not in span_id_to_index:
            print("Skipped a span")
            continue

        server_record = {}
        server_record['traceID'] = span[1]
        server_record['startTime'] = int(span[2]) * 1000
        server_record['spanID'] = span[3]
        server_record['caller'] = span[4]
        server_record['requestType'] = span[5]
        server_record['callee'] = span[6]
        server_record['interface'] = span[7]
        server_record['duration'] = abs(int(span[8]) * 1000)

        server_record['tags'] = []
        tags_data = {}
        tags_data['key'] = "span.kind"
        tags_data['value'] = "server"
        server_record['tags'].append(tags_data)

        server_record['references'] = []
        references_data = {}
        if span[3] != trace_cg['root']:
            references_data['refType'] = "CHILD_OF"
            parent_rpc_id = ".".join(span[3].split(".")[:-1])
            references_data['traceID'] = span[1]
            references_data['spanID'] = parent_rpc_id
            server_record['references'].append(references_data)

        server_record['processID'] = span[6]
        trace_data['spans'].append(server_record)

        if span[3] != trace_cg['root']:
            client_record = copy.deepcopy(server_record)
            client_record['tags'][0]['value'] = "client"
            client_record['processID'] = span[4]
            trace_data['spans'].append(client_record)

    trace_dict['data'].append(trace_data)

    Path(path + 'traces' + str(dataset_id) + '/').mkdir(parents = True, exist_ok = True)
    with open(path + 'traces' + str(dataset_id) + '/' + trace_data['traceID'] + '.json', 'w', encoding = 'utf-8') as f:
        json.dump(trace_dict, f, ensure_ascii = False, indent = 4)

path = "/scratch/sachina3/projects/clusterdata/cluster-trace-microservices-v2021/data/MSCallGraph/"

num_shards = 144
# num_shards = 0
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
overall_valid = 0
trace_count = 0

for dataset_id in range(0, num_shards + 1):

    shard_directory = path + "shard" + str(dataset_id) + "/"
    trace_paths = getAllTracesInDir(shard_directory)

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
                  "\nValid: ", overall_valid, (overall_valid * 100) / trace_count,
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
            trace_cg, span_id_to_index, valid = buildCallGraph(trace)
            if not valid:
                overall_not_valid += 1
            else:
                overall_valid += 1
                convertToJaegerFormat(trace, trace_cg, span_id_to_index, dataset_id)

    compress(path + 'traces' + str(dataset_id) + '.tar.lama', path + 'traces' + str(dataset_id) + '/', "w:xz")
    shutil.rmtree(path + 'traces' + str(dataset_id) + '/')
