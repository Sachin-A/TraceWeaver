import os
import csv
import tarfile
from pathlib import Path

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

path = "/scratch/sachina3/projects/clusterdata/cluster-trace-microservices-v2021/data/MSCallGraph/"

num_shards = 144
num_lookback_errors = 0
overall_trace_ids = [[] for i in range(num_shards + 1)]
original_dataset = {}

for dataset_id in range(0, num_shards + 1):
    uncompress(path, path + "MSCallGraph_" + str(dataset_id) + ".tar.gz")

    Path(path + "shard" + str(dataset_id) + "/").mkdir(parents = True, exist_ok = True)

    traces = {}
    trace_ids = []

    with open(path + "MSCallGraph_" + str(dataset_id) + ".csv", mode = "r") as file:
        csv_file = csv.reader(file)
        trace_count = 1
        for row in csv_file:
            if row[1] == 'traceid':
                continue
            if row[1] not in traces:
                trace_ids.append(row[1])
                overall_trace_ids[dataset_id].append(row[1])
                traces[row[1]] = []
                if trace_count % 10000 == 0:
                    print("Reading: ", dataset_id, trace_count)
                trace_count += 1
            if row[1] not in original_dataset:
                original_dataset[row[1]] = dataset_id
            traces[row[1]].append(row)

    for i in range(len(trace_ids)):
        lookback_flag = False
        if i % 10000 == 0:
            print("Writing: ", dataset_id, i)

        output_path = returnOutputPath(trace_ids[i])

        if os.path.isfile(output_path + trace_ids[i] + ".tar.gz"):
            print("Lookback fail: ", trace_ids[i])
            num_lookback_errors += 1
            lookback_flag = True
            uncompress(output_path, output_path + trace_ids[i] + ".tar.gz")
            os.remove(output_path + trace_ids[i] + ".tar.gz")

        with open(output_path + trace_ids[i] + ".csv", "a+", newline = "") as csvfile:
            log_writer = csv.writer(csvfile, delimiter = ',')
            for row in traces[trace_ids[i]]:
                log_writer.writerow(row)

        if lookback_flag:
            print("Compressing: ", trace_ids[i])
            compress(output_path + str(trace_ids[i]) + ".tar.gz", output_path + str(trace_ids[i]) + ".csv")
            os.remove(output_path + str(trace_ids[i]) + ".csv")

    lookback = 5
    if dataset_id >= lookback:
        if dataset_id != num_shards:
            for b in overall_trace_ids[dataset_id - lookback]:
                output_path = returnOutputPath(b)
                if os.path.isfile(output_path + b + ".tar.gz"):
                    continue
                else:
                    compress(output_path + b + ".tar.gz", output_path + b + ".csv")
                    os.remove(output_path + b + ".csv")
        else:
            for x in range(num_shards - lookback, num_shards + 1):
                for b in overall_trace_ids[x]:
                    output_path = returnOutputPath(b)
                    if os.path.isfile(output_path + b + ".tar.gz"):
                        continue
                    else:
                        compress(output_path + b + ".tar.gz", output_path + b + ".csv")
                        os.remove(output_path + b + ".csv")

    os.remove(path + "MSCallGraph_" + str(dataset_id) + ".csv")
    print("Done with dataset: ", dataset_id)
    print("Lookback errors: ", num_lookback_errors)

print("Lookback errors: ", num_lookback_errors)
