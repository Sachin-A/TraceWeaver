#!/bin/bash

root="../.."
data_directory="$root/data/alibaba_microservices"
tar_file="$data_directory/call_graph_data.tar.lzma"

download_tar_file() {
    local url="$1"
    echo "Downloading $url to $tar_file..."
    curl -o "$tar_file" "$url"
}

extract_tar_file() {
    echo "Extracting $tar_file ..."
    tar -xJf "$tar_file" -C "$data_directory"
}

run_python_script() {
    python3.11 "$root/src/trace_reconstructor/ports/python/executor.py" \
    --relative_path "$1" \
    --compressed "$2" \
    --cache_rate "$3" \
    --fix "$4" \
    --test_name "$5" \
    --load_level "$6" \
    --compress_factor "$7" \
    --repeat_factor "$8" \
    --execute_parallel "$9" \
    --results_directory "${10}" \
    --clear_cache "${11}" \
    --predictor_indices "${12}" &
}

clear_cache="$1"
test_name_suffix="load_multiple"
results_directory=$(pwd)/results/
rm -rf $results_directory
mkdir -p $results_directory

mkdir -p $data_directory
# tar_url="waiting_on_dryad_approval"
# download_tar_file "$tar_url"
extract_tar_file

# Predictor options
# ["MaxScoreBatch", "Timing2(all_spans, all_processes)"],
# ["MaxScoreBatchParallel", "Timing2(all_spans, all_processes)"],
# ["MaxScore", "Timing(all_spans, all_processes)"],
# ["WAP5", "WAP5(all_spans, all_processes)"],
# ["FCFS", "FCFS(all_spans, all_processes)"],
# ["ArrivalOrder", "ArrivalOrder(all_spans, all_processes)"],
# ["vPathOld", "vPathOld(all_spans, all_processes)"],
# ["vPath", "vPath(all_spans, all_processes)"],
# ["MaxScoreBatchParallelWithoutIterations", "Timing3(all_spans, all_processes)"],
# ["MaxScoreBatchParallel", "Timing3(all_spans, all_processes)"],
# ["MaxScoreBatchSubsetWithSkips", "Timing3(all_spans, all_processes)"]

predictor_indices="3,4,7,10"

tests=(
    "data/alibaba_microservices/call_graph_data/call_graph_0 0 0 5 alibaba_cg_0_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_1 0 0 5 alibaba_cg_1_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_2 0 0 5 alibaba_cg_2_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_3 0 0 5 alibaba_cg_3_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_4 0 0 5 alibaba_cg_4_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_5 0 0 5 alibaba_cg_5_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_6 0 0 5 alibaba_cg_6_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_7 0 0 5 alibaba_cg_7_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_8 0 0 5 alibaba_cg_8_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_9 0 0 5 alibaba_cg_9_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_10 0 0 5 alibaba_cg_10_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_11 0 0 5 alibaba_cg_11_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_12 0 0 5 alibaba_cg_12_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_13 0 0 5 alibaba_cg_13_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_14 0 0 5 alibaba_cg_14_$test_name_suffix 1 1 1 0 $results_directory $clear_cache $predictor_indices"

    "data/alibaba_microservices/call_graph_data/call_graph_0 0 0 5 alibaba_cg_0_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_1 0 0 5 alibaba_cg_1_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_2 0 0 5 alibaba_cg_2_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_3 0 0 5 alibaba_cg_3_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_4 0 0 5 alibaba_cg_4_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_5 0 0 5 alibaba_cg_5_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_6 0 0 5 alibaba_cg_6_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_7 0 0 5 alibaba_cg_7_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_8 0 0 5 alibaba_cg_8_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_9 0 0 5 alibaba_cg_9_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_10 0 0 5 alibaba_cg_10_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_11 0 0 5 alibaba_cg_11_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_12 0 0 5 alibaba_cg_12_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_13 0 0 5 alibaba_cg_13_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_14 0 0 5 alibaba_cg_14_$test_name_suffix 1 200 1 0 $results_directory $clear_cache $predictor_indices"

    "data/alibaba_microservices/call_graph_data/call_graph_0 0 0 5 alibaba_cg_0_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_1 0 0 5 alibaba_cg_1_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_2 0 0 5 alibaba_cg_2_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_3 0 0 5 alibaba_cg_3_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_4 0 0 5 alibaba_cg_4_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_5 0 0 5 alibaba_cg_5_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_6 0 0 5 alibaba_cg_6_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_7 0 0 5 alibaba_cg_7_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_8 0 0 5 alibaba_cg_8_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_9 0 0 5 alibaba_cg_9_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_10 0 0 5 alibaba_cg_10_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_11 0 0 5 alibaba_cg_11_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_12 0 0 5 alibaba_cg_12_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_13 0 0 5 alibaba_cg_13_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_14 0 0 5 alibaba_cg_14_$test_name_suffix 1 1000 1 0 $results_directory $clear_cache $predictor_indices"

    "data/alibaba_microservices/call_graph_data/call_graph_0 0 0 5 alibaba_cg_0_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_1 0 0 5 alibaba_cg_1_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_2 0 0 5 alibaba_cg_2_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_3 0 0 5 alibaba_cg_3_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_4 0 0 5 alibaba_cg_4_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_5 0 0 5 alibaba_cg_5_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_6 0 0 5 alibaba_cg_6_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_7 0 0 5 alibaba_cg_7_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_8 0 0 5 alibaba_cg_8_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_9 0 0 5 alibaba_cg_9_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_10 0 0 5 alibaba_cg_10_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_11 0 0 5 alibaba_cg_11_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_12 0 0 5 alibaba_cg_12_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_13 0 0 5 alibaba_cg_13_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_14 0 0 5 alibaba_cg_14_$test_name_suffix 1 4000 1 0 $results_directory $clear_cache $predictor_indices"

    "data/alibaba_microservices/call_graph_data/call_graph_0 0 0 5 alibaba_cg_0_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_1 0 0 5 alibaba_cg_1_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_2 0 0 5 alibaba_cg_2_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_3 0 0 5 alibaba_cg_3_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_4 0 0 5 alibaba_cg_4_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_5 0 0 5 alibaba_cg_5_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_6 0 0 5 alibaba_cg_6_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_7 0 0 5 alibaba_cg_7_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_8 0 0 5 alibaba_cg_8_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_9 0 0 5 alibaba_cg_9_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_10 0 0 5 alibaba_cg_10_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_11 0 0 5 alibaba_cg_11_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_12 0 0 5 alibaba_cg_12_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_13 0 0 5 alibaba_cg_13_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_14 0 0 5 alibaba_cg_14_$test_name_suffix 1 10000 1 0 $results_directory $clear_cache $predictor_indices"

    "data/alibaba_microservices/call_graph_data/call_graph_0 0 0 5 alibaba_cg_0_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_1 0 0 5 alibaba_cg_1_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_2 0 0 5 alibaba_cg_2_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_3 0 0 5 alibaba_cg_3_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_4 0 0 5 alibaba_cg_4_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_5 0 0 5 alibaba_cg_5_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_6 0 0 5 alibaba_cg_6_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_7 0 0 5 alibaba_cg_7_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_8 0 0 5 alibaba_cg_8_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_9 0 0 5 alibaba_cg_9_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_10 0 0 5 alibaba_cg_10_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_11 0 0 5 alibaba_cg_11_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_12 0 0 5 alibaba_cg_12_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_13 0 0 5 alibaba_cg_13_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
    "data/alibaba_microservices/call_graph_data/call_graph_14 0 0 5 alibaba_cg_14_$test_name_suffix 1 15000 1 0 $results_directory $clear_cache $predictor_indices"
)

for test in "${tests[@]}"; do
    params=($test)
    run_python_script "${params[0]}" "${params[1]}" "${params[2]}" "${params[3]}" "${params[4]}" "${params[5]}" "${params[6]}" "${params[7]}" "${params[8]}" "${params[9]}" "${params[10]}" "${params[11]}"
done

wait

echo "All tests have concluded."

output_file_name_1="$results_directory/fig6a.pdf"
python3.11 "$root/utils/plot_accuracy_vs_load_multiple_cgs.py" $results_directory $test_name_suffix $output_file_name_1

output_file_name_2="$results_directory/fig6b.pdf"
python3.11 "$root/utils/plot_accuracy_vs_confidence_multiple_cgs.py" $results_directory $test_name_suffix $output_file_name_2
