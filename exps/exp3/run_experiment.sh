#!/bin/bash

root="../.."

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
test_name_suffix="interleaving"
results_directory=$(pwd)/results/
rm -rf $results_directory
mkdir -p $results_directory

# Predictor options
# ["MaxScoreBatch", "Timing2(all_spans, all_processes)"],
# ["MaxScoreBatchParallel", "Timing2(all_spans, all_processes)"],
# ["MaxScore", "Timing(all_spans, all_processes)"],
# ["WAP5", "WAP5(all_spans, all_processes)"],
# ["FCFS", "FCFS(all_spans, all_processes)"],
# ["ArrivalOrder", "ArrivalOrder(all_spans, all_processes)"],
# ["vPath", "vPath(all_spans, all_processes)"],
# ["DeepFlow", "DeepFlow(all_spans, all_processes)"],
# ["MaxScoreBatchParallelWithoutIterations", "Timing3(all_spans, all_processes)"],
# ["MaxScoreBatchParallel", "Timing3(all_spans, all_processes)"],
# ["MaxScoreBatchSubsetWithSkips", "Timing3(all_spans, all_processes)"]

predictor_indices="7,10"

tests=(
    "data/nodejs_microservices_with_arbitrary_file_io/node_0/ 0 0 0 node_0_${test_name_suffix} 50 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/nodejs_microservices_with_arbitrary_file_io/node_0.2/ 0 0 0 node_0.2_${test_name_suffix} 50 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/nodejs_microservices_with_arbitrary_file_io/node_0.4/ 0 0 0 node_0.4_${test_name_suffix} 50 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/nodejs_microservices_with_arbitrary_file_io/node_0.6/ 0 0 0 node_0.6_${test_name_suffix} 50 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/nodejs_microservices_with_arbitrary_file_io/node_0.8/ 0 0 0 node_0.8_${test_name_suffix} 50 1 1 0 $results_directory $clear_cache $predictor_indices"
    "data/nodejs_microservices_with_arbitrary_file_io/node_1/ 0 0 0 node_1_${test_name_suffix} 50 1 1 0 $results_directory $clear_cache $predictor_indices"
)

for test in "${tests[@]}"; do
    params=($test)
    run_python_script "${params[0]}" "${params[1]}" "${params[2]}" "${params[3]}" "${params[4]}" "${params[5]}" "${params[6]}" "${params[7]}" "${params[8]}" "${params[9]}" "${params[10]}" "${params[11]}"
done

wait

echo "All tests have concluded."

output_file_name_1="$results_directory/fig4.pdf"
python3.11 "$root/utils/plot_accuracy_vs_interleaving_intensity.py" $results_directory $test_name_suffix $output_file_name_1
