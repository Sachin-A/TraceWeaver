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
    --results_directory "${10}" &
}

test_name_suffix="interleaving"
results_directory=$(pwd)/results/
mkdir -p $results_directory

tests=(
    "data/node-0/ 0 0 0 node_0_${test_name_suffix} 50 1 1 0 $results_directory"
    "data/node-0.2/ 0 0 0 node_0.2_${test_name_suffix} 50 1 1 0 $results_directory"
    "data/node-0.4/ 0 0 0 node_0.4_${test_name_suffix} 50 1 1 0 $results_directory"
    "data/node-0.6/ 0 0 0 node_0.6_${test_name_suffix} 50 1 1 0 $results_directory"
    "data/node-0.8/ 0 0 0 node_0.8_${test_name_suffix} 50 1 1 0 $results_directory"
    "data/node-1/ 0 0 0 node_1_${test_name_suffix} 50 1 1 0 $results_directory"
)

for test in "${tests[@]}"; do
    params=($test)
    run_python_script "${params[0]}" "${params[1]}" "${params[2]}" "${params[3]}" "${params[4]}" "${params[5]}" "${params[6]}" "${params[7]}" "${params[8]}" "${params[9]}"
done

wait

echo "All tests have concluded."

output_file_name_1="$results_directory/fig4.pdf"
python3.11 "$root/utils/plot_accuracy_vs_interleaving_intensity.py" $results_directory $test_name_suffix $output_file_name_1
