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

test_name_suffix="cache_rate"
results_directory=$(pwd)/results/
mkdir -p $results_directory

tests=(
    "data/hotel-load150/ 0 0.0 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.05 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.1 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.15 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.2 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.25 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.3 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.35 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.4 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.45 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.5 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.55 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.6 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.65 2 $test_name_suffix 150 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0.7 2 $test_name_suffix 150 1 1 0 $results_directory"
)

for test in "${tests[@]}"; do
    params=($test)
    run_python_script "${params[0]}" "${params[1]}" "${params[2]}" "${params[3]}" "${params[4]}" "${params[5]}" "${params[6]}" "${params[7]}" "${params[8]}" "${params[9]}"
done

wait

echo "All tests have concluded."

output_file_name_1="$results_directory/fig2.pdf"
python3.11 "$root/utils/plot_accuracy_vs_cache_hit_rate.py" $results_directory $test_name_suffix $output_file_name_1
