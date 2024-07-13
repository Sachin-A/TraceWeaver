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

test_name_suffix="test"
results_directory=$(pwd)/results/
mkdir -p $results_directory

tests=(
    "data/hotel-load25/ 0 0 2 hotel_$test_name_suffix 25 1 1 0 $results_directory"
    "data/hotel-load50/ 0 0 2 hotel_$test_name_suffix 50 1 1 0 $results_directory"
    "data/hotel-load75/ 0 0 2 hotel_$test_name_suffix 75 1 1 0 $results_directory"
    "data/hotel-load100/ 0 0 2 hotel_$test_name_suffix 100 1 1 0 $results_directory"
    "data/hotel-load125/ 0 0 2 hotel_$test_name_suffix 125 1 1 0 $results_directory"
    "data/hotel-load150/ 0 0 2 hotel_$test_name_suffix 150 1 1 0 $results_directory"

    "data/node-load25/ 0 0 0 node_$test_name_suffix 25 1 1 0 $results_directory"
    "data/node-load50/ 0 0 0 node_$test_name_suffix 50 1 1 0 $results_directory"
    "data/node-load75/ 0 0 0 node_$test_name_suffix 75 1 1 0 $results_directory"
    "data/node-load100/ 0 0 0 node_$test_name_suffix 100 1 1 0 $results_directory"
    "data/node-load125/ 0 0 0 node_$test_name_suffix 125 1 1 0 $results_directory"
    "data/node-load150/ 0 0 0 node_$test_name_suffix 150 1 1 0 $results_directory"

    "data/media-load25/ 0 0 1 media_$test_name_suffix 25 1 1 0 $results_directory"
    "data/media-load50/ 0 0 1 media_$test_name_suffix 50 1 1 0 $results_directory"
    "data/media-load75/ 0 0 1 media_$test_name_suffix 75 1 1 0 $results_directory"
    "data/media-load100/ 0 0 1 media_$test_name_suffix 100 1 1 0 $results_directory"
    "data/media-load125/ 0 0 1 media_$test_name_suffix 125 1 1 0 $results_directory"
    "data/media-load150/ 0 0 1 media_$test_name_suffix 150 1 1 0 $results_directory"
)

for test in "${tests[@]}"; do
    params=($test)
    run_python_script "${params[0]}" "${params[1]}" "${params[2]}" "${params[3]}" "${params[4]}" "${params[5]}" "${params[6]}" "${params[7]}" "${params[8]}" "${params[9]}"
done

wait

echo "All tests have concluded."

output_file_name_1="$results_directory/fig1a.pdf"
python3.11 "$root/utils/plot_accuracy_vs_load_multiple_apps.py" $results_directory $test_name_suffix $output_file_name_1

output_file_name_2="$results_directory/fig1b.pdf"
python3.11 "$root/utils/plot_accuracy_vs_response_times_multiple_apps.py" $results_directory $test_name_suffix $output_file_name_2
