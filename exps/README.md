# Reproducing experiments

cd into each experiment sub-directory and run

    bash ./run_experiment.sh <clear_cache>

The script run_experiment.sh contains configuration parameters for the various experiments which are then run in parallel. For the provided dataset slices, the evaluation should finish running in <1 hour for each experiment depending on machine configuration. The output for experiment N would be generated within ./expN/results directory with intermediate results being in .pkl format and the final results plotted in .pdf format.

Pass <clear_cache> as 1 if you want to clear the pre-processed list of time-ordered file names corresponding to the traces, else 0. Clearing the cache is good as it re-populates the list if new traces have been added to the data directories or the directory structure has changed. Using the pre-populated cache enables fast iterations of the same experiment with varying parameters where the input data and the directory structure has not changed.
