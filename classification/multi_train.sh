#!/bin/bash

# Check if three parameters are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <config> <start_run> <end_run>"
    exit 1
fi

# Assign the command line arguments to variables
config=$1
start_run=$2
end_run=$3

# Loop through the specified range of runs
for ((i = start_run; i <= end_run; i++)); do
    echo "Running config $config, run $i"
    python train.py --config "$config" --run "run$i"
done
