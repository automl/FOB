#!/bin/bash

# List of experiment paths (input)
input_dir_list=("[../example/experiments/mnist/adamw_baseline]"
                "[../example/experiments/mnist/adamw_baseline, ../example/experiments/mnist/sgd_baseline]"
                )

# List of names for the output
exp_name_list=("mnist_adamw"
                "mnist_adamw-sgd"
                )

output_file="log_create_plots.txt"
yaml="plot.yaml"
plotting_script="../../evaluate_experiment.py"

# Clear contents of the output file
> "$output_file"

# Iterate over pairs of input directory and target names 
for i in "${!input_dir_list[@]}"; do
    input_dir="${input_dir_list[i]}"
    exp_name="${exp_name_list[i]}"

    python_output=$(python "$plotting_script" "$yaml" "data_dirs=$input_dir" "experiment_name=$exp_name")
    echo "Output of Python command for directory '$input_dir' is:" >> "$output_file"
    echo "  $python_output" >> "$output_file"
    echo "" >> "$output_file"

done
