#!/bin/bash

# List of experiment paths
input_dir_list=("[../evaluation/example/experiments/mnist/adamw_baseline]"
                "[../evaluation/example/experiments/mnist/adamw_baseline, ../evaluation/example/experiments/mnist/sgd_baseline]"
                )

exp_name_list=("exp1"
                "exp2"
                )

output_file="log.txt"
yaml="plot.yaml"
plotting_script="../evaluate_experiment.py"
# Clear contents of the output file
> "$output_file"


# Iterate over each input directory
for i in "${!input_dir_list[@]}"; do
    input_dir="${input_dir_list[i]}"
    exp_name="${exp_name_list[i]}"

    python_output=$(python "$plotting_script" "$yaml" "data_dirs=$input_dir" "experiment_name=$exp_name")
    echo "Output of Python command for directory '$input_dir' is:" >> "$output_file"
    echo "  $python_output" >> "$output_file"
    echo "" >> "$output_file"

done
