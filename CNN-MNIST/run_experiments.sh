#!/bin/sh
set -e

cd "$(dirname "$0")"

for model in 1 2; do
    i=1
    while [ "$i" -le 8 ]; do
        config_file="config/config_${i}_model_${model}.yaml"
        echo "Running experiment config_${i}_model_${model}: ${config_file}"
        python main.py --config "$config_file"
        i=$((i + 1))
    done
done

echo "All experiments finished."
