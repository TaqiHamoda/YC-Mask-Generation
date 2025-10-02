#!/bin/bash

# Activate virtual environment (optional)
# source /path/to/venv/bin/activate

# List of directories to process
dirs=(
    "/media/vicorob/Filesystem2/YC/250326/plot1_m1_250326_gps"
    "/media/vicorob/Filesystem2/YC/250326/plot1_m2_250326_gps"
    "/media/vicorob/Filesystem2/YC/250326/plot1_m3_250326_gps"
)

# Loop over each directory
for dir in "${dirs[@]}"; do
    # Execute ReadCSV.arg.py
    python3 ReadCSV-arg.py --input_csv "${dir}/seedpoints_on_images.csv" --output_dir "${dir}/"
    # Execute Inference-arg.py
    python3 Inference-arg.py --input_folder "${dir}"
done