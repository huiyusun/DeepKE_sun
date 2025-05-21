#!/bin/bash

DATASET="tacred"
DEMO_PATH="./data/train.json"
OUTPUT_DIR="./generated"

echo "Running model..."
python modelsDA.py \
  --demo_path $DEMO_PATH \
  --output_dir $OUTPUT_DIR \
  --dataset $DATASET \
  --k 3