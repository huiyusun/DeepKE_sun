#!/bin/bash

DATASET="tacred"
DEMO_PATH="./data/train.json"
OUTPUT_DIR="./generated"

echo "Running model..."
python modelsDA_skewed_dist.py \
  --api_key $OPEN_AI_API \
  --demo_path $DEMO_PATH \
  --output_dir $OUTPUT_DIR \
  --dataset $DATASET \
  --k 3