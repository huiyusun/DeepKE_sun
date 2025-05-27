#!/bin/bash

MODEL=modelsDA_skewed.py
DATASET="tacred"
DEMO_PATH="./${DATASET}/train.json"
OUTPUT_DIR="./generated/${DATASET}"

echo "Running model..."
python $MODEL \
  --api_key $OPEN_AI_API \
  --demo_path $DEMO_PATH \
  --output_dir $OUTPUT_DIR \
  --dataset $DATASET \
  --k 8 \
  --timestamp_output