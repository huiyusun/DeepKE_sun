#!/bin/bash

DATASET="tacred"
DEMO_PATH="/Users/huiyu/Documents/research2025/datasets/TACRED/train.json"
OUTPUT_DIR="./generated_output"

if [ "$1" == "gpt3" ]; then
  echo "Running GPT-3 data augmentation..."
  python gpt3DA.py \
    --api_key $OPENAI_API_KEY \
    --demo_path $DEMO_PATH \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --k 3 \
    --timestamp_output

elif [ "$1" == "deepseek" ]; then
  echo "Running DeepSeek model..."
  python deepseekDA.py \
    --demo_path $DEMO_PATH \
    --output_dir $OUTPUT_DIR \
    --dataset $DATASET \
    --k 3 \
    --timestamp_output

else
  echo "Usage: ./run.sh [gpt3 | deepseek]"
  exit 1
fi