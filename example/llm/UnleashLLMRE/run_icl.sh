#!/bin/bash

TRAIN_PATH="./data/train.json"
TEST_PATH="./data/test.json"
OUTPUT="./output_icl"

echo "Running model..."
python T5ICL.py \
  --train_path $TRAIN_PATH \
  --test_path $TEST_PATH \
  --output_success $OUTPUT \
  --output_nores $OUTPUT \
  --prompt 'instruct'\
  --k 1 \
