#!/bin/bash
# Shell script chạy pipeline Baseline (Uncertainty-only)
set -e

python3 scripts/run_experiment.py \
    --name baseline_seed42 \
    --pipeline baseline \
    --rounds 2 \
    --batch 500 \
    --budget 500 \
    --epochs 15 \
    --seed 42 \
    --device 0

