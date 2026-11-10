#!/bin/bash
# Shell script chạy pipeline Stratified (Metadata-guided)
set -e

python3 scripts/run_experiment.py \
    --name stratified_seed42 \
    --pipeline stratified \
    --rounds 2 \
    --batch 500 \
    --budget 500 \
    --epochs 15 \
    --seed 42 \
    --device 0

