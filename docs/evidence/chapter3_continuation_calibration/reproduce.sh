#!/bin/bash
set -e
# From the sttw-offroad worktree; choose a fresh output path.
# Prepare and run using the documented source config, then analyze.
# Raw run identity and fixed inputs are preserved in config.json/declaration.json.
PYTHONPATH=learning/src /home/qy/miniconda3/envs/env_isaaclab/bin/python learning/cli/evaluate_ditch_batch.py --config "$1/config.json" --output "$1/evaluation"
/home/qy/miniconda3/envs/env_isaaclab/bin/python learning/cli/analyze_continuation_calibration.py --run "$1"
