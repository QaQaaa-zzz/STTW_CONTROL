#!/bin/bash
set -e
cd /home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad
export PYTHONPATH="$PWD/learning/src"
/home/qy/miniconda3/envs/env_isaaclab/bin/python learning/cli/evaluate_ditch_batch.py --config /media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_shared_head_online/config.json --output /media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_shared_head_online/evaluation >/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_shared_head_online/run.log 2>&1
