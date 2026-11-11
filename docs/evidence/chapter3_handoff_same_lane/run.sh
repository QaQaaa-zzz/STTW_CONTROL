#!/bin/bash
set -e
cd /home/qy/STTW_CONTROL/runs/worktrees/sttw-offroad
export PYTHONPATH="$PWD/learning/src"
/home/qy/miniconda3/envs/env_isaaclab/bin/python learning/cli/evaluate_ditch_batch.py --config /media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_handoff_same_lane/cold_a/config.json --output /media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_handoff_same_lane/cold_a/evaluation >/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_handoff_same_lane/cold_a/run.log 2>&1
/home/qy/miniconda3/envs/env_isaaclab/bin/python learning/cli/evaluate_ditch_batch.py --config /media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_handoff_same_lane/cold_b/config.json --output /media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_handoff_same_lane/cold_b/evaluation >/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_handoff_same_lane/cold_b/run.log 2>&1
/home/qy/miniconda3/envs/env_isaaclab/bin/python learning/cli/evaluate_ditch_batch.py --config /media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_handoff_same_lane/measured/config.json --output /media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_handoff_same_lane/measured/evaluation >/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_handoff_same_lane/measured/run.log 2>&1
