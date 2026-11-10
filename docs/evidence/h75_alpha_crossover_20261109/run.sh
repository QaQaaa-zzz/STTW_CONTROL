#!/bin/bash
set -e
cd /home/qy/STTW_CONTROL/runs/worktrees/sttw-gravel
export PYTHONPATH="$PWD/learning/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 JAX_PLATFORMS=cpu
R=/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_alpha_crossover
N=/home/qy/miniconda3/envs/mjp/bin/python
trap '$N learning/scripts/notify_sttw_job.py --message "STTW进场偏好交叉实验异常，请查看h75_alpha_crossover/runtime.log" --receipt "$R/error_notice.json"' ERR
$N -u learning/cli/run_alpha_mechanism.py --run "$R" >"$R/runtime.log" 2>&1
$N learning/scripts/notify_sttw_job.py --message "STTW进场偏好交叉实验完成，24组五分支。h75_alpha_crossover/RESULTS.md" --receipt "$R/completion_notice.json"
