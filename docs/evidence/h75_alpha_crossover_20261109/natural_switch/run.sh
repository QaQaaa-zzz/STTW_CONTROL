#!/bin/bash
set -e
cd /home/qy/STTW_CONTROL/runs/worktrees/sttw-gravel
export PYTHONPATH="$PWD/learning/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 JAX_PLATFORMS=cpu
R=/media/qy/6A7E3DB07E3D75C3/STTW_runs/h75_alpha_natural_switch
N=/home/qy/miniconda3/envs/mjp/bin/python
trap '$N learning/scripts/notify_sttw_job.py --message "STTW无快照分段alpha确认实验异常，请查看h75_alpha_natural_switch/runtime.log" --receipt "$R/error_notice.json"' ERR
$N -u learning/cli/run_alpha_mechanism.py --run "$R" --resume >"$R/runtime_resume.log" 2>&1
$N learning/cli/summarize_alpha_schedules.py --run "$R" >"$R/analysis.log" 2>&1
$N learning/scripts/notify_sttw_job.py --message "STTW无快照分段alpha确认实验完成，12组11车道。h75_alpha_natural_switch/RESULTS.md" --receipt "$R/completion_notice.json"
