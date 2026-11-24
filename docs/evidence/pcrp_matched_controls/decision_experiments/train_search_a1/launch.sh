#!/usr/bin/env bash
set -eu
work=/home/qy/STTW_CONTROL/runs/worktrees/pcrp-decision-delivery
stage="$(dirname "$(realpath "$0")")"
notify="$work/learning/scripts/notify_sttw_job.py"
if test "${1:-}" = wait30; then
  if systemctl --user is-active --quiet pcrp-window-a1.service; then
    /usr/bin/python3 "$notify" --message 'A1 等待已满半小时，固定预算下的并行搜索/复验仍在执行；完成或异常会单独通知。' --receipt "$stage/wait30_bark.json"
  fi
  exit 0
fi
if test "${1:-}" != leased; then
  exec /bin/bash "$work/learning/scripts/gpu_queue.sh" /bin/bash "$stage/launch.sh" leased
fi
export PYTHONPATH="$work/learning/src:$work/learning/cli"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
py=/home/qy/mujoco_playground/.venv/bin/python
/usr/bin/python3 "$notify" --message 'PCRP A1 开始。4×32并行，8代，含同槽零偏置和复验总上限1162轨迹。' --receipt "$stage/start_bark.json" || true
status=0
"$py" -B "$stage/a1.py" execute || status=$?
"$py" -B "$stage/a1.py" record-exit "$status"
result=$("$py" -c "import json,sys; r=json.load(open(sys.argv[1])); print(r.get('decision',r['stage']), r['finished'])" "$stage/status.json")
/usr/bin/python3 "$notify" --message "PCRP A1 结束。状态和已完成轨迹数 $result，退出码 $status。见 train_search_a1/status.json。" --receipt "$stage/terminal_bark.json"
exit "$status"
