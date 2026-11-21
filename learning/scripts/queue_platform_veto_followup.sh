#!/bin/bash
set -eu
first=$1
follow=$2
upstream=$3
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/learning/src:$PWD/learning/cli:/home/qy/DVGC/JIT/src"
notify() {
    code=$?
    if ((code == 0)); then message='顺序复验已结束，诊断、采用判定与逐图统计见 veto_fallback_followup/RESULTS.md。'; else message="顺序复验出现错误，退出码 $code，见 veto_fallback_followup/queue.log。"; fi
    /usr/bin/python3 learning/scripts/notify_chapter3_job.py --title '第三章保底队列终态' --message "$message" --receipt "$follow/terminal_bark_$code.json"
}
trap notify EXIT
/home/qy/mujoco_playground/.venv/bin/python -u learning/cli/queue_platform_veto_followup.py "$first" "$follow" "$upstream" >"$follow/queue.log" 2>&1
