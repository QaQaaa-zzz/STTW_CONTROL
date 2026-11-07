#!/usr/bin/env bash
set -e
GRAVEL_RUN=runs/h75_gravel_p0
trap '/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "P0流水线失败，已保留日志。请回复继续处理。" --receipt "$GRAVEL_RUN/error_notice.json"' ERR
bash learning/scripts/gpu_queue.sh /home/qy/miniconda3/envs/mjp/bin/python -u learning/cli/evaluate_isaac_terrain_sweep.py --config learning/configs/isaac_h75_gravel_sweep.json --assets runs/h75_gravel_g0/assets.json --output "$GRAVEL_RUN/evaluation" --batch-cases 24 >"$GRAVEL_RUN/runtime.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/cli/analyze_gravel_p0.py --evaluation "$GRAVEL_RUN/evaluation" --config learning/configs/isaac_h75_gravel_sweep.json >"$GRAVEL_RUN/contact_analysis.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "P0的144条石子对照与分析完成。请回复继续，判断接触/偏好差异；没有自动启动P1。" --receipt "$GRAVEL_RUN/completion_notice.json"
