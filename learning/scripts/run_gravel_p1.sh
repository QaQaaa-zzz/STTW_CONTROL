#!/usr/bin/env bash
set -e
GRAVEL_RUN=runs/h75_gravel_p1
trap '/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "石子P1失败，日志已保留，请回复继续处理。" --receipt "$GRAVEL_RUN/error_notice.json"' ERR
bash learning/scripts/gpu_queue.sh /home/qy/miniconda3/envs/mjp/bin/python -u learning/cli/evaluate_isaac_terrain_sweep.py --config learning/configs/isaac_h75_gravel_p1.json --assets "$GRAVEL_RUN/assets.json" --output "$GRAVEL_RUN/evaluation" --batch-cases 341 --resume >"$GRAVEL_RUN/runtime.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/cli/analyze_gravel_p0.py --evaluation "$GRAVEL_RUN/evaluation" --config learning/configs/isaac_h75_gravel_p1.json >"$GRAVEL_RUN/contact_analysis.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "P1因子分离矩阵完成：360条含72条原P0复用；未自动启动后续阶段。" --receipt "$GRAVEL_RUN/completion_notice.json"
