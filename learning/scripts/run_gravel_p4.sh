#!/usr/bin/env bash
set -e
GRAVEL_RUN=runs/h75_gravel_p4
trap '/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "石子P4失败，已保留日志。请回复继续处理。" --receipt "$GRAVEL_RUN/error_notice.json"' ERR
bash learning/scripts/gpu_queue.sh /home/qy/miniconda3/envs/env_isaaclab/bin/python -u learning/cli/evaluate_isaac_gpu.py --backend lab --configured-lanes --bundle runs/h75_selected_local300_flat/bundle --config "$GRAVEL_RUN/settings.json" --terrain "$GRAVEL_RUN/assets/S2_map20261108" --output "$GRAVEL_RUN/batch/evaluation" --seconds 10 >"$GRAVEL_RUN/runtime.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/cli/analyze_gravel_selection.py --run "$GRAVEL_RUN" >"$GRAVEL_RUN/analysis.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "P4首200逻辑对照（152条去重物理轨迹）完成；收益门槛已保存，不自动扩展或重新建表。" --receipt "$GRAVEL_RUN/completion_notice.json"
