#!/usr/bin/env bash
set -e
GRAVEL_RUN=runs/h75_gravel_p3
trap '/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "石子P3失败，已保留日志。请回复继续处理。" --receipt "$GRAVEL_RUN/error_notice.json"' ERR
bash learning/scripts/gpu_queue.sh /home/qy/miniconda3/envs/env_isaaclab/bin/python -u learning/cli/evaluate_isaac_gpu.py --backend lab --configured-lanes --bundle runs/h75_selected_local300_flat/bundle --config "$GRAVEL_RUN/settings.json" --terrain runs/h75_gravel_p1/assets/S6 --output "$GRAVEL_RUN/batch/evaluation" --seconds 10 >"$GRAVEL_RUN/runtime.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/cli/analyze_gravel_recovery.py --run "$GRAVEL_RUN" >"$GRAVEL_RUN/analysis.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "P3首75条同起点恢复档评测完成。需先审查prefix及实际收益，不自动扩展或启动P4。" --receipt "$GRAVEL_RUN/completion_notice.json"
