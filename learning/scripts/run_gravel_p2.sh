#!/usr/bin/env bash
set -e
GRAVEL_RUN=runs/h75_gravel_p2
trap '/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "石子P2失败，已保留日志。请回复继续处理。" --receipt "$GRAVEL_RUN/error_notice.json"' ERR
bash learning/scripts/gpu_queue.sh /home/qy/miniconda3/envs/env_isaaclab/bin/python -u learning/cli/evaluate_isaac_gpu.py --backend lab --configured-lanes --bundle "$GRAVEL_RUN/bundle" --config "$GRAVEL_RUN/settings.json" --terrain "$GRAVEL_RUN/assets/gap0p2" --output "$GRAVEL_RUN/batch/evaluation" --seconds 10 >"$GRAVEL_RUN/runtime.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/cli/analyze_gravel_phase.py --run "$GRAVEL_RUN" >"$GRAVEL_RUN/analysis.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "P2三簇间距150条评测与分析完成，独立延长路线；未自动启动P3/P4。" --receipt "$GRAVEL_RUN/completion_notice.json"
