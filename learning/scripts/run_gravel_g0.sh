#!/usr/bin/env bash
set -e
GRAVEL_RUN=runs/h75_gravel_g0
GRAVEL_PY=/home/qy/miniconda3/envs/env_isaaclab/bin/python
GRAVEL_ANALYSIS=/home/qy/miniconda3/envs/env_isaaclab/bin/python
trap '/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "G0失败，已保留日志；请回复继续以处理异常。" --receipt "$GRAVEL_RUN/error_notice.json"' ERR
bash learning/scripts/gpu_queue.sh "$GRAVEL_PY" -u learning/cli/evaluate_isaac_gpu.py --backend lab --configured-lanes --bundle runs/h75_selected_local300_flat/bundle --config "$GRAVEL_RUN/g0_config.json" --terrain "$GRAVEL_RUN/assets/S2_10mm" --output "$GRAVEL_RUN/evaluation" --seconds 10 >"$GRAVEL_RUN/evaluation.log" 2>&1
"$GRAVEL_ANALYSIS" learning/cli/analyze_gravel_contact_phase.py --run "$GRAVEL_RUN/evaluation" --assets "$GRAVEL_RUN/assets.json" >"$GRAVEL_RUN/analysis.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "G0四条检查与接触/分辨率分析完成。P0尚未启动。请回复继续，查看验收结果。" --receipt "$GRAVEL_RUN/completion_notice.json"
