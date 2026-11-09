#!/usr/bin/env bash
set -e
RIDGE_RUN=runs/h75_ridge_heights
trap '/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "新增1/3/5/8cm凸脊评测失败，日志已保留。" --receipt "$RIDGE_RUN/error_notice.json"' ERR
for RIDGE_HEIGHT in 1 3 5 8; do
 LD_LIBRARY_PATH=/home/qy/miniconda3/envs/go2_isaacgym/lib:${LD_LIBRARY_PATH:-} /home/qy/miniconda3/envs/go2_isaacgym/bin/python learning/cli/place_isaac_terrain.py --terrain bend_bump --scene-config "$RIDGE_RUN/oblique_${RIDGE_HEIGHT}cm.json" --output "$RIDGE_RUN/oblique_${RIDGE_HEIGHT}cm" >"$RIDGE_RUN/terrain_${RIDGE_HEIGHT}cm.log" 2>&1
done
bash learning/scripts/gpu_queue.sh /home/qy/miniconda3/envs/mjp/bin/python -u learning/cli/evaluate_isaac_terrain_sweep.py --config learning/configs/isaac_h75_ridge_heights.json --assets "$RIDGE_RUN/assets.json" --output "$RIDGE_RUN/evaluation" --batch-cases 341 "$@" >"$RIDGE_RUN/runtime.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/cli/analyze_gravel_p0.py --evaluation "$RIDGE_RUN/evaluation" --config learning/configs/isaac_h75_ridge_heights.json >"$RIDGE_RUN/contact_analysis.log" 2>&1
/home/qy/miniconda3/envs/mjp/bin/python learning/scripts/notify_sttw_job.py --message "新增1/3/5/8cm凸脊144条评测与XY、跨alpha均值图、接触指标完成，零训练更新。" --receipt "$RIDGE_RUN/completion_notice.json"
