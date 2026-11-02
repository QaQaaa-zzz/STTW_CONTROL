"""Summarize a completed nominal five-alpha screen against its original path."""
import json
import math
import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: task_preference_screen.py RUN_DIR")
    root = Path(sys.argv[1])
    plan = json.loads((root / "plan.json").read_text())
    rows = json.loads((root / "analysis/results.json").read_text())
    task = plan["protocol"]["task_design_diagnostic"]
    narrow = task["narrow_centerline_half_width_m"]
    wide = task["wide_centerline_half_width_m"]
    output = []
    for row in rows:
        with np.load(row["diagnostic_task_file"]) as trace:
            cross = np.asarray(trace["original_cross_track_m"])
        observed = int(row["active_lower_ticks"])
        arrival = row["endpoint"]["first_arrival_s"]
        before_goal = min(observed, int(math.ceil(arrival / 0.005))) if arrival is not None else observed
        max_cross = float(np.max(np.abs(cross[:before_goal])))
        tail = cross[1600:2000] if observed == 2000 else None
        turn_speed = row["actual_turn_speed"]["mean_m_s"]
        output.append({
            "case_id": row["case_id"],
            "bank": row["bank_index"],
            "route": row["route"],
            "alpha": row["alpha"],
            "requested_speed_m_s": row["speed"],
            "turn_speed_mean_m_s": turn_speed,
            "turn_speed_deficit_m_s": row["speed"] - turn_speed if turn_speed is not None else None,
            "original_goal_arrived": row["endpoint"]["goal_arrived"],
            "first_arrival_s": arrival,
            "physical_outcome": row["endpoint"]["physical_outcome"],
            "pre_goal_max_abs_original_cross_track_m": max_cross,
            "narrow_centerline_proxy_pass": row["endpoint"]["goal_arrived"] and row["endpoint"]["physical_outcome"] == "survived" and max_cross <= narrow,
            "wide_centerline_proxy_pass": row["endpoint"]["goal_arrived"] and row["endpoint"]["physical_outcome"] == "survived" and max_cross <= wide,
            "last_2s_original_cross_rmse_m": float(np.sqrt(np.mean(tail**2))) if tail is not None else None,
        })
    alphas = plan["protocol"]["alphas"]
    summary = []
    for alpha in alphas:
        group = [row for row in output if row["alpha"] == alpha]
        observed_speed = [row["turn_speed_mean_m_s"] for row in group if row["turn_speed_mean_m_s"] is not None]
        summary.append({
            "alpha": alpha,
            "n": len(group),
            "arrived": sum(row["original_goal_arrived"] for row in group),
            "survived": sum(row["physical_outcome"] == "survived" for row in group),
            "narrow_centerline_proxy_pass": sum(row["narrow_centerline_proxy_pass"] for row in group),
            "wide_centerline_proxy_pass": sum(row["wide_centerline_proxy_pass"] for row in group),
            "turn_speed_observed_cases": len(observed_speed),
            "mean_turn_speed_m_s": sum(observed_speed) / len(observed_speed) if observed_speed else None,
        })
    result = {"data_role": plan["data_role"], "scope": "centerline feasibility diagnostic only; vehicle footprint, speed-threshold and final task success unverified", "widths_m": {"narrow": narrow, "wide": wide}, "rows": output, "summary": summary}
    target = root / "analysis/task_preference_diagnostic.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Task preference nominal screen", "", "Centerline diagnostic only. Original-path deviations are measured before first original-goal entry; the 8–10 s error is reported separately. The centerline proxy does not certify vehicle-body corridor passage or the speed-preserving task.", "", "## Actual/reference XY panels", ""]
    frozen_id = plan["protocol"].get("frozen_registry_middle_id")
    suffix = "rule" if plan["protocol"].get("traditional_rule") else "candidate" if plan["protocol"].get("candidate_model") else {"STTW_MIDDLE_FULLTAIL_PPO75_U77": "h75", "STTW_MIDDLE_WITNESS_S918501_SUP400": "w400", "STTW_MIDDLE_DISTILLED_V3_U325": "u325"}.get(frozen_id, "h75")
    for bank in plan["protocol"]["bank_indices"]:
        for route in (context["route"] for context in plan["contexts"]):
            name = f"bank{bank}_{route}_zero_fixed_{suffix}_xy.png"
            lines.append(f"![bank{bank} {route} actual/reference XY]({name})")
    lines += ["", "## All five fixed preferences", "", "| alpha | n | original goal | survived H10 | narrow centerline proxy | wide centerline proxy | mean turn speed (m/s) |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        speed = f"{row['mean_turn_speed_m_s']:.3f}" if row["mean_turn_speed_m_s"] is not None else "not observed"
        lines.append(f"| {row['alpha']:g} | {row['n']} | {row['arrived']}/{row['n']} | {row['survived']}/{row['n']} | {row['narrow_centerline_proxy_pass']}/{row['n']} | {row['wide_centerline_proxy_pass']}/{row['n']} | {speed} ({row['turn_speed_observed_cases']}/{row['n']}) |")
    lines += ["", "All 40 individual outcomes and speed/path measurements are in [task_preference_diagnostic.json](task_preference_diagnostic.json). This screen used the frozen U170/local200 controller and 10 Hz causal reconnection, with zero optimizer updates. It is development evidence; no tuned conventional comparator, vehicle footprint check or independent final test has been run.", ""]
    (root / "analysis/TASK_SCREEN.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
