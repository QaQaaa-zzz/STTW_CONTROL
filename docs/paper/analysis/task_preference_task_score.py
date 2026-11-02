"""Score frozen task preferences from a completed native nominal screen."""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: task_preference_task_score.py RUN_DIR")
    root = Path(sys.argv[1])
    plan = json.loads((root / "plan.json").read_text())
    diagnostic = json.loads((root / "analysis/task_preference_diagnostic.json").read_text())
    contract = plan["protocol"]["task_contract"]
    rows = []
    for row in diagnostic["rows"]:
        common = (row["original_goal_arrived"] and row["physical_outcome"] == "survived"
                  and row["last_2s_original_cross_rmse_m"] is not None
                  and row["last_2s_original_cross_rmse_m"] <= contract["post_goal_last2s_original_cross_rmse_max_m"])
        outcome = dict(row)
        for task in ("narrow", "wide"):
            limits = contract[task]
            inside = row["pre_goal_max_abs_original_cross_track_m"] + contract["buffer_radius_m"] <= limits["corridor_half_width_m"]
            speed = limits["speed_min_ratio"] is None or (row["turn_speed_mean_m_s"] is not None
                    and row["turn_speed_mean_m_s"] >= limits["speed_min_ratio"] * row["requested_speed_m_s"])
            outcome[f"{task}_inside_proxy"] = inside
            outcome[f"{task}_speed_met"] = speed
            outcome[f"{task}_success"] = bool(common and inside and speed)
        rows.append(outcome)
    summary = []
    for alpha in plan["protocol"]["alphas"]:
        group = [row for row in rows if row["alpha"] == alpha]
        summary.append({"alpha": alpha, "n": len(group),
                        "narrow_success": sum(row["narrow_success"] for row in group),
                        "wide_success": sum(row["wide_success"] for row in group),
                        "narrow_corridor_proxy": sum(row["narrow_inside_proxy"] for row in group),
                        "wide_speed_met": sum(row["wide_speed_met"] for row in group)})
    result = {"contract": contract, "data_role": plan["data_role"], "rows": rows, "summary": summary}
    (root / "analysis/task_scores.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Frozen task-preference score", "", f"The corridor uses a {contract['buffer_radius_m']:g} m circular buffer around recorded vehicle XY. This is an anchor-disk proxy, not an exact swept-mesh or real-wall test. The original goal, H10 survival and 8–10 s original-path tracking are required for both tasks. Threshold provenance: {contract['source']}", "", "## Actual/reference XY", ""]
    frozen_id = plan["protocol"].get("frozen_registry_middle_id")
    suffix = "rule" if plan["protocol"].get("traditional_rule") else "candidate" if plan["protocol"].get("candidate_model") else {"STTW_MIDDLE_FULLTAIL_PPO75_U77": "h75", "STTW_MIDDLE_WITNESS_S918501_SUP400": "w400", "STTW_MIDDLE_DISTILLED_V3_U325": "u325"}.get(frozen_id, "h75")
    for image in sorted((root / "analysis").glob(f"*_fixed_{suffix}_xy.png")):
        lines.append(f"![{image.stem} actual/reference XY]({image.name})")
    lines += ["", "## Outcomes by fixed alpha", "", "| alpha | narrow task | wide task | narrow corridor proxy | wide speed threshold |", "|---:|---:|---:|---:|---:|"]
    for item in summary:
        lines.append(f"| {item['alpha']:g} | {item['narrow_success']}/{item['n']} | {item['wide_success']}/{item['n']} | {item['narrow_corridor_proxy']}/{item['n']} | {item['wide_speed_met']}/{item['n']} |")
    if {0.0, 0.5, 1.0}.issubset(set(plan["protocol"]["alphas"])):
        lines += ["", "## Preassigned arms by route", "", "| route | narrow α0 | narrow fixed α0.5 | wide α1 | wide fixed α0.5 |", "|---|---:|---:|---:|---:|"]
        for context in plan["contexts"]:
            route = context["route"]
            group = [row for row in rows if row["route"] == route]
            pick = lambda alpha, task: sum(row[f"{task}_success"] for row in group if row["alpha"] == alpha)
            n = sum(row["alpha"] == 0.0 for row in group)
            lines.append(f"| {route} | {pick(0.0, 'narrow')}/{n} | {pick(0.5, 'narrow')}/{n} | {pick(1.0, 'wide')}/{n} | {pick(0.5, 'wide')}/{n} |")
    lines += ["", f"All {len(rows)} outcomes and failure reasons are in [task_scores.json](task_scores.json).", ""]
    (root / "analysis/TASK_SCORES.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
