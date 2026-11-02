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
    lines = ["# Frozen task-preference score", "", "The corridor uses a 0.5 m circular buffer around recorded vehicle XY. This is an anchor-disk proxy, not an exact swept-mesh or real-wall test. The original goal, H10 survival and 8–10 s original-path tracking are required for both tasks. Widths and speed threshold were fixed from the earlier DEV2 panel before this run.", "", "## Actual/reference XY", ""]
    for bank in plan["protocol"]["bank_indices"]:
        for context in plan["contexts"]:
            name = f"bank{bank}_{context['route']}_zero_fixed_candidate_xy.png"
            lines.append(f"![bank{bank} {context['route']} actual/reference XY]({name})")
    lines += ["", "## Outcomes by fixed alpha", "", "| alpha | narrow task | wide task | narrow corridor proxy | wide speed threshold |", "|---:|---:|---:|---:|---:|"]
    for item in summary:
        lines.append(f"| {item['alpha']:g} | {item['narrow_success']}/{item['n']} | {item['wide_success']}/{item['n']} | {item['narrow_corridor_proxy']}/{item['n']} | {item['wide_speed_met']}/{item['n']} |")
    lines += ["", "## Preassigned arms by route", "", "| route | narrow α0 | narrow fixed α0.5 | wide α1 | wide fixed α0.5 |", "|---|---:|---:|---:|---:|"]
    for context in plan["contexts"]:
        route = context["route"]
        group = [row for row in rows if row["route"] == route]
        pick = lambda alpha, task: sum(row[f"{task}_success"] for row in group if row["alpha"] == alpha)
        n = len(plan["protocol"]["bank_indices"])
        lines.append(f"| {route} | {pick(0.0, 'narrow')}/{n} | {pick(0.5, 'narrow')}/{n} | {pick(1.0, 'wide')}/{n} | {pick(0.5, 'wide')}/{n} |")
    lines += ["", "All 40 paired outcomes and failure reasons are in [task_scores.json](task_scores.json). The preassigned comparison is narrow α0 versus fixed α0.5, and wide α1 versus fixed α0.5; the other levels diagnose the preference response. No tuned conventional rule has been tested in this panel.", ""]
    (root / "analysis/TASK_SCORES.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
