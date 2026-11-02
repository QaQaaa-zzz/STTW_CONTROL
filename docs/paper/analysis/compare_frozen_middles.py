"""Compare saved task-score traces for four frozen middle policies."""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 6:
        raise SystemExit("usage: compare_frozen_middles.py U170_RUN H75_RUN W400_RUN U325_RUN OUTPUT_DIR")
    labels = ("U170", "H75", "W400", "u325")
    runs = [Path(arg) for arg in sys.argv[1:5]]
    output = Path(sys.argv[5]);output.mkdir(parents=True, exist_ok=True)
    contract = json.loads((runs[2] / "plan.json").read_text())["protocol"]["task_contract"]
    routes = {context["route"] for context in json.loads((runs[2] / "plan.json").read_text())["contexts"]}
    summary = []
    case_rows = []
    for label, root in zip(labels, runs):
        data = json.loads((root / "analysis/task_preference_diagnostic.json").read_text())
        rows = [row for row in data["rows"] if row["route"] in routes]
        case_rows.extend(dict(model=label, **row) for row in rows)
        for alpha in (0, .25, .5, .75, 1):
            group = [row for row in rows if row["alpha"] == alpha]
            common = [row["original_goal_arrived"] and row["physical_outcome"] == "survived" and row["last_2s_original_cross_rmse_m"] is not None and row["last_2s_original_cross_rmse_m"] <= contract["post_goal_last2s_original_cross_rmse_max_m"] for row in group]
            narrow = sum(valid and row["pre_goal_max_abs_original_cross_track_m"] + contract["buffer_radius_m"] <= contract["narrow"]["corridor_half_width_m"] for valid, row in zip(common, group))
            wide = sum(valid and row["pre_goal_max_abs_original_cross_track_m"] + contract["buffer_radius_m"] <= contract["wide"]["corridor_half_width_m"] and row["turn_speed_mean_m_s"] is not None and row["turn_speed_mean_m_s"] >= contract["wide"]["speed_min_ratio"] * row["requested_speed_m_s"] for valid, row in zip(common, group))
            speeds = [row["turn_speed_mean_m_s"] for row in group if row["turn_speed_mean_m_s"] is not None]
            summary.append({"model": label, "alpha": alpha, "n": len(group), "arrived": sum(row["original_goal_arrived"] for row in group), "survived": sum(row["physical_outcome"] == "survived" for row in group), "narrow": narrow, "wide": wide, "mean_turn_speed_m_s": sum(speeds) / len(speeds) if speeds else None, "mean_pre_goal_max_cross_m": sum(row["pre_goal_max_abs_original_cross_track_m"] for row in group) / len(group)})
    matched = {(row["route"], row["bank"], row["alpha"]) for row in case_rows if row["model"] == labels[0]}
    if len(matched) != 20 or any({(row["route"], row["bank"], row["alpha"]) for row in case_rows if row["model"] == label} != matched for label in labels):
        raise ValueError("the four frozen middle panels must contain the same 20 route/bank/alpha conditions")
    record = {"contract": contract, "routes": sorted(routes), "runs": {label: str(root) for label, root in zip(labels, runs)}, "summary": summary, "case_rows": case_rows}
    (output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    lines = ["# Four frozen middle policies on matched conflict routes", "", "All policies use local200, wheel friction 0.8, 10 Hz reconnection and the same prepared banks1/3. U170 traces were already inspected when the relaxed thresholds were chosen; this is model-selection development evidence.", "", "| model | alpha | original goal | H10 survival | narrow task | wide task | mean turn speed m/s | mean pre-goal maximum cross-track m |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        speed = f"{row['mean_turn_speed_m_s']:.3f}" if row["mean_turn_speed_m_s"] is not None else "NA"
        lines.append(f"| {row['model']} | {row['alpha']:g} | {row['arrived']}/{row['n']} | {row['survived']}/{row['n']} | {row['narrow']}/{row['n']} | {row['wide']}/{row['n']} | {speed} | {row['mean_pre_goal_max_cross_m']:.3f} |")
    lines += ["", "Task success uses the frozen relaxed buffered-anchor thresholds in [comparison.json](comparison.json). The saved source runs contain actual/reference XY panels and individual case outcomes. Prepared banks are paired initial states, not independent training seeds.", ""]
    (output / "COMPARISON.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
