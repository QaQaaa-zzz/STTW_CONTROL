"""Summarize paired H75/U170 task scores on the frozen new combinations."""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare_holdout_middles.py H75_RUN U170_RUN OUTPUT_DIR")
    h75, u170, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    runs = {"H75": h75, "U170": u170}
    data = {model: json.loads((run / "analysis/task_scores.json").read_text()) for model, run in runs.items()}
    key = lambda row: (row["route"], row["bank"], row["alpha"])
    hrows, urows = data["H75"]["rows"], data["U170"]["rows"]
    if data["H75"]["contract"] != data["U170"]["contract"] or len(hrows) != 40 or len(urows) != 40 or {key(row) for row in hrows} != {key(row) for row in urows}:
        raise ValueError("the frozen H75/U170 panels must share the same 40 conditions and task contract")
    summary = []
    by_route = []
    for model, rows in (("H75", hrows), ("U170", urows)):
        for alpha in (0, .25, .5, .75, 1):
            group = [row for row in rows if row["alpha"] == alpha]
            summary.append(dict(model=model, alpha=alpha, n=len(group),
                                arrived=sum(row["original_goal_arrived"] for row in group),
                                survived=sum(row["physical_outcome"] == "survived" for row in group),
                                narrow=sum(row["narrow_success"] for row in group),
                                wide=sum(row["wide_success"] for row in group),
                                mean_turn_speed_m_s=sum(row["turn_speed_mean_m_s"] for row in group) / len(group),
                                mean_pre_goal_max_cross_m=sum(row["pre_goal_max_abs_original_cross_track_m"] for row in group) / len(group)))
        for route in sorted({row["route"] for row in rows}):
            for alpha in (0, .25, .5, .75, 1):
                group = [row for row in rows if row["route"] == route and row["alpha"] == alpha]
                by_route.append(dict(model=model, route=route, alpha=alpha, n=len(group),
                                     narrow=sum(row["narrow_success"] for row in group),
                                     wide=sum(row["wide_success"] for row in group),
                                     mean_turn_speed_m_s=sum(row["turn_speed_mean_m_s"] for row in group) / len(group),
                                     mean_pre_goal_max_cross_m=sum(row["pre_goal_max_abs_original_cross_track_m"] for row in group) / len(group)))
    result = dict(data_role="Prospective exact route/bank-combination check with previously DEV-selected task thresholds",
                  contract=data["H75"]["contract"], sources={model: str(run) for model, run in runs.items()},
                  summary=summary, by_route=by_route, rows={model: rows for model, rows in (("H75", hrows), ("U170", urows))})
    (output / "holdout_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# H75 and U170 on new task combinations", "",
             "Same local200 lower, 0.8 wheel friction, 10 Hz geometric reconnection, four newly declared low/high single/S routes, prepared banks 2/5 and all five fixed alpha settings. The relaxed score was selected before these exact route/bank outcomes. This checks new combinations, not unseen training distribution or physical-domain randomization.", "",
             "| Model | Alpha | Original goal | H10 survival | Narrow | Wide | Mean turn speed m/s | Mean pre-goal max original cross-track m |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['model']} | {row['alpha']:g} | {row['arrived']}/{row['n']} | {row['survived']}/{row['n']} | {row['narrow']}/{row['n']} | {row['wide']}/{row['n']} | {row['mean_turn_speed_m_s']:.3f} | {row['mean_pre_goal_max_cross_m']:.3f} |")
    lines += ["", "## By route", "", "| Model | Route | Alpha | Narrow | Wide | Turn speed m/s | Max cross-track m |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in by_route:
        lines.append(f"| {row['model']} | {row['route']} | {row['alpha']:g} | {row['narrow']}/{row['n']} | {row['wide']}/{row['n']} | {row['mean_turn_speed_m_s']:.3f} | {row['mean_pre_goal_max_cross_m']:.3f} |")
    lines += ["", "Every underlying case is in [holdout_comparison.json](holdout_comparison.json). Actual/reference XY panels for every route, bank and model are in the source runs and portable package.", ""]
    (output / "HOLDOUT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
