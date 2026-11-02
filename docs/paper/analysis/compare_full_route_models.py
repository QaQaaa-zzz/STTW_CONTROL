"""Compare U170, H75 and W400 on the four declared task routes."""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 7:
        raise SystemExit("usage: compare_full_route_models.py U170_RUN H75_FIRST H75_EXTENSION W400_FIRST W400_EXTENSION OUTPUT_DIR")
    u170, h75_first, h75_extension, w400_first, w400_extension, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads((w400_first / "plan.json").read_text())["protocol"]["task_contract"]
    original = json.loads((u170 / "analysis/task_preference_diagnostic.json").read_text())["rows"]
    h75 = [*json.loads((h75_first / "analysis/task_preference_diagnostic.json").read_text())["rows"],
           *json.loads((h75_extension / "analysis/task_preference_diagnostic.json").read_text())["rows"]]
    w400 = [*json.loads((w400_first / "analysis/task_preference_diagnostic.json").read_text())["rows"],
            *json.loads((w400_extension / "analysis/task_preference_diagnostic.json").read_text())["rows"]]
    key = lambda row: (row["route"], row["bank"], row["alpha"])
    if any(len(group) != 40 or {key(row) for row in group} != {key(row) for row in original} for group in (original, h75, w400)):
        raise ValueError("all middle panels require the same 40 route/bank/alpha conditions")
    rows = []
    for model, group in (("U170", original), ("H75", h75), ("W400", w400)):
        for row in group:
            common = (row["original_goal_arrived"] and row["physical_outcome"] == "survived"
                      and row["last_2s_original_cross_rmse_m"] <= contract["post_goal_last2s_original_cross_rmse_max_m"])
            rows.append(dict(model=model, route=row["route"], bank=row["bank"], alpha=row["alpha"],
                             arrived=row["original_goal_arrived"], survived=row["physical_outcome"] == "survived",
                             narrow=common and row["pre_goal_max_abs_original_cross_track_m"] + contract["buffer_radius_m"] <= contract["narrow"]["corridor_half_width_m"],
                             wide=common and row["pre_goal_max_abs_original_cross_track_m"] + contract["buffer_radius_m"] <= contract["wide"]["corridor_half_width_m"]
                             and row["turn_speed_mean_m_s"] >= contract["wide"]["speed_min_ratio"] * row["requested_speed_m_s"],
                             speed_m_s=row["turn_speed_mean_m_s"], max_cross_m=row["pre_goal_max_abs_original_cross_track_m"],
                             last2_cross_rmse_m=row["last_2s_original_cross_rmse_m"]))
    summary = []
    for model in ("U170", "H75", "W400"):
        for route in sorted({row["route"] for row in rows}):
            for alpha in (0, .25, .5, .75, 1):
                group = [row for row in rows if (row["model"], row["route"], row["alpha"]) == (model, route, alpha)]
                summary.append(dict(model=model, route=route, alpha=alpha, n=len(group),
                                    arrived=sum(row["arrived"] for row in group), survived=sum(row["survived"] for row in group),
                                    narrow=sum(row["narrow"] for row in group), wide=sum(row["wide"] for row in group),
                                    mean_turn_speed_m_s=sum(row["speed_m_s"] for row in group) / len(group),
                                    mean_pre_goal_max_cross_m=sum(row["max_cross_m"] for row in group) / len(group)))
    record = dict(contract=contract, data_role="DEV model selection; thresholds chosen after inspecting U170 task40",
                  sources=dict(U170=str(u170), H75_first=str(h75_first), H75_extension=str(h75_extension),
                               W400_first=str(w400_first), W400_extension=str(w400_extension)), rows=rows, summary=summary)
    (output / "full_route_comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    lines = ["# Four-route frozen middle comparison", "", "Same local200 lower controller, friction 0.8, 10 Hz reconnection, prepared banks 1 and 3, five fixed alpha values and 10 s observation. These are model-selection development results; the relaxed task thresholds were set after inspecting U170's earlier task panel.", "", "| model | route | alpha | original goal | H10 survival | narrow | wide | mean turn speed m/s | mean max original cross-track m |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['model']} | {row['route']} | {row['alpha']:g} | {row['arrived']}/{row['n']} | {row['survived']}/{row['n']} | {row['narrow']}/{row['n']} | {row['wide']}/{row['n']} | {row['mean_turn_speed_m_s']:.3f} | {row['mean_pre_goal_max_cross_m']:.3f} |")
    lines += ["", "The machine-readable file contains every route, bank and alpha outcome. Actual/reference XY panels are retained in the source runs and the portable result package.", ""]
    (output / "FULL_ROUTES.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
