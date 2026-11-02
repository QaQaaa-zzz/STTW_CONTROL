"""Summarize the declared paired 90-degree turn panel and retain all XY plots."""
import json
import sys
from pathlib import Path
from shutil import copyfile


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare_right_angle.py U170_RUN RULE_RUN OUTPUT_DIR")
    uroot, rroot, output = map(Path, sys.argv[1:])
    up, rp = (json.loads((root / "plan.json").read_text()) for root in (uroot, rroot))
    u, r = (json.loads((root / "analysis/task_scores.json").read_text()) for root in (uroot, rroot))
    if (len(u["rows"]) != 40 or len(r["rows"]) != 8 or u["contract"] != r["contract"]
            or any(up[k] != rp[k] for k in ("source", "initial_source", "config", "replanning", "lower_transfer"))
            or up["contexts"] != rp["contexts"]):
        raise ValueError("complete matched 90-degree turn panel required")
    key = lambda row: (row["route"], row["bank"])
    contexts = {key(row) for row in r["rows"]}
    if len(contexts) != 8 or any({key(row) for row in u["rows"] if row["alpha"] == alpha} != contexts
                                 for alpha in (0, .25, .5, .75, 1)):
        raise ValueError("all route, bank and preference pairs required")
    summary = []
    for label, rows in [("Rule", r["rows"])] + [(f"U170 alpha={alpha:g}", [x for x in u["rows"] if x["alpha"] == alpha])
                                                  for alpha in (0, .25, .5, .75, 1)]:
        summary.append(dict(method=label, n=len(rows), goal=sum(x["original_goal_arrived"] for x in rows),
                            survived=sum(x["physical_outcome"] == "survived" for x in rows),
                            narrow=sum(x["narrow_success"] for x in rows), wide=sum(x["wide_success"] for x in rows),
                            mean_turn_speed_m_s=sum(x["turn_speed_mean_m_s"] for x in rows) / len(rows),
                            mean_max_cross_m=sum(x["pre_goal_max_abs_original_cross_track_m"] for x in rows) / len(rows)))
    output.mkdir(parents=True, exist_ok=True)
    galleries = []
    for label, root in (("u170", uroot), ("rule", rroot)):
        gallery = output / "xy" / label
        gallery.mkdir(parents=True, exist_ok=True)
        for image in sorted((root / "analysis").glob("*xy.png")):
            copyfile(image, gallery / image.name)
        images = sorted(gallery.glob("*xy.png"))
        if len(images) != 8:
            raise ValueError("all eight route/bank XY panels required per method")
        galleries.append((label, images))
    result = dict(data_role="Exploratory 90-degree heading-change panel selected after earlier results; finite-radius smooth corners",
                  u170_run=str(uroot), rule_run=str(rroot), contract=u["contract"], summary=summary,
                  u170_rows=u["rows"], rule_rows=r["rows"])
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Matched 90-degree left/right turn experiment", "", "## Actual/reference XY first", "",
             "Four smooth finite-radius routes change heading by 90 degrees: left/right at low request 2.0 m/s, radius 1.6 m and high request 3.05 m/s, radius 3.0 m. This is not a zero-radius mathematical corner. Prepared banks 2/5, friction 0.8, 10 Hz causal path reconnection, original-goal task score, H10 and frozen controllers are shared. All 48 episodes are retained.", ""]
    for label, images in galleries:
        lines += [f"### {label}", ""] + [f"- [{image.stem}](xy/{label}/{image.name})" for image in images] + [""]
    lines += ["| Method | Goal | H10 | Narrow | Wide | Turn speed (m/s) | Max original cross (m) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {x['method']} | {x['goal']}/{x['n']} | {x['survived']}/{x['n']} | {x['narrow']}/{x['n']} | {x['wide']}/{x['n']} | {x['mean_turn_speed_m_s']:.3f} | {x['mean_max_cross_m']:.3f} |" for x in summary]
    lines += ["", "Exact case scores and raw run locations are in [comparison.json](comparison.json). This panel was requested after inspecting nominal and signed-pulse outcomes and is exploratory.", ""]
    (output / "COMPARISON.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
