"""Select two task-specific pure-pursuit speed caps on frozen DEV runs."""
import json
import sys
from pathlib import Path
from shutil import copyfile


def main():
    if len(sys.argv) != 6:
        raise SystemExit("usage: select_traditional_rule.py DEV1 DEV2 DEV3 DEV4 OUTPUT_DIR")
    roots = list(map(Path, sys.argv[1:5]))
    output = Path(sys.argv[5])
    plans = [json.loads((root / "plan.json").read_text()) for root in roots]
    scores = [json.loads((root / "analysis/task_scores.json").read_text()) for root in roots]
    keys = [{(r["route"], r["bank"]) for r in s["rows"]} for s in scores]
    if any(len(s["rows"]) != 8 for s in scores) or any(s["contract"] != scores[0]["contract"] for s in scores) or any(k != keys[0] for k in keys):
        raise ValueError("four complete matched DEV rule screens are required")
    summary = []
    for root, plan, score in zip(roots, plans, scores):
        rows = score["rows"]
        summary.append(dict(source=str(root), rule=plan["protocol"]["traditional_rule"],
                            narrow=sum(r["narrow_success"] for r in rows), wide=sum(r["wide_success"] for r in rows),
                            goal=sum(r["original_goal_arrived"] for r in rows),
                            survived=sum(r["physical_outcome"] == "survived" for r in rows),
                            mean_turn_speed_m_s=sum(r["turn_speed_mean_m_s"] for r in rows) / 8,
                            mean_max_cross_m=sum(r["pre_goal_max_abs_original_cross_track_m"] for r in rows) / 8))
    narrow = min(summary, key=lambda r: (-r["narrow"], r["mean_max_cross_m"], r["rule"]["id"]))
    wide = min(summary, key=lambda r: (-r["wide"], -r["mean_turn_speed_m_s"], r["rule"]["id"]))
    output.mkdir(parents=True, exist_ok=True)
    gallery = output / "xy"
    gallery.mkdir(exist_ok=True)
    xy_lines = []
    for root, row in zip(roots, summary):
        images = sorted((root / "analysis").glob("*xy.png"))
        if len(images) != 8:
            raise ValueError("eight actual/original XY panels required per DEV rule")
        slug = row["rule"]["id"]
        folder = gallery / slug
        folder.mkdir(exist_ok=True)
        xy_lines.append(f"### {slug}")
        xy_lines.append("")
        for image in images:
            copyfile(image, folder / image.name)
            xy_lines.append(f"- [{image.stem}](xy/{slug}/{image.name})")
        xy_lines.append("")
    result = dict(data_role="Rule hyperparameter selection on four previously inspected DEV routes and prepared banks1/3; exact holdout outcomes not used", contract=scores[0]["contract"], candidates=summary, selected={"narrow": narrow["rule"], "wide": wide["rule"]}, tie_break="narrow: lower mean maximum original cross; wide: higher mean actual turn speed")
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Traditional rule DEV selection", "", "## Actual/reference XY first", "", "Every declared rule, route and prepared bank has an actual/original XY panel.", "", *xy_lines,
             "Pure pursuit geometry follower with a 6 m preview maximum-curvature speed cap; the same lower stack and task score as the middle policies. This rule does not query a learned middle Actor. Alpha 0.5 in the native run is only a reward-bookkeeping field.", "", "| Lateral acceleration cap (m/s²) | Goal | H10 | Narrow | Wide | Mean turn speed (m/s) | Mean max original cross (m) |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['rule']['lateral_acceleration_limit_m_s2']:.1f} | {r['goal']}/8 | {r['survived']}/8 | {r['narrow']}/8 | {r['wide']}/8 | {r['mean_turn_speed_m_s']:.3f} | {r['mean_max_cross_m']:.3f} |")
    lines += ["", f"Selected narrow: {narrow['rule']['id']}; selected wide: {wide['rule']['id']}. Selection used DEV only; holdout testing follows after these parameters are frozen.", ""]
    (output / "SELECTION.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
