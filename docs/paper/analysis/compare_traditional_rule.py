"""Compare a DEV-selected geometric rule with frozen U170 task preferences."""
import json
import sys
from pathlib import Path
from shutil import copyfile


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare_traditional_rule.py MIDDLE_COMPARISON_JSON RULE_RUN OUTPUT_DIR")
    middle, rule_root, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    learned = json.loads(middle.read_text())
    rule = json.loads((rule_root / "analysis/task_scores.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    urows = learned["rows"]["U170"]
    rrows = rule["rows"]
    key = lambda row: (row["route"], row["bank"])
    contexts = {key(row) for row in rrows}
    if len(rrows) != 8 or len(contexts) != 8 or rule["contract"] != learned["contract"] or selection["selected"]["narrow"] != selection["selected"]["wide"]:
        raise ValueError("one complete matched frozen-rule task panel required")
    if contexts != {key(row) for row in urows if row["alpha"] == 0}:
        raise ValueError("rule and learned panels use different route/initial-state contexts")
    plan = json.loads((rule_root / "plan.json").read_text())
    if plan["protocol"]["traditional_rule"] != selection["selected"]["narrow"]:
        raise ValueError("executed rule differs from DEV-selected rule")
    summary = []
    groups = [("Rule", rrows)] + [(f"U170 alpha={alpha:g}", [row for row in urows if row["alpha"] == alpha]) for alpha in (0, .5, 1)]
    for label, rows in groups:
        summary.append(dict(method=label, n=len(rows), goal=sum(row["original_goal_arrived"] for row in rows),
                            survived=sum(row["physical_outcome"] == "survived" for row in rows),
                            narrow=sum(row["narrow_success"] for row in rows), wide=sum(row["wide_success"] for row in rows),
                            mean_turn_speed_m_s=sum(row["turn_speed_mean_m_s"] for row in rows) / len(rows),
                            mean_max_cross_m=sum(row["pre_goal_max_abs_original_cross_track_m"] for row in rows) / len(rows)))
    by_rule = {key(row): row for row in rrows}
    paired = {}
    for task, alpha in (("narrow", 0), ("wide", 1)):
        rows = [row for row in urows if row["alpha"] == alpha]
        paired[task] = dict(learned_only=sum(row[f"{task}_success"] and not by_rule[key(row)][f"{task}_success"] for row in rows),
                            rule_only=sum(by_rule[key(row)][f"{task}_success"] and not row[f"{task}_success"] for row in rows),
                            both=sum(row[f"{task}_success"] and by_rule[key(row)][f"{task}_success"] for row in rows),
                            neither=sum(not row[f"{task}_success"] and not by_rule[key(row)][f"{task}_success"] for row in rows))
    output.mkdir(parents=True, exist_ok=True)
    gallery = output / "xy"
    gallery.mkdir(exist_ok=True)
    for image in sorted((rule_root / "analysis").glob("*xy.png")):
        copyfile(image, gallery / image.name)
    if len(list(gallery.glob("*xy.png"))) != 8:
        raise ValueError("eight original/actual XY rule panels required")
    result = dict(data_role="DEV-selected rule on already opened exact-combination panel; later matched comparison, not blinded independent TEST",
                  rule=plan["protocol"]["traditional_rule"], rule_run=str(rule_root), middle_comparison=str(middle),
                  contract=rule["contract"], summary=summary, paired=paired, rule_rows=rrows)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# DEV-tuned geometric rule on the new task combinations", "", "## Actual/reference XY first", "",
             "Eight frozen-rule panels cover all four new routes and prepared banks2/5; each plots actual XY against the original route. The five-alpha U170/H75 matched panels are in the linked middle result package.", ""]
    lines += [f"- [{p.stem}](xy/{p.name})" for p in sorted(gallery.glob("*xy.png"))]
    lines += ["", "The pure-pursuit curvature-cap rule uses no learned middle output. Its 3.5 m/s² lateral-acceleration cap and 6 m preview were selected on earlier DEV conditions, before executing these eight rule cases. The middle new-combination outcomes were already opened when this comparator was constructed.", "",
              "| Method | Goal | H10 | Narrow | Wide | Mean turn speed (m/s) | Mean max original cross (m) |", "|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['method']} | {r['goal']}/{r['n']} | {r['survived']}/{r['n']} | {r['narrow']}/{r['n']} | {r['wide']}/{r['n']} | {r['mean_turn_speed_m_s']:.3f} | {r['mean_max_cross_m']:.3f} |" for r in summary]
    lines += ["", "Paired U170 endpoint versus rule: " + "; ".join(f"{task}: U170-only {d['learned_only']}, rule-only {d['rule_only']}, both {d['both']}, neither {d['neither']}" for task, d in paired.items()) + ".", "", "All eight individual rule scores, including failures and continuous margins, are in [comparison.json](comparison.json). The matched five-alpha middle cases are in [the separate full panel](../task_preference_holdout/HOLDOUT.md).", ""]
    (output / "COMPARISON.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
