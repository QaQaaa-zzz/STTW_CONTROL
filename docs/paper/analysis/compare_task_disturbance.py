"""Summarize the frozen paired signed-pulse task extension."""
import json
import sys
from pathlib import Path
from shutil import copyfile


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: compare_task_disturbance.py U170_RUN RULE_RUN OUTPUT_DIR")
    uroot, rroot, output = map(Path, sys.argv[1:])
    u = json.loads((uroot / "analysis/task_scores.json").read_text())
    r = json.loads((rroot / "analysis/task_scores.json").read_text())
    up = json.loads((uroot / "plan.json").read_text())
    rp = json.loads((rroot / "plan.json").read_text())
    event = lambda row: row["case_id"].rsplit("/", 1)[-1]
    key = lambda row: (row["route"], row["bank"], event(row))
    urows, rrows = u["rows"], r["rows"]
    contexts = {key(row) for row in rrows}
    if (len(urows) != 32 or len(rrows) != 16 or len(contexts) != 16 or
            u["contract"] != r["contract"] or
            any(up[k] != rp[k] for k in ("source", "initial_source", "config", "replanning", "lower_transfer")) or
            any({key(row) for row in urows if row["alpha"] == a} != contexts for a in (0, 1)) or
            any(not e.startswith("t2.8_F") for _, _, e in contexts)):
        raise ValueError("complete paired ±36 N task panel required")
    groups = [("Rule", rrows)] + [(f"U170 alpha={a:g}", [row for row in urows if row["alpha"] == a]) for a in (0, 1)]
    summary = []
    for label, rows in groups:
        for force in ("all", "F-36", "F+36"):
            sample = rows if force == "all" else [row for row in rows if event(row).endswith(force)]
            summary.append(dict(method=label, force=force, n=len(sample),
                                goal=sum(row["original_goal_arrived"] for row in sample),
                                survived=sum(row["physical_outcome"] == "survived" for row in sample),
                                narrow=sum(row["narrow_success"] for row in sample),
                                wide=sum(row["wide_success"] for row in sample),
                                mean_turn_speed_m_s=sum(row["turn_speed_mean_m_s"] for row in sample) / len(sample),
                                mean_max_cross_m=sum(row["pre_goal_max_abs_original_cross_track_m"] for row in sample) / len(sample)))
    rule = {key(row): row for row in rrows}
    paired = {}
    for task, alpha in (("narrow", 0), ("wide", 1)):
        rows = [row for row in urows if row["alpha"] == alpha]
        paired[task] = dict(u170_only=sum(row[f"{task}_success"] and not rule[key(row)][f"{task}_success"] for row in rows),
                            rule_only=sum(rule[key(row)][f"{task}_success"] and not row[f"{task}_success"] for row in rows),
                            both=sum(row[f"{task}_success"] and rule[key(row)][f"{task}_success"] for row in rows),
                            neither=sum(not row[f"{task}_success"] and not rule[key(row)][f"{task}_success"] for row in rows))
    output.mkdir(parents=True, exist_ok=True)
    galleries = {}
    for label, root in (("u170", uroot), ("rule", rroot)):
        gallery = output / "xy" / label
        gallery.mkdir(parents=True, exist_ok=True)
        for picture in sorted((root / "analysis").glob("*xy.png")):
            copyfile(picture, gallery / picture.name)
        pictures = sorted(gallery.glob("*xy.png"))
        if len(pictures) != 16:
            raise ValueError("all sixteen route/bank/force XY panels required for each method")
        galleries[label] = pictures
    result = dict(data_role="Exploratory disturbance extension designed after nominal outcomes were seen",
                  u170_run=str(uroot), rule_run=str(rroot), contract=u["contract"],
                  summary=summary, paired=paired, u170_rows=urows, rule_rows=rrows)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Signed-pulse task extension", "", "## Actual/reference XY first", "",
             "All four original new routes, banks 2/5 and both pulse signs are retained. Each U170 panel overlays alpha 0 and 1; the matched rule panel plots the same original path.", ""]
    for label, pictures in galleries.items():
        lines += [f"### {label}", ""] + [f"- [{p.stem}](xy/{label}/{p.name})" for p in pictures] + [""]
    lines += ["Signed 36 N world-fixed force pulses begin at 2.8 s and last 0.15 s. U170 and the development-frozen rule use identical four routes, two prepared banks, local200 lower controller, friction 0.8, H10 and original-path task score. This extension was specified after the nominal result and is exploratory.", "",
              "| Method | Pulse | Goal | H10 | Narrow | Wide | Mean turn speed (m/s) | Mean max original cross (m) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {row['method']} | {row['force']} | {row['goal']}/{row['n']} | {row['survived']}/{row['n']} | {row['narrow']}/{row['n']} | {row['wide']}/{row['n']} | {row['mean_turn_speed_m_s']:.3f} | {row['mean_max_cross_m']:.3f} |" for row in summary]
    lines += ["", "Paired task endpoint versus rule: " + "; ".join(f"{task}: U170-only {d['u170_only']}, rule-only {d['rule_only']}, both {d['both']}, neither {d['neither']}" for task, d in paired.items()) + ".", "",
              "All individual continuous margins, arrival and failure labels are in [comparison.json](comparison.json).", ""]
    (output / "COMPARISON.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
