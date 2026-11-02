"""Exploratory rescore of completed U170 physical-domain trajectories."""
import json
import math
import sys
from pathlib import Path
from shutil import copyfile

import numpy as np


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: rescore_physical_domain.py DOMAIN_RUN OUTPUT_DIR")
    root, output = map(Path, sys.argv[1:])
    output.mkdir(parents=True, exist_ok=True)
    xy_dir = output / "xy"
    xy_dir.mkdir(exist_ok=True)
    contract = json.loads((Path(__file__).resolve().parents[3] / "docs/results/sttw-tolerance-20261101/task_preference_holdout/holdout_comparison.json").read_text())["contract"]
    rows = []
    domains = []
    for domain in range(4):
        source = root / f"domain{domain}"
        binding = json.loads((source / "binding/plan.json").read_text())["selected_domain"]
        params = binding["parameters"]
        domains.append(dict(domain=domain, id=binding["domain_id"], **params))
        original = json.loads((source / "evaluation/analysis/results.json").read_text())
        if len(original) != 20:
            raise ValueError("expected 20 complete frozen trajectories per physical domain")
        for image in sorted((source / "evaluation/analysis").glob("*xy.png")):
            copyfile(image, xy_dir / f"domain{domain}_{image.name}")
        for row in original:
            with np.load(row["diagnostic_task_file"]) as trace:
                cross = np.asarray(trace["original_cross_track_m"])
            n = row["active_lower_ticks"]
            arrival = row["endpoint"]["first_arrival_s"]
            before_goal = min(n, int(math.ceil(arrival / .005))) if arrival is not None else n
            max_cross = float(np.max(np.abs(cross[:before_goal])))
            tail = float(np.sqrt(np.mean(np.square(cross[1600:2000])))) if n == 2000 else None
            turn_speed = row["actual_turn_speed"]["mean_m_s"]
            common = arrival is not None and row["endpoint"]["physical_outcome"] == "survived" and tail is not None and tail <= contract["post_goal_last2s_original_cross_rmse_max_m"]
            narrow = common and max_cross + contract["buffer_radius_m"] <= contract["narrow"]["corridor_half_width_m"]
            wide = common and max_cross + contract["buffer_radius_m"] <= contract["wide"]["corridor_half_width_m"] and turn_speed >= contract["wide"]["speed_min_ratio"] * row["speed"]
            rows.append(dict(domain=domain, route=row["route"], bank=row["bank_index"], alpha=row["alpha"],
                             original_goal_arrived=arrival is not None, survived=row["endpoint"]["physical_outcome"] == "survived",
                             narrow=bool(narrow), wide=bool(wide), turn_speed_m_s=float(turn_speed),
                             max_original_cross_m=max_cross, post_original_cross_rmse_m=tail,
                             raw_file=row["raw_file"], diagnostic_file=row["diagnostic_task_file"]))
    summary = []
    for alpha in (0, .25, .5, .75, 1):
        part = [r for r in rows if r["alpha"] == alpha]
        summary.append(dict(alpha=alpha, n=len(part), goal=sum(r["original_goal_arrived"] for r in part),
                            survived=sum(r["survived"] for r in part), narrow=sum(r["narrow"] for r in part),
                            wide=sum(r["wide"] for r in part), mean_turn_speed_m_s=float(np.mean([r["turn_speed_m_s"] for r in part])),
                            mean_max_original_cross_m=float(np.mean([r["max_original_cross_m"] for r in part]))))
    result = dict(data_role="Exploratory post hoc task rescore of a completed independently sampled physical-domain TEST; original test only predeclared goal/survival", contract=contract, domains=domains, summary=summary, rows=rows)
    (output / "domain_task_rescore.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# U170 physical-domain task rescore", "", "## Actual/reference XY before scores", "",
             "Each panel overlays the original reference and all five executed alpha trajectories. The 16 PNG files are copied from the archived source analysis.", "",
             "| Domain | Low single, bank0 | High S, bank0 | Low single, bank6 | High S, bank6 |", "|---|---|---|---|---|"]
    for domain in range(4):
        cells = [f"[XY](xy/domain{domain}_bank{bank}_{route}_zero_fixed_candidate_xy.png)" for bank, route in ((0, "low_single_left"), (0, "high_S_left"), (6, "low_single_left"), (6, "high_S_left"))]
        lines.append(f"| test_{domain:02d} | " + " | ".join(cells) + " |")
    lines += ["",
             "This uses the same task score as the later task-combination panel on already completed trajectories. The four physical domains were independent TEST draws for the original arrival/survival study, but this task score was applied after inspecting that study and is exploratory, not a fresh confirmatory domain test.", "",
             "| Domain | Wheel friction | Chassis mass scale |", "|---|---:|---:|"]
    lines += [f"| {d['id']} | {d['wheel_sliding_friction']:.3f} | {d['chassis_mass_scale']:.3f} |" for d in domains]
    lines += ["", "| Alpha | Goal | Survived | Narrow | Wide | Mean turn speed (m/s) | Mean maximum original cross (m) |", "|---:|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r['alpha']:g} | {r['goal']}/{r['n']} | {r['survived']}/{r['n']} | {r['narrow']}/{r['n']} | {r['wide']}/{r['n']} | {r['mean_turn_speed_m_s']:.3f} | {r['mean_max_original_cross_m']:.3f} |" for r in summary]
    lines += ["", "The [JSON](domain_task_rescore.json) contains all 80 paired cases, physical-domain parameters, raw data paths and continuous outcomes.", ""]
    (output / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
