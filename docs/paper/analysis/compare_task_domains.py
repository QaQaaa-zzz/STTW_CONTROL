"""Compare task scores across two existing physical-domain models."""
import json
import sys
from pathlib import Path
from shutil import copyfile


def main():
    if len(sys.argv) != 6:
        raise SystemExit("usage: compare_task_domains.py RULE_D0 U170_D0 RULE_D3 U170_D3 OUTPUT_DIR")
    paths = list(map(Path, sys.argv[1:5]))
    output = Path(sys.argv[5])
    domain_source = json.loads((Path(__file__).resolve().parents[2] / "results/sttw-tolerance-20261101/physical_domain_rescore/domain_task_rescore.json").read_text())
    domain_parameters = {row["domain"]: row for row in domain_source["domains"]}
    rows = []
    summaries = []
    pairs = []
    galleries = []
    contract = None
    for domain, roots in ((0, paths[:2]), (3, paths[2:])):
        rule_root, uroot = roots
        rp, up = (json.loads((root / "plan.json").read_text()) for root in roots)
        r, u = (json.loads((root / "analysis/task_scores.json").read_text()) for root in roots)
        if (len(r["rows"]) != 8 or len(u["rows"]) != 16 or r["contract"] != u["contract"] or
                any(rp[k] != up[k] for k in ("source", "initial_source", "config", "replanning", "lower_transfer"))):
            raise ValueError("complete same-domain rule/U170 task panel required")
        contract = r["contract"]
        key = lambda row: (row["route"], row["bank"])
        contexts = {key(row) for row in r["rows"]}
        if len(contexts) != 8 or any({key(row) for row in u["rows"] if row["alpha"] == alpha} != contexts for alpha in (0, 1)):
            raise ValueError("same route and prepared-bank contexts required")
        rule = {key(row): row for row in r["rows"]}
        for method, sample in (("Rule", r["rows"]), ("U170 alpha=0", [x for x in u["rows"] if x["alpha"] == 0]), ("U170 alpha=1", [x for x in u["rows"] if x["alpha"] == 1])):
            summaries.append(dict(domain=domain, method=method, n=len(sample),
                                  goal=sum(x["original_goal_arrived"] for x in sample),
                                  survived=sum(x["physical_outcome"] == "survived" for x in sample),
                                  narrow=sum(x["narrow_success"] for x in sample),
                                  wide=sum(x["wide_success"] for x in sample),
                                  mean_turn_speed_m_s=sum(x["turn_speed_mean_m_s"] for x in sample) / len(sample),
                                  mean_max_cross_m=sum(x["pre_goal_max_abs_original_cross_track_m"] for x in sample) / len(sample)))
            rows.extend(dict(domain=domain, method=method, **x) for x in sample)
        for task, alpha in (("narrow", 0), ("wide", 1)):
            sample = [x for x in u["rows"] if x["alpha"] == alpha]
            pairs.append(dict(domain=domain, task=task,
                              u170_only=sum(x[f"{task}_success"] and not rule[key(x)][f"{task}_success"] for x in sample),
                              rule_only=sum(rule[key(x)][f"{task}_success"] and not x[f"{task}_success"] for x in sample),
                              both=sum(x[f"{task}_success"] and rule[key(x)][f"{task}_success"] for x in sample),
                              neither=sum(not x[f"{task}_success"] and not rule[key(x)][f"{task}_success"] for x in sample)))
        for label, root in (("rule", rule_root), ("u170", uroot)):
            gallery = output / "xy" / f"domain{domain}_{label}"
            gallery.mkdir(parents=True, exist_ok=True)
            for image in sorted((root / "analysis").glob("*xy.png")):
                copyfile(image, gallery / image.name)
            images = sorted(gallery.glob("*xy.png"))
            if len(images) != 8:
                raise ValueError("all eight original/actual XY panels required per domain and method")
            galleries.append((domain, label, images))
    result = dict(data_role="Exploratory task transfer to previously sampled physical domains selected by friction extrema after prior outcomes were opened",
                  domain_parameters=[domain_parameters[d] for d in (0, 3)], runs=[str(p) for p in paths],
                  contract=contract, summary=summaries, paired=pairs, rows=rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Physical-domain and prepared-state task transfer", "", "## Actual/reference XY first", "",
             "Two existing physical-domain models were chosen as the highest and lowest wheel-friction draws from a four-domain TEST set. Each domain uses its own complete prepared banks 0/6, four new nominal routes, and the same frozen task score.", ""]
    for domain, label, images in galleries:
        lines += [f"### Domain {domain}: {label}", ""] + [f"- [{image.stem}](xy/domain{domain}_{label}/{image.name})" for image in images] + [""]
    lines += ["| Domain | Wheel friction | Chassis mass scale | Method | Goal | H10 | Narrow | Wide | Mean turn speed (m/s) | Mean max original cross (m) |",
              "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        physical = domain_parameters[row["domain"]]
        lines.append(f"| {row['domain']} | {physical['wheel_sliding_friction']:.3f} | {physical['chassis_mass_scale']:.3f} | {row['method']} | {row['goal']}/{row['n']} | {row['survived']}/{row['n']} | {row['narrow']}/{row['n']} | {row['wide']}/{row['n']} | {row['mean_turn_speed_m_s']:.3f} | {row['mean_max_cross_m']:.3f} |")
    lines += ["", "Paired task endpoints versus rule:", ""]
    lines += [f"- Domain {x['domain']} {x['task']}: U170-only {x['u170_only']}, rule-only {x['rule_only']}, both {x['both']}, neither {x['neither']}." for x in pairs]
    lines += ["", "This is exploratory: the domain models existed and older outcomes had been viewed before this task score and route extension. All cases, including failures, are in [comparison.json](comparison.json); full raw NPZ trajectories and immutable source manifests remain in the linked run directories.", ""]
    (output / "COMPARISON.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
