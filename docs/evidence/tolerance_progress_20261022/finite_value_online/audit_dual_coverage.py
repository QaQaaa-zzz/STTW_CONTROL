"""Read existing lineage trajectories; no simulator, training, or relabel overwrite."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    root = Path(root)
    output = root / "runs/dual_coverage_audit_v1"
    output.mkdir(exist_ok=True)
    registry = json.loads((root / "learning/configs/frozen_middle_registry.json").read_text())
    identities = {}
    for alpha in (0, 1):
        record = registry["records"][f"STTW_MIDDLE_DUAL_COVERAGE_U200_A{alpha}"]
        for field in ("actor", "checkpoint"):
            assert sha(record[field]) == record[field + "_sha256"]
        identities[str(alpha)] = {key: record[key] for key in
                                 ("actor_sha256", "checkpoint_sha256")}
    early = {}
    for name in ("pulse_pilot_v1_attempt2", "boundary_focus_v1"):
        directory = root / "runs" / name
        manifest = json.loads((directory / "manifest.json").read_text())
        for endpoint, identity in manifest["checkpoint_identities"].items():
            for key, value in identities[endpoint[-1]].items():
                assert identity[key] == value
        cases = json.loads((directory / "results.json").read_text())["cases"]
        early[name] = {"n": len(cases), "manifest_sha256": sha(directory / "manifest.json"),
                       "results_sha256": sha(directory / "results.json"),
                       **{key: sum(bool(c[key]) for c in cases) for key in
                          ("success", "physical_failure", "domain_failure", "working_failure")}}
    revisit = root / "runs/dual_endpoint_tolerance_revisit_v1"
    revisit_cases = json.loads((revisit / "analysis.json").read_text())["cases"]
    for alpha in (0, 1):
        declaration = revisit / f"alpha{alpha}/declaration.json"
        policy = json.loads(declaration.read_text())["policies"][f"dual_alpha{alpha}"]
        for key, value in identities[str(alpha)].items():
            assert policy[key] == value
        cases = [c["native20"] for c in revisit_cases if c["endpoint"] == f"alpha{alpha}"]
        early[f"dual_revisit_alpha{alpha}"] = {
            "n": len(cases), "declaration_sha256": sha(declaration),
            "analysis_sha256": sha(revisit / "analysis.json"),
            **{key: sum(bool(c[key]) for c in cases) for key in
               ("task_recovered", "physical_failure", "domain_failure", "working_failure")}}
    directory = root / "runs/lineage_nominal_common_interface_v1"
    source = directory / "analysis_corrected_roles_v1/results.json"
    rows = json.loads(source.read_text())
    contexts = json.loads((directory / "plan.json").read_text())["contexts"]
    goals = {}
    for key, context in contexts.items():
        assert sha(context["path"]) == context["path_sha256"]
        with np.load(context["path"]) as p:
            goals[key] = np.array([np.interp(float(p["goal"]), p["s"], p["xy"][:, j])
                                   for j in range(2)])
    audited = []
    summary = defaultdict(lambda: {"n": 0, "arrived_0p5m": 0, "arrived_0p75m": 0,
                                   "arrived_1m": 0, "contact": 0, "fault": 0})
    for row in rows:
        assert sha(row["raw_file"]) == row["raw_sha256"]
        with np.load(row["raw_file"]) as p:
            active = p["active_tick"].astype(bool)
            assert active.sum() == 2000
            assert np.array_equal(p["physical_tick"][active], np.arange(2000))
            assert np.allclose(p["time"][active], np.arange(2000) * .005, atol=2e-6)
            contact = p["contact_failure"][active].astype(bool)
            fault = np.logical_or.reduce([p[k][active].astype(bool) for k in
                                          ("policy_fault", "lower_fault", "nonfinite")])
            assert not contact.any() and not fault.any()
            distance = np.linalg.norm(p["actual_xy"][active].astype(float) - goals[row["context"]], axis=1)
            minimum = float(distance.min())
            assert abs(minimum - row["minimum_upright_goal_distance_m"]) < 2e-5
            hits = {radius: np.flatnonzero(distance <= radius) for radius in (.5, .75, 1.)}
            assert bool(len(hits[.5])) == bool(row["goal_arrived"])
            result = {"case_id": row["case_id"], "context": row["context"], "role": row["role"],
                      "raw_file": row["raw_file"], "raw_sha256": row["raw_sha256"],
                      "minimum_goal_distance_m": minimum,
                      "first_arrival_s": {str(r): float(p["time"][active][ix[0]]) if len(ix) else None
                                          for r, ix in hits.items()}}
            audited.append(result)
            aggregate = summary[row["role"]]
            aggregate["n"] += 1
            for radius, label in ((.5, "0p5m"), (.75, "0p75m"), (1., "1m")):
                aggregate["arrived_" + label] += int(len(hits[radius]) > 0)
    receipt = {"scope": "existing traces only; zero new physics/training; AI artifact checked",
               "identity": identities, "early_report_counts": early,
               "lineage_source_sha256": sha(source), "lineage_summary": dict(summary),
               "lineage_rows": audited, "new_physics_ticks": 0,
               "limitations": ["Nominal common-interface weight transfer, not native architecture ablation",
                               "No independent training repeats; no disturbance in lineage120",
                               "Early physical_failure label is not silently equated to modern contact-only label",
                               "Domain-stopped tails remain unknown; original outcome files immutable"]}
    (output / "raw_readback.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"early": early, "lineage": dict(summary)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
