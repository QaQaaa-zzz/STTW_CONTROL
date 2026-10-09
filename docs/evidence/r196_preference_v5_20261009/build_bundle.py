#!/usr/bin/env python3
"""Build the compact, reviewable Preference V5 evidence bundle from immutable runs."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
STAGE1 = REPO / "runs/preference_v5_20261009"
STAGE2 = REPO / "runs/preference_v5_stage2_override_20261009"
CASES = ["straight_hold", "speed_changes", "gentle_positive", "gentle_negative", "steer_reversal", "fast_turn"]
METHODS = ["B0", "alpha0", "alpha1"]
TRACE_FIELDS = [
    "time", "limited_command", "target", "governed", "latent_z", "actual_forward_speed",
    "actual_delta", "phi", "phi_dot", "e_psi_unwrapped", "yaw_rate", "actual_xy",
    "reference_xy", "along", "lateral", "wheel_speed_proxy", "slip_proxy", "u_nom",
    "u_goal", "requested_residual", "applied_residual", "lower_action", "final_command",
    "offsets", "physical_failure", "scored_tick_reward", "checkpoint_update", "alpha",
]


def copy(source: Path, relative: str) -> None:
    destination = OUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def compact_npz(source_root: Path, update: int, cases: list[str], destination: str) -> None:
    payload = {}
    for case in cases:
        for method in METHODS:
            source = source_root / f"evaluation{update}/{case}/{method}.npz"
            with np.load(source, allow_pickle=False) as data:
                fields = TRACE_FIELDS + [k for k in data.files if k.startswith("scored_cost_")]
                for field in dict.fromkeys(fields):
                    if field in data:
                        payload[f"{case}__{method}__{field}"] = data[field]
    np.savez_compressed(OUT / "data" / destination, **payload)


def write_stage1_summary() -> None:
    fields = ["update", "turn_sign", "method", "complete5s", "physical_failure", "peak_roll_rad",
              "speed_rmse_m_s", "steer_rmse_rad", "pre_turn_speed_mean_m_s", "mean_actual_speed_drop_m_s",
              "drop_ge_0p2_longest_s", "pair_gate_passed"]
    with (OUT / "data/stage1_pair_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for update in (20, 40):
            gate = json.loads((STAGE1 / f"evaluation{update}/stage1_gate.json").read_text())
            for sign, result in gate["signs"].items():
                for method, row in result["metrics"].items():
                    writer.writerow(dict(update=update, turn_sign=sign, method=method,
                        complete5s=row["complete5s"], physical_failure=row["physical_failure"],
                        peak_roll_rad=row["peak_roll"], speed_rmse_m_s=row["speed_rmse"],
                        steer_rmse_rad=row["steer_rmse"], pre_turn_speed_mean_m_s=row["pre_turn_speed_mean"],
                        mean_actual_speed_drop_m_s=row["mean_drop"],
                        drop_ge_0p2_longest_s=row["drop_ge_0p2_longest_seconds"], pair_gate_passed=result["passed"]))


def write_stage2_summary() -> None:
    fields = ["update", "scenario", "method", "window_s", "observed_ticks", "physical_failure",
              "speed_rmse_m_s", "steer_rmse_rad", "heading_rmse_rad", "final_speed_rmse_m_s",
              "final_steer_rmse_rad", "final_heading_rmse_rad", "final_speed_hold", "final_steer_hold",
              "final_heading_hold", "joint_final_hold", "peak_roll_rad", "roll_violation_s",
              "final_xy_error_m", "final_along_m", "final_lateral_m", "final_heading_error_rad"]
    with (OUT / "data/stage2_six_case_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for update in (80, 120):
            metrics = json.loads((STAGE2 / f"evaluation{update}/stage2_metrics.json").read_text())
            for case in CASES:
                for method in METHODS:
                    for window, key in [(10, "main10")] + ([(16, "extension16")] if case == "fast_turn" else []):
                        row = metrics[case][method][key]
                        with np.load(STAGE2 / f"evaluation{update}/{case}/{method}.npz", allow_pickle=False) as trace:
                            valid = np.flatnonzero(trace["time"] < window - 1e-6); index = int(valid[-1])
                            xy_error = float(np.linalg.norm(trace["actual_xy"][index] - trace["reference_xy"][index]))
                            along, lateral = float(trace["along"][index]), float(trace["lateral"][index])
                            heading = float(trace["e_psi_unwrapped"][index])
                        hold = row["final_hold_speed_steer_heading"]
                        writer.writerow(dict(update=update, scenario=case, method=method, window_s=window,
                            observed_ticks=row["observed_ticks"], physical_failure=row["physical_failure"],
                            speed_rmse_m_s=row["command"]["speed_rmse"], steer_rmse_rad=row["command"]["steer_rmse"],
                            heading_rmse_rad=row["command"]["heading_rmse"], final_speed_rmse_m_s=row["final"]["speed_rmse"],
                            final_steer_rmse_rad=row["final"]["steer_rmse"], final_heading_rmse_rad=row["final"]["heading_rmse"],
                            final_speed_hold=hold[0], final_steer_hold=hold[1], final_heading_hold=hold[2],
                            joint_final_hold=row["joint_final_hold"], peak_roll_rad=row["peak_roll"],
                            roll_violation_s=row["working_roll_violation_s"], final_xy_error_m=xy_error,
                            final_along_m=along, final_lateral_m=lateral, final_heading_error_rad=heading))


def write_chain_summary() -> None:
    fields = ["update", "method", "raw_speed_m_s", "raw_steer_rad", "governed_speed_m_s", "governed_steer_rad",
              "upper_speed_offset_m_s", "upper_steer_offset_rad", "final_front_rad_s", "final_rear_rad_s",
              "actual_speed_m_s", "actual_steer_rad", "speed_request_below_m0p15_longest_s",
              "speed_request_below_m0p30_longest_s", "speed_governed_below_m0p15_longest_s",
              "speed_governed_below_m0p30_longest_s"]
    with (OUT / "data/fast_turn_chain_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for update in (80, 120):
            metrics = json.loads((STAGE2 / f"evaluation{update}/stage2_metrics.json").read_text())
            for method in METHODS:
                chain = metrics["fast_turn"][method]["chain"]["windows"]["steady_2_4"]
                full = metrics["fast_turn"][method]["chain"]["windows"]["full"]
                writer.writerow(dict(update=update, method=method, raw_speed_m_s=2.6, raw_steer_rad=.25,
                    governed_speed_m_s=chain["governed"][0]["mean"], governed_steer_rad=chain["governed"][1]["mean"],
                    upper_speed_offset_m_s=chain["upper_offset"][0]["mean"], upper_steer_offset_rad=chain["upper_offset"][1]["mean"],
                    final_front_rad_s=chain["final_command"][0]["mean"], final_rear_rad_s=chain["final_command"][1]["mean"],
                    actual_speed_m_s=chain["actual_forward_speed"]["mean"], actual_steer_rad=chain["actual_delta"]["mean"],
                    speed_request_below_m0p15_longest_s=full["proposal_offset_negative_dwell"]["-0.15"]["longest_seconds"],
                    speed_request_below_m0p30_longest_s=full["proposal_offset_negative_dwell"]["-0.3"]["longest_seconds"],
                    speed_governed_below_m0p15_longest_s=full["upper_offset_negative_dwell"]["-0.15"]["longest_seconds"],
                    speed_governed_below_m0p30_longest_s=full["upper_offset_negative_dwell"]["-0.3"]["longest_seconds"]))


def tensorboard_scalars() -> dict[str, dict[str, list]]:
    result = {"alpha0": {}, "alpha1": {}}
    rows = []
    for stage, root in [("stage1", STAGE1), ("stage2", STAGE2)]:
        for alpha in (0, 1):
            acc = EventAccumulator(str(root / f"tensorboard/alpha{alpha}"), size_guidance={"scalars": 0}); acc.Reload()
            for tag in acc.Tags().get("scalars", []):
                events = acc.Scalars(tag)
                result[f"alpha{alpha}"].setdefault(tag, []).extend(events)
                rows.extend(dict(stage=stage, alpha=alpha, tag=tag, step=e.step, wall_time=e.wall_time, value=e.value) for e in events)
    with (OUT / "data/tensorboard_scalars.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stage", "alpha", "tag", "step", "wall_time", "value"])
        writer.writeheader(); writer.writerows(sorted(rows, key=lambda x: (x["alpha"], x["tag"], x["step"])))
    return result


def plot_tensorboard(data: dict[str, dict[str, list]]) -> None:
    panels = [("train/mean_step_reward", "Mean step reward"), ("physical/speed_rmse", "Training speed RMSE (m/s)"),
              ("physical/steer_rmse", "Training steer RMSE (rad)"), ("physical/heading_rmse", "Training heading RMSE (rad)"),
              ("ppo/mean_kl", "Mean KL"), ("ppo/temporal_coefficient", "Actor temporal coefficient")]
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    colors = {"alpha0": "#0072b2", "alpha1": "#d55e00"}
    for ax, (tag, title) in zip(axes.ravel(), panels):
        for run in ("alpha0", "alpha1"):
            events = sorted(data[run].get(tag, []), key=lambda e: e.step)
            ax.plot([e.step for e in events], [e.value for e in events], color=colors[run], label=run, lw=1.3)
        for step in (20, 40, 80, 120): ax.axvline(step, color=".65", ls=":" if step != 40 else "--", lw=1)
        ax.set_title(title); ax.grid(alpha=.25); ax.legend()
    axes[-1, 0].set_xlabel("Cumulative PPO update"); axes[-1, 1].set_xlabel("Cumulative PPO update")
    fig.suptitle("Preference V5 TensorBoard scalars | Stage1 1-40, Stage2 41-120")
    fig.tight_layout(); fig.savefig(OUT / "figures/tensorboard_training_curves.png", dpi=150); plt.close(fig)


def copy_evidence() -> None:
    for update in (20, 40):
        for sign in ("positive", "negative"):
            for kind in ("control", "xy"):
                copy(STAGE1 / f"evaluation{update}/{sign}_5s_{kind}.png", f"figures/stage1_u{update}_{sign}_{kind}.png")
        copy(STAGE1 / f"evaluation{update}/stage1_gate.json", f"data/stage1_gate_update{update}.json")
    for case in CASES:
        for kind in ("control", "xy"):
            copy(STAGE2 / f"evaluation120/{case}_10s_{kind}.png", f"figures/stage2_u120_{case}_{kind}.png")
    for update in (80, 120):
        copy(STAGE2 / f"evaluation{update}/fast_turn_16s_control.png", f"figures/stage2_u{update}_fast_turn_16s_control.png")
        copy(STAGE2 / f"evaluation{update}/fast_turn_16s_xy.png", f"figures/stage2_u{update}_fast_turn_16s_xy.png")
        copy(STAGE2 / f"evaluation{update}/stage2_metrics.json", f"data/stage2_metrics_update{update}.json")
    for kind in ("motor_chain", "reward", "reward_components"):
        copy(STAGE2 / f"evaluation120/fast_turn_10s_{kind}.png", f"figures/stage2_u120_fast_turn_{kind}.png")
    for name in ("manifest.json", "status.json", "stage_transition.json", "budget.json"):
        copy(STAGE2 / name, f"identity/stage2_{name}")
    copy(STAGE1 / "status.json", "identity/stage1_status.json")
    copy(STAGE1 / "manifest.json", "identity/stage1_manifest.json")
    copy(STAGE2 / "analysis/decision.json", "identity/decision.json")
    copy(STAGE2 / "alpha0/frozen_config_stage2.json", "config/alpha0_stage2.json")
    copy(STAGE2 / "alpha1/frozen_config_stage2.json", "config/alpha1_stage2.json")
    copy(STAGE1 / "alpha0/frozen_config_stage1.json", "config/alpha0_stage1.json")
    copy(STAGE1 / "alpha1/frozen_config_stage1.json", "config/alpha1_stage1.json")


def write_source_identity() -> None:
    commits = subprocess.check_output(["git", "log", "--oneline", "--reverse", "e1a6cc1..68996da"], cwd=REPO, text=True)
    (OUT / "identity/source_commits.txt").write_text(commits)
    stat = subprocess.check_output(["git", "diff", "--stat", "e1a6cc1..68996da"], cwd=REPO, text=True)
    (OUT / "identity/source_diff_stat.txt").write_text(stat)
    final = {}
    for alpha in (0, 1):
        p = STAGE2 / f"alpha{alpha}/checkpoints/update_0120.pt"
        final[f"alpha{alpha}"] = {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "update": 120}
    (OUT / "identity/final_checkpoints.json").write_text(json.dumps(final, indent=2) + "\n")


def write_manifest() -> None:
    files = []
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        files.append({"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "manifest.json").write_text(json.dumps({"schema": "r196_preference_v5_compact_evidence_v1",
        "source_revision": "68996dac9e8fef0bdb440093f44033eb2e8229a0", "files": files}, indent=2) + "\n")


def main() -> None:
    for folder in (OUT / "data", OUT / "figures", OUT / "identity", OUT / "config"): folder.mkdir(parents=True, exist_ok=True)
    copy_evidence(); write_stage1_summary(); write_stage2_summary(); write_chain_summary()
    compact_npz(STAGE1, 20, ["positive", "negative"], "stage1_update20_timeseries.npz")
    compact_npz(STAGE1, 40, ["positive", "negative"], "stage1_update40_timeseries.npz")
    compact_npz(STAGE2, 80, CASES, "stage2_update80_timeseries.npz")
    compact_npz(STAGE2, 120, CASES, "stage2_update120_timeseries.npz")
    plot_tensorboard(tensorboard_scalars()); write_source_identity(); write_manifest()


if __name__ == "__main__": main()
