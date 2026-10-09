#!/usr/bin/env python3
"""Export the finished V5.1 fixed-policy evidence without running simulation."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2] / "runs/preference_v51_strong_primary_100_20261009"
CASES = ("straight_hold", "speed_changes", "gentle_positive", "gentle_negative",
         "steer_reversal", "fast_turn")
METHODS = ("B0", "alpha0", "alpha1")
FIELDS = ("time", "limited_command", "target", "governed", "latent_z",
          "actual_forward_speed", "actual_delta", "phi", "phi_dot",
          "e_psi_unwrapped", "yaw_rate", "actual_xy", "reference_xy",
          "wheel_speed_proxy", "slip_proxy", "u_nom", "u_goal",
          "requested_residual", "applied_residual", "lower_action",
          "final_command", "physical_failure", "scored_tick_reward",
          "checkpoint_update", "chi", "g", "peak_roll")


def copy(src: Path, dest: str) -> None:
    target = OUT / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)


def summary() -> None:
    rows = []
    for update, cases in ((25, ("straight_hold", "fast_turn")), (100, CASES)):
        metrics = json.loads((ROOT / f"evaluation{update}/stage2_metrics.json").read_text())
        for case in cases:
            for method in METHODS:
                item = metrics[case][method]
                windows = [(10, "main10", "v51_contract")]
                if case == "fast_turn":
                    windows.append((16, "extension16", "v51_contract_extension16"))
                for horizon, physical_key, contract_key in windows:
                    p = item[physical_key]
                    c = item[contract_key]
                    cost = c["component_integrals_and_caps"]
                    integrals = {name: sum(mode["cost_integrals"].get(name, 0.)
                                           for mode in cost.values())
                                 for name in ("primary_excess", "yaw_recovery", "heading",
                                              "speed", "steer", "roll", "roll_rate",
                                              "command_compatibility")}
                    cap = {name: sum(mode["seconds"] * (mode["cap_fraction"].get(name) or 0.)
                                     for mode in cost.values()) / horizon
                           for name in ("primary_excess", "yaw_recovery", "heading",
                                        "speed", "steer", "roll", "roll_rate")}
                    rows.append(dict(update=update, case=case, seed=77001, method=method,
                        horizon_s=horizon, observed_ticks=p["observed_ticks"],
                        physical_failure=p["physical_failure"],
                        steer_rmse_steady_rad=c["steady_steer_rmse_rad"],
                        speed_rmse_steady_m_s=c["steady_speed_rmse_m_s"],
                        actual_steer_steady_rad=c["steady_actual_steer_mean_rad"],
                        actual_speed_steady_m_s=c["steady_actual_speed_mean_m_s"],
                        peak_roll_rad=c["peak_abs_roll_rad"],
                        roll_over_0p302_s=c["roll_over_0p302_s"],
                        actual_drop_ge_0p1_longest_s=c["actual_drop_ge_0p1_longest_s"],
                        actual_drop_ge_0p2_longest_s=c["actual_drop_ge_0p2_longest_s"],
                        final_heading_rmse_rad=p["final"]["heading_rmse"],
                        final_speed_hold=p["final_hold_speed_steer_heading"][0],
                        final_steer_hold=p["final_hold_speed_steer_heading"][1],
                        final_heading_hold=p["final_hold_speed_steer_heading"][2],
                        joint_final_hold=p["joint_final_hold"],
                        mean_abs_speed_offset_m_s=p["mean_abs_offsets"][0],
                        mean_abs_steer_offset_rad=p["mean_abs_offsets"][1],
                        reward_audit_passed=item.get("reward_audit", {}).get("passed"),
                        **{f"cost_{k}_integral": v for k, v in integrals.items()},
                        **{f"cap_{k}_fraction": v for k, v in cap.items()}))
    with (OUT / "data/fixed_case_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def traces() -> None:
    for update, cases in ((25, ("straight_hold", "fast_turn")), (100, CASES)):
        data = {}
        for case in cases:
            for method in METHODS:
                with np.load(ROOT / f"evaluation{update}/{case}/{method}.npz", allow_pickle=False) as src:
                    names = [k for k in src.files if k in FIELDS or k.startswith("scored_cost_")
                             or k.startswith("component_capped_")]
                    for key in names:
                        data[f"{case}__{method}__{key}"] = src[key]
        np.savez_compressed(OUT / f"data/update{update}_timeseries.npz", **data)


def tensorboard() -> None:
    panels = (("train/mean_step_reward", "Mean training step reward"),
              ("physical/speed_rmse", "Training speed RMSE (m/s)"),
              ("physical/steer_rmse", "Training steer RMSE (rad)"),
              ("physical/heading_rmse", "Training heading RMSE (rad)"),
              ("ppo/mean_kl", "Mean KL"),
              ("ppo/temporal_coefficient", "Actor temporal coefficient"))
    scalars = {}
    rows = []
    for alpha in (0, 1):
        acc = EventAccumulator(str(ROOT / f"tensorboard/alpha{alpha}"),
                               size_guidance={"scalars": 0})
        acc.Reload()
        scalars[alpha] = {}
        for tag in acc.Tags().get("scalars", []):
            events = acc.Scalars(tag)
            scalars[alpha][tag] = events
            rows.extend(dict(alpha=alpha, tag=tag, step=e.step, wall_time=e.wall_time,
                             value=e.value) for e in events)
    with (OUT / "data/tensorboard_scalars.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("alpha", "tag", "step", "wall_time", "value"),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["alpha"], r["tag"], r["step"])))
    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)
    for ax, (tag, title) in zip(axes.flat, panels):
        for alpha, color in ((0, "#0072b2"), (1, "#d55e00")):
            events = scalars[alpha].get(tag, ())
            ax.plot([e.step for e in events], [e.value for e in events],
                    color=color, label=f"alpha{alpha}", linewidth=1.3)
        for step in (25, 100):
            ax.axvline(step, color=".6", linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.grid(alpha=.25)
        ax.legend()
    axes[-1, 0].set_xlabel("PPO update")
    axes[-1, 1].set_xlabel("PPO update")
    fig.suptitle("R196 Preference V5.1 | fresh alpha0 / alpha1 | updates 1–100")
    fig.tight_layout()
    fig.savefig(OUT / "figures/tensorboard_training_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    for dirname in ("data", "figures", "config", "identity"):
        (OUT / dirname).mkdir(exist_ok=True)
    for alpha in (0, 1):
        copy(ROOT / f"alpha{alpha}/frozen_config_stage51.json",
             f"config/alpha{alpha}.json")
    for name in ("manifest.json", "status.json", "evaluation100_error_receipt.json",
                 "evaluation100_recovery.json"):
        copy(ROOT / name, f"identity/{name}")
    for update, cases in ((25, ("straight_hold", "fast_turn")), (100, CASES)):
        copy(ROOT / f"evaluation{update}/stage2_metrics.json",
             f"data/update{update}_metrics.json")
        for case in cases:
            horizons = (10, 16) if case == "fast_turn" else (10,)
            for horizon in horizons:
                for kind in ("xy", "control", "reward", "reward_components"):
                    name = f"{case}_{horizon}s_{kind}.png"
                    copy(ROOT / f"evaluation{update}/{name}",
                         f"figures/update{update}_{name}")
        if update == 100:
            copy(ROOT / "evaluation100/fast_turn_10s_motor_chain.png",
                 "figures/update100_fast_turn_10s_motor_chain.png")
    summary()
    traces()
    tensorboard()


if __name__ == "__main__":
    main()
