"""Offline phase audit of retained NPZs. No environment or learner is imported."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
CONTROL = ROOT / "runs/path_safety_control_200_20261011"
CANDIDATE = ROOT / "runs/path_safety_candidate_200_20261011"
DT = .005
NAMES = ["straight_2.0", "straight_2.6", "left90_R2_2.0",
         "left90_R2_2.6", "right90_R2_2.0", "right90_R2_2.6"]
SOURCES = {
    "B0 local300": CONTROL / "zero_upper",
    "Coverage alpha0": CONTROL / "alpha0/validation/update_0000",
    "Coverage alpha1": CANDIDATE / "alpha1/validation/update_0000",
    "Control alpha0 u200": CONTROL / "alpha0/validation/update_0200",
    "Safety alpha1 u200": CANDIDATE / "alpha1/validation/update_0200",
}
COLORS = [".5", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def read(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def event(d, index):
    return dict(index=int(index), time_s=float(d["time"][index] + DT),
                progress_m=float(d["path_progress"][index]),
                signed_path_m=float(d["path_cross_track"][index]),
                peak_roll_rad=float(d["peak_roll"][index]),
                heading_error_rad=float(d["path_heading_error"][index]))


result = dict(schema="paired200_phase_audit_v1", physics_interactions=0,
              sources={key: str(value) for key, value in SOURCES.items()},
              time_convention="time + .005 = post-lower-tick; peak_roll includes physics substeps",
              cases={}, preference={}, limits="one training seed per endpoint; historical DEV; no overlimit recovery")
traces = {}
for name in NAMES:
    result["cases"][name] = {}
    traces[name] = {}
    path = read(CONTROL / "zero_upper" / f"{name}_path.npz")
    turn_end = float(path["turn_end"])
    for label, folder in SOURCES.items():
        d = read(folder / f"{name}.npz")
        traces[name][label] = d
        assert len(d["time"]) == 4000 and np.all(d["active_tick"])
        assert not np.any(d["physical_failure"] | d["domain_exit"])
        tail = d["time"] + DT > 19.
        ey, ev = d["path_cross_track"], d["actual_forward_speed"] - d["v_user"]
        over = np.flatnonzero(np.abs(ey) > .4)
        crossed = np.flatnonzero(d["path_progress"] >= turn_end)
        item = dict(max_roll_rad=float(d["peak_roll"].max()),
                    roll_exceed_seconds=float(np.sum(d["peak_roll"] > .302) * DT),
                    max_path_m=float(np.abs(ey).max()),
                    path_exceed_seconds=float(len(over) * DT),
                    final_max_speed_error=float(np.abs(ev[tail]).max()),
                    final_max_path_error=float(np.abs(ey[tail]).max()),
                    final_max_heading_error=float(np.abs(d["path_heading_error"][tail]).max()),
                    first_corridor_exceed=event(d, over[0]) if len(over) else None,
                    roll_peak=event(d, np.argmax(d["peak_roll"])),
                    path_peak=event(d, np.argmax(np.abs(ey))),
                    first_turn_end=event(d, crossed[0]) if turn_end > 0 and len(crossed) else None,
                    turn_end_m=turn_end,
                    final_chain={key: np.mean(d[key][tail], axis=0).tolist() for key in
                                 ("nominal", "offsets", "governed", "final_command", "actual_delta",
                                  "actual_forward_speed", "chi", "effective_cost_path", "effective_cost_speed")})
        item["final_hold"] = (item["final_max_speed_error"] <= .1 and
                              item["final_max_path_error"] <= .1 and
                              item["final_max_heading_error"] <= .05)
        curvature = np.max(np.abs(np.stack([
            np.interp(d["path_progress"] + lookahead, path["s"], path["curvature"])
            for lookahead in (0., .5, 1.)])), axis=0)
        mask = ((d["chi"] <= 1e-6) & (curvature <= 1e-3)
                & (np.maximum(np.abs(d["phi"]), np.abs(d["phi"] + .1*d["phi_dot"])) <= .26)
                & (np.abs(ey) <= .1) & (np.abs(ev) <= .1)
                & (np.abs(d["path_heading_error"]) <= .05))
        item["post_lower_dev_mask_proxy_fraction"] = float(mask.mean())
        item["mask_caveat"] = "post-lower DEV proxy; NOT training pre-action mask coverage"
        result["cases"][name][label] = item
    if name.endswith("2.6") and turn_end > 0:
        edges = np.append(np.arange(3., turn_end, .5), turn_end)
        bins = {}
        for label in ("Coverage alpha0", "Coverage alpha1"):
            d = traces[name][label]
            errors, counts = [], []
            for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
                m = (d["path_progress"] >= lo) & ((d["path_progress"] <= hi) if index == len(edges)-2 else (d["path_progress"] < hi))
                counts.append(int(m.sum()))
                assert m.any(), "Incomplete turn bins must not be silently discarded"
                errors.append([np.mean((d["actual_forward_speed"][m]-d["v_user"][m])**2),
                               np.mean(d["path_cross_track"][m]**2)])
            ev, ey = np.sqrt(np.mean(errors, axis=0))
            bins[label] = dict(speed_rmse=float(ev), path_rmse=float(ey), counts=counts)
        result["preference"][name] = dict(bins=bins,
            Dv=bins["Coverage alpha0"]["speed_rmse"]-bins["Coverage alpha1"]["speed_rmse"],
            Dy=bins["Coverage alpha1"]["path_rmse"]-bins["Coverage alpha0"]["path_rmse"],
            qualification="descriptive; parent alpha1 violates .40m corridor")

fig, axes = plt.subplots(3, 2, figsize=(13, 13))
for column, name in enumerate(("left90_R2_2.6", "right90_R2_2.6")):
    path = read(CONTROL / "zero_upper" / f"{name}_path.npz")
    end = float(path["turn_end"])
    m = path["s"] <= end + 3
    axes[0, column].plot(*path["xy"][m].T, "k--", label="original fixed path")
    for (label, d), color in zip(traces[name].items(), COLORS):
        m = (d["path_progress"] >= 0) & (d["path_progress"] <= end + 3)
        axes[0, column].plot(*d["actual_xy"][m].T, color=color, label=label)
        axes[1, column].plot(d["path_progress"][m], d["peak_roll"][m], color=color, label=label)
        axes[2, column].plot(d["path_progress"][m], d["path_cross_track"][m], color=color, label=label)
        first = result["cases"][name][label]["first_corridor_exceed"]
        if first is not None:
            axes[2, column].plot(first["progress_m"], first["signed_path_m"], "o", color=color, ms=5)
    axes[0, column].set(title=name + " | turn detail XY", xlabel="World X (m)", ylabel="World Y (m)")
    axes[0, column].set_aspect("equal", adjustable="box")
    axes[1, column].axhline(.302, color="k", ls="--", label="roll limit .302 rad")
    axes[1, column].set(xlabel="Original path progress (m)", ylabel="5ms peak roll (rad)")
    axes[2, column].axhline(.4, color="k", ls="--")
    axes[2, column].axhline(-.4, color="k", ls="--", label="corridor +/- .40 m")
    axes[2, column].set(xlabel="Original path progress (m)", ylabel="Signed path error (m)")
    for row in (1, 2):
        axes[row, column].axvspan(3., end, color="k", alpha=.05, label="original curved section")
        axes[row, column].axvline(end, color="k", ls=":")
for ax in axes.flat:
    ax.grid(alpha=.2)
    ax.legend(fontsize=7)
fig.suptitle("Preserve learned coordination; reduce in-turn detour and later forgetting\n"
             "Spatial detail from complete 20s traces; all survive | fixed original path, no external disturbance | 1 train seed/endpoint")
fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(OUT / "TURN_PHASE_REASSESSMENT.png", dpi=145)
plt.close(fig)
(OUT / "phase_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(complete_traces=30, physics_interactions=0, preference=result["preference"]), indent=2))
