"""Score the completed batches of a windowed PCRP CEM run in their actual packing."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_pcrp_e1 import GATES, finish, frontier, physical_lane, table, zero_rewards
from sttw_control.isaac_path_reporting import replay_rewards

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs/paper/english"))
from score_decision_oracle import score_trajectories


def read(path):
    return json.loads(Path(path).read_text())


def saved(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def batch_role(name):
    if name == "zero":
        return "same_slot_zero", 0
    if name.startswith("g") and name[1:].isdigit():
        return "pure_search", int(name[1:])
    if name.startswith("single_"):
        return "isolated_replay", 0
    if name.startswith("shift_"):
        return "ridge_shift", int(name.split("_")[1])
    raise ValueError(f"Unknown evaluated batch {name}")


def batch_rows(run, archive, spec, registered):
    records = []
    for p in sorted((run / "physics").glob("*/score.json")):
        folder = p.parent
        role, index = batch_role(folder.name)
        splits = read(p)
        cases = read(folder / "cases.json")
        slots = 0
        if len(cases) != len(splits):
            raise ValueError(f"Case count mismatch at {p}")
        with np.load(folder / "knots.npz") as knot:
            biases = knot["offsets"]
        for info, recorded in zip(cases, splits, strict=True):
            raw = Path(info["output"])
            actual = score_trajectories(raw, info["case_labels"])
            contract = read(raw / "contract.json")
            name = recorded[0]["case"]
            settings = contract["settings"]
            terrain_hash = (
                spec[name]["offsets"][str(index)]["sha256"]
                if role == "ridge_shift"
                else spec[name]["terrain_sha256"]
            )
            if (
                contract["identity"]["files"]
                != {k: registered["identity"][k] for k in ("actor.npz", "config.json", "route.npz")}
                or settings["lower_sha256"] != registered["identity"]["local300"]
                or info["seed"] != spec[name]["seed"]
                or info["randomization"] != spec[name]["randomization"]
                or any(t["sha256"] != terrain_hash for t in contract["terrain"])
            ):
                raise ValueError(f"Case identity differs from registration at {raw}")
            if len(actual) != len(recorded) or len(info["case_labels"]) != len(recorded):
                raise ValueError(f"Invalid lane count at {p}, {name}")
            for lane, (orig, rec) in enumerate(zip(actual, recorded, strict=True)):
                for key in (
                    "A",
                    "B",
                    "C",
                    "full",
                    "hold",
                    "terminal",
                    "V",
                    "speed_rmse_m_s",
                    "peak_path_m",
                    "time_s",
                ):
                    if orig[key] != rec[key]:
                        raise ValueError(f"{name} lane {lane}: saved {key} disagrees with trajectory")
                origin = contract["origin"][lane]
                if rec["origin"] != origin or float(info["alphas"][lane]) != spec[name]["alpha"]:
                    raise ValueError(f"{name} lane {lane}: alpha or world origin mismatch")
                if role in ("pure_search", "same_slot_zero") and (
                    np.any(biases[slots + lane]) != (role == "pure_search")
                ):
                    raise ValueError(f"Zero or pure search changed role at {raw}")
                records.append(
                    dict(
                        rec,
                        role=role,
                        batch=folder.name,
                        case=name,
                        candidate=lane,
                        slot=slots + lane,
                        origin_x_m=origin[0],
                        origin_y_m=origin[1],
                        origin_z_m=origin[2],
                        generation=index if role == "pure_search" else 0,
                        shift_cm=index if role == "ridge_shift" else 0,
                        knots_npz=str(folder / "knots.npz"),
                        knots_lane=slots + lane,
                    )
                )
            slots += len(recorded)
        if slots != len(biases):
            raise ValueError(f"Knots and physical lane counts differ at {p}")
    for row in archive:
        if (
            row["case"] not in spec
            or row["method"].lower() != "zero"
            and not row["method"].startswith("fixed")
        ):
            continue
        c = read(Path(row["raw"]) / "contract.json")
        lane = int(row["lane"])
        origin = c["origin"][lane]
        r = dict(row)
        for k in ("full", "hold", "goal", "A", "B", "C", "lane"):
            r[k] = int(r[k])
        for k in ("time_s", "peak_path_m", "path_rmse_m", "speed_rmse_m_s", "V"):
            r[k] = float(r[k])
        r.update(
            role="historical_reference",
            batch="historical",
            generation=0,
            candidate=lane,
            slot=None,
            origin=origin,
            origin_x_m=origin[0],
            origin_y_m=origin[1],
            origin_z_m=origin[2],
            alpha=float(c["settings"]["alphas"][lane]),
            shift_cm=0,
            knots_npz="",
            knots_lane=None,
        )
        records.append(r)
    return records


def collect_arrays(records, out):
    """Read each packed file once. Every real lane retains its own actual stopping mask."""
    sources = defaultdict(list)
    for r in records:
        sources[r["raw"]].append(r)
    for raw, group in sources.items():
        folder = Path(raw)
        c = read(folder / "contract.json")
        n = len(c["origin"])
        arrays = out / group[0]["case"] / "arrays"
        arrays.mkdir(parents=True, exist_ok=True)
        filename = arrays / f"{group[0]['batch']}.npz"
        bundle = Path(c["bundle"])
        cfg = read(bundle / "config.json")
        with np.load(bundle / "route.npz") as route:
            ref_xy = route["xy"]
            ref_s = route["s"]
        with np.load(folder / "trajectory.npz") as z:
            state = {
                k: z[k]
                for k in z.files
                if k.startswith("effective_")
                or k in ("valid", "failure", "reward", "lower_trace", "offsets")
            }
            lower = state["lower_trace"].reshape(-1, n, 11)
            valid = state["valid"].astype(bool)
            active = lower[:, :, 6].astype(bool)
            if not np.isfinite(lower[:, :, :6][active]).all():
                raise ValueError(f"Nonfinite valid trace at {raw}")
            reward, parts, error = replay_rewards(state, cfg)
            policy_reward = reward.reshape(-1, 4, n).sum(axis=1)
            times = np.arange(len(lower))[:, None] * c["lower_dt"] + lower[:, :, 7] * c["physics_dt"]
            data = dict(
                time_s=times,
                xy_m=lower[:, :, :2],
                valid_lower=active,
                valid_policy=valid,
                policy_step=np.arange(1, len(valid) + 1),
                reference_xy_m=ref_xy,
                reference_s_m=ref_s,
                role=group[0]["role"],
                origin_m=np.array(c["origin"]),
                speed_m_s=lower[:, :, 2],
                cross_track_m=lower[:, :, 3],
                delivered_reference=lower[:, :, 8:10],
                offsets=state["offsets"],
                reward=policy_reward,
                reward_cumulative=np.cumsum(policy_reward, axis=0),
                progress_m=z["progress"],
                request_speed_m_s=c["settings"]["user_speed_m_s"],
            )
            for k, part in parts.items():
                data["reward_" + k] = part.reshape(-1, 4, n).sum(axis=1)
            for r in group:
                lane = r["lane"]
                ids = np.flatnonzero(active[:, lane])
                if not len(ids) or abs(times[ids[-1], lane] - r["time_s"]) > 1e-8:
                    raise ValueError(f"Observed terminal time disagrees at {raw}, lane {lane}")
                r.update(
                    valid_policy_steps=int(valid[:, lane].sum()),
                    valid_lower_samples=len(ids),
                    reward_closure_max=float(error),
                    series=str(filename.relative_to(out)),
                )
            np.savez_compressed(filename, **data)
    return records


def write_step_csvs(out):
    """Export each lane through its actual last transition, retaining terminal reward."""
    groups = defaultdict(list)
    index = out / "episodes.csv"
    for row in read_rows(index):
        groups[row["series"]].append(row)
    with TemporaryDirectory(dir=out) as tmp:
        archive = Path(tmp) / "per_rollout_steps.zip"
        with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=3) as z:
            z.write(index, "episodes.csv")
            for series, rows in groups.items():
                with np.load(out / series) as batch:
                    reward_fields = [k for k in batch.files if k == "reward" or k.startswith("reward_")]
                    rewards = {k: batch[k] for k in reward_fields}
                    valid = batch["valid_policy"]
                    steps = batch["policy_step"]
                    lower_valid = batch["valid_lower"].reshape(len(steps), -1, valid.shape[1])
                    end_times = np.where(batch["valid_lower"], batch["time_s"], -np.inf)
                    end_times = end_times.reshape(lower_valid.shape).max(axis=1)
                    for row in rows:
                        lane = int(row["lane"])
                        mask = valid[:, lane]
                        times = end_times[mask, lane]
                        if not (
                            len(times) == int(row["valid_policy_steps"])
                            and np.isfinite(times).all()
                            and np.isclose(times[-1], float(row["time_s"]), atol=1e-8, rtol=0)
                        ):
                            raise ValueError(f"Invalid policy-step endpoint at {series}, lane {lane}")
                        columns = [steps[mask], times, *(rewards[k][mask, lane] for k in reward_fields)]
                        name = f"{row['case']}/{row['batch']}/lane_{lane:02d}.csv"
                        with z.open(name, "w") as stream:
                            np.savetxt(
                                stream,
                                np.column_stack(columns),
                                delimiter=",",
                                comments="",
                                header="policy_step,end_time_s," + ",".join(reward_fields),
                                fmt=["%d"] + ["%.17g"] * (len(columns) - 1),
                            )
        archive.replace(out / archive.name)


def population_panels(rows, same_slot, out, case_dir):
    """Pair each sampled candidate with its own frozen-policy baseline and original route."""
    if not rows:
        return []
    groups = defaultdict(list)
    for r in rows:
        groups[r["generation"]].append(r)
    pages = []
    with np.load(out / same_slot[0]["series"]) as z:
        baseline = {
            k: z[k]
            for k in ("xy_m", "valid_lower", "reward", "valid_policy", "reference_xy_m", "reference_s_m")
        }
    for generation, group in sorted(groups.items()):
        with np.load(out / group[0]["series"]) as d:
            actual = {k: d[k] for k in ("xy_m", "valid_lower", "reward", "valid_policy")}
        group = sorted(group, key=lambda r: r["candidate"])
        for kind in ("xy", "reward"):
            fig, axes = plt.subplots(4, 8, figsize=(16, 8), layout="constrained")
            for ax, r in zip(axes.flat, group, strict=True):
                ix = r["lane"]
                if kind == "xy":
                    xy = actual["xy_m"][:, ix][actual["valid_lower"][:, ix]]
                    control = baseline["xy_m"][:, ix][baseline["valid_lower"][:, ix]]
                    route = baseline["reference_xy_m"][
                        baseline["reference_s_m"] <= max(r["time_s"] * 2.6 + 11, 15)
                    ]
                    ax.plot(*route.T, "k:", lw=0.65)
                    ax.plot(*control.T, color="0.6", lw=1)
                    ax.plot(*xy.T, color="#1461a0", lw=1)
                    ax.plot(*xy[-1], marker="o" if r["full"] else "x", color="#1461a0", ms=3)
                    ax.set_aspect("equal", adjustable="datalim")
                else:
                    for source, color in ((baseline, "0.6"), (actual, "#1461a0")):
                        keep = source["valid_policy"][:, ix]
                        v = source["reward"][:, ix][keep]
                        ax.plot(np.flatnonzero(keep) + 1, v, color=color, lw=0.85)
                    ax.set_yscale("symlog")
                ax.tick_params(labelsize=6)
                ax.set_title(f"slot {r['slot']} · C {r['C']} · {r['time_s']:.2f}s", fontsize=7)
            label = (
                "XY (m), dotted original route"
                if kind == "xy"
                else "signed reward vs. actual 50 Hz step"
            )
            fig.suptitle(
                f"{case_dir.name} · generation {generation} · {label} · blue candidate / grey same-slot H75",
                fontsize=12,
            )
            target = case_dir / f"g{generation:02d}_{kind}.png"
            finish(fig, target)
            pages.append(target.name)
    return pages


def case_report(out, name, rows, config):
    folder = out / name
    folder.mkdir(exist_ok=True)
    pure = [r for r in rows if r["role"] == "pure_search"]
    zero = [r for r in rows if r["role"] == "same_slot_zero"]
    warm = [r for r in rows if r["role"] == "historical_reference"]
    refs = [r for r in rows if r["role"] in ("isolated_replay", "ridge_shift")]
    winner = min(pure, key=lambda r: (r["V"], r["generation"], r["candidate"])) if pure else None
    if len(warm) != 6:
        raise ValueError(f"Expected five fixed preferences and one historical Zero for {name}")
    if refs:
        with np.load(winner["knots_npz"]) as z:
            selected_knots = z["offsets"][winner["knots_lane"]]
        for r in refs:
            with np.load(r["knots_npz"]) as z:
                if r["origin"] != winner["origin"] or not np.array_equal(
                    selected_knots, z["offsets"][r["knots_lane"]]
                ):
                    raise ValueError(
                        f"World origin or control knots differ from selected candidate in {r['raw']}"
                    )
    control = next((r for r in zero if winner is not None and r["slot"] == winner["slot"]), None)
    for r in pure:
        ref = next(z for z in zero if z["slot"] == r["slot"])
        if r["origin"] != ref["origin"] or r["alpha"] != ref["alpha"]:
            raise ValueError(f"Changed zero baseline identity at {r['raw']}")
        r["same_slot_zero_V"] = ref["V"]
        r["delta_V_from_zero"] = r["V"] - ref["V"]
    reps = [
        (row, label)
        for row, label in [(winner, "CEM winner"), (control, "same-slot zero offset")]
        if row
    ]
    if not reps and zero:
        reps = [(zero[0], "zero offset, lane 0")]
    reps.extend(
        (r, "single, same origin" if r["role"] == "isolated_replay" else f"ridge {r['shift_cm']:+d} cm")
        for r in refs
    )
    unsuccessful = [r for r in pure if not r["C"] and r is not winner]
    if unsuccessful:
        negative = min(unsuccessful, key=lambda r: (r["V"], r["generation"], r["candidate"]))
        reps.append((negative, f"lowest-V remaining C failure (slot {negative['slot']})"))
        if control and negative["slot"] != control["slot"]:
            reps.append(
                (
                    next(r for r in zero if r["slot"] == negative["slot"]),
                    f"zero, failure-case slot {negative['slot']}",
                )
            )
    hist = [(r, r["method"]) for r in warm]

    def panels(title, size):
        fig, axes = plt.subplots(2, 1, figsize=size, layout="constrained")
        fig.suptitle(name + " · " + title)
        return fig, axes

    xy, ax = panels("measured and original reference XY", (8.6, 8))
    reward, rx = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    reward.suptitle(name + " · 50 Hz signed reward, through actual endpoint")
    for i, (set_, label) in enumerate(
        [(reps, "A1 measured layouts"), (hist, "historical frozen H75 and Zero")]
    ):
        target = ax[i]
        target.set_title(label, fontsize=10)
        for k, (r, tag) in enumerate(set_):
            with np.load(out / r["series"]) as z:
                lane = r["lane"]
                valid = z["valid_lower"][:, lane]
                mask = z["valid_policy"][:, lane]
                x = z["xy_m"][:, lane][valid]
                t = z["policy_step"][mask]
                rew = z["reward"][:, lane][mask]
                progress = z["progress_m"][:, lane][valid]
                if not k:
                    ids = z["reference_s_m"] <= max(30.0, float(progress.max()) + 0.4)
                    target.plot(
                        *z["reference_xy_m"][ids].T, ls="--", lw=1, c="black", label="original route"
                    )
                if r["method"].lower() == "zero":
                    score = zero_rewards(r, float(config["alpha"]))
                    t = score["policy_step"]
                    rew = score["total"]
                    tag += f" (rescored α={config['alpha']:g})"
                    np.savez_compressed(
                        folder / "historical_zero_paired_reward.npz", alpha=config["alpha"], **score
                    )
                elif r["role"] == "historical_reference":
                    tag += f" (α={r['alpha']:g})"
                (line,) = target.plot(*x.T, lw=1.2, label=tag)
                target.scatter(*x[-1], s=24, marker="o" if r["full"] else "x", color=line.get_color())
                rx[i, 0].plot(t, rew, lw=1, label=tag)
                rx[i, 1].plot(t, np.cumsum(rew), lw=1, label=tag)
        target.set_aspect("equal", adjustable="datalim")
        target.set(xlabel="route-frame x (m)", ylabel="route-frame y (m)")
        for j, lab in enumerate(("reward / actual policy step", "cumulative reward")):
            rx[i, j].set(xlabel="policy step", ylabel=lab, title=label, yscale="symlog")
            rx[i, j].grid(alpha=0.2)
        if set_:
            target.legend(fontsize=7, loc="best")
            rx[i, 1].legend(fontsize=6)
        target.grid(alpha=0.2)
    finish(xy, folder / "xy.png")
    finish(reward, folder / "reward.png")
    eligible, front = frontier(pure)
    fig, f = plt.subplots(figsize=(7.1, 5.3), layout="constrained")
    for tag, ls, marker in (
        ("pure_search", "CEM full, no hold", "."),
        ("same_slot_zero", "zero-offset full+hold", "s"),
        ("historical_reference", "historical full+hold", "^"),
        ("isolated_replay", "single full+hold", "D"),
        ("ridge_shift", "shift full+hold", "x"),
    ):
        sample = [
            r
            for r in rows
            if r["role"] == tag and r["full"] and (not r["hold"] if tag == "pure_search" else r["hold"])
        ]
        if sample:
            f.scatter(
                [r["peak_path_m"] for r in sample],
                [r["speed_rmse_m_s"] for r in sample],
                s=16,
                marker=marker,
                label=ls,
                alpha=0.7,
            )
    if eligible:
        f.scatter(
            [r["peak_path_m"] for r in eligible],
            [r["speed_rmse_m_s"] for r in eligible],
            s=14,
            color="#239566",
            label="CEM full+hold",
        )
    if front:
        seq = sorted(front, key=lambda r: r["peak_path_m"])
        f.plot(
            [r["peak_path_m"] for r in seq],
            [r["speed_rmse_m_s"] for r in seq],
            color="#136339",
            label="observed CEM frontier",
        )
    wp, wv = GATES["C"]
    f.plot([0, wp, wp], [wv, wv, 0], "k--", label="Task C bounds")
    f.set(xlabel="peak absolute cross-track error (m)", ylabel="speed RMSE (m/s)", title=name)
    f.grid(alpha=0.2)
    f.legend(fontsize=8)
    finish(fig, folder / "frontier.png")
    fields = [
        "role",
        "batch",
        "generation",
        "candidate",
        "slot",
        "lane",
        "full",
        "hold",
        "A",
        "B",
        "C",
        "V",
        "peak_path_m",
        "speed_rmse_m_s",
        "time_s",
        "raw",
        "series",
    ]
    table(folder / "frontier.csv", [{k: r.get(k) for k in fields} for r in front], fields)
    within = []
    for slot in sorted({r["slot"] for r in pure}):
        within += frontier([r for r in pure if r["slot"] == slot])[1]
    table(folder / "within_slot_frontier.csv", [{k: r.get(k) for k in fields} for r in within], fields)
    population = population_panels(pure, zero, out, folder)
    (folder / "population.md").write_text(
        "# 全部非零候选与同槽零偏置对照\n\n" + "\n".join(f"- [{f}]({f})" for f in population) + "\n"
    )
    timeline = folder / "timeline.png"
    if winner:
        c = read(Path(winner["raw"]) / "contract.json")
        with np.load(Path(c["bundle"]) / "route.npz") as z:
            route = {k: z[k] for k in ("xy", "s", "turn_end")}
        fig, axs = plt.subplots(4, 2, figsize=(10, 10), sharex=True, layout="constrained")
        fields = [
            ("bias_v_m_s", "Δv (m/s)"),
            ("bias_steer_rad", "Δδ (rad)"),
            ("speed_m_s", "body speed (m/s)"),
            ("steer_rad", "steer (rad)"),
            ("roll_rad", "roll (rad)"),
            ("path_error_m", "signed path error (m)"),
            ("front_normal_resultant_N", "front contact force (N)"),
            ("rear_normal_resultant_N", "rear contact force (N)"),
        ]
        events = []
        for r, tag in reps:
            contract = read(Path(r["raw"]) / "contract.json")
            index = r["lane"]
            data, event = physical_lane(
                dict(r, generation=int(r["role"] != "same_slot_zero")), route, contract["terrain"][index]
            )
            event.update(label=tag, slot=r["slot"], raw=r["raw"])
            events.append(event)
            t = data["time_s"]
            for a, (key, lab) in zip(axs.flat, fields):
                color = a.plot(t, data[key], lw=1, label=tag)[0].get_color()
                a.set_ylabel(lab)
                a.grid(alpha=0.17)
                if event["entry_s"] is not None:
                    a.axvline(event["entry_s"], color=color, ls=":", lw=0.7)
        axs[0, 0].legend(fontsize=7)
        axs[-1, 0].set_xlabel("time (s)")
        axs[-1, 1].set_xlabel("time (s)")
        fig.suptitle(name + " · measured interval, dotted first ridge entry")
        finish(fig, timeline)
        table(folder / "events.csv", events)

    def statistics(pool):
        return dict(
            n=len(pool),
            full=sum(r["full"] for r in pool),
            hold=sum(r["hold"] for r in pool),
            A=sum(r["A"] for r in pool),
            B=sum(r["B"] for r in pool),
            C=sum(r["C"] for r in pool),
            min_V=min((r["V"] for r in pool), default=None),
        )

    return dict(
        case=name,
        alpha=config["alpha"],
        pure=statistics(pure),
        same_slot_zeros=statistics(zero),
        selected={
            k: winner[k]
            for k in (
                "batch",
                "slot",
                "lane",
                "origin",
                "V",
                "A",
                "B",
                "C",
                "peak_path_m",
                "speed_rmse_m_s",
            )
        }
        if winner
        else None,
        matched_control={k: control[k] for k in ("slot", "origin", "V", "C")} if control else None,
        archive_n=len(warm),
        pure_full_hold_frontier_size=len(front),
        nominal=statistics([r for r in refs if r["role"] == "isolated_replay"]),
        shifted={
            str(r["shift_cm"]): dict(full=r["full"], hold=r["hold"], C=r["C"], V=r["V"])
            for r in refs
            if r["role"] == "ridge_shift"
        },
    )


def analyze(run, out):
    reg = read(run / "registration.json")
    status = read(run / "status.json")
    specs = {case["case"]: case for case in reg["cases"]}
    archive = read_rows(run.parent / "archive_baseline/scores.csv")
    rows = batch_rows(run, archive, specs, reg)
    out.mkdir(parents=True, exist_ok=True)
    collect_arrays(rows, out)
    cases = [case_report(out, n, [r for r in rows if r["case"] == n], spec) for n, spec in specs.items()]
    measured = [r for r in rows if r["role"] != "historical_reference"]
    if status["stage"] == "complete" and len(measured) != status["finished"]:
        raise ValueError(
            f"Score files account for {len(measured)} lanes, terminal status records {status['finished']}"
        )
    unscored = []
    for manifest in sorted((run / "physics").glob("*/cases.json")):
        if not (manifest.parent / "score.json").is_file():
            unscored.append(
                dict(
                    batch=manifest.parent.name,
                    planned_lanes=sum(len(c["case_labels"]) for c in read(manifest)),
                    archived_trajectories=[
                        c["output"]
                        for c in read(manifest)
                        if (Path(c["output"]) / "trajectory.npz").is_file()
                    ],
                )
            )
    decision = read(run / "decision.json") if (run / "decision.json").is_file() else None
    bitwise = (
        read(run / "single_comparisons.json") if (run / "single_comparisons.json").is_file() else []
    )
    nominal_checks = []
    for case in cases:
        best = case["selected"]
        isolated = next(
            (r for r in measured if r["case"] == case["case"] and r["role"] == "isolated_replay"), None
        )
        if isolated is None:
            continue
        pick = next(
            r
            for r in measured
            if r["case"] == case["case"] and r["batch"] == best["batch"] and r["slot"] == best["slot"]
        )
        if pick["origin"] != isolated["origin"] or isolated["slot"] != 0:
            raise ValueError("Single-environment nominal identity changed")
        with (
            np.load(Path(pick["raw"]) / "trajectory.npz") as a,
            np.load(Path(isolated["raw"]) / "trajectory.npz") as b,
        ):
            if set(a.files) != set(b.files):
                raise ValueError("Replay members differ")
            equality = {}
            for key in a.files:
                axis = (
                    2
                    if key
                    in {
                        "lower_trace",
                        "wheel_contact_points",
                        "wheel_contact_forces",
                        "executed_commands",
                        "body_poses",
                    }
                    else 1
                )
                left = a[key].take(pick["lane"], axis=axis)
                right = b[key].take(0, axis=axis)
                equality[key] = (
                    left.dtype == right.dtype
                    and left.shape == right.shape
                    and left.tobytes() == right.tobytes()
                )
        official = next((r for r in bitwise if r["case"] == case["case"]), None)
        if official and (
            equality != {k: v["bitwise"] for k, v in official["arrays"].items()}
            or all(equality.values()) != official["all_recorded_arrays_bitwise"]
        ):
            raise ValueError("Independent replay differs from runner's member-by-member record")
        nominal_checks.append(
            dict(
                case=case["case"],
                all_recorded_arrays_bitwise=all(equality.values()),
                equal_members=sum(equality.values()),
                members=len(equality),
                fields=equality,
            )
        )
    positive = None
    verified_ridges = []
    if decision:
        flat, *ridges = cases
        positive = bool(flat["selected"]["V"] <= 0.676 and flat["nominal"]["min_V"] <= 0.676)
        for c in ridges:
            if set(c["shifted"]) != {"-1", "1"} or c["nominal"]["n"] != 1:
                raise ValueError("Terminal decision lacks nominal or shifted checks")
            verified_ridges.append(
                bool(
                    c["selected"]["C"]
                    and c["nominal"]["C"]
                    and sum(r["full"] for r in c["shifted"].values()) >= 1
                )
            )
        if (
            decision["flat_passed"] != positive
            or [r["qualifies"] for r in decision["ridge_cases"]] != verified_ridges
        ):
            raise ValueError("Terminal decision disagrees with frozen gates")
        n = sum(verified_ridges)
        good = {r["case"] for r in bitwise if r["all_recorded_arrays_bitwise"]}
        verdict = (
            "flat_control_failed"
            if not positive
            else "route_A_no_robust_solution"
            if not n
            else "one_ridge_partial"
            if n == 1
            else "route_B_eligible"
        )
        if verdict == "route_B_eligible" and (
            flat["case"] not in good
            or sum(ok and c["case"] in good for ok, c in zip(verified_ridges, ridges)) < 2
        ):
            verdict = "nominal_replay_disagreement_no_expansion"
        if verdict != decision["verdict"]:
            raise ValueError("Disagreement on the registered route")
    else:
        verdict = "verification_incomplete"
    saved(
        out / "decision_audit.json",
        dict(
            source_stage=status,
            verified_route=verdict,
            confirmed_flat=positive,
            confirmed_ridge_gates=verified_ridges,
            completed_scores=len(measured),
            cases=cases,
            unscored_batches=unscored,
            single_bitwise=nominal_checks,
        ),
    )
    fields = [
        "case",
        "role",
        "batch",
        "generation",
        "candidate",
        "slot",
        "lane",
        "origin_x_m",
        "origin_y_m",
        "origin_z_m",
        "alpha",
        "shift_cm",
        "full",
        "hold",
        "A",
        "B",
        "C",
        "V",
        "hold_ratio",
        "peak_path_m",
        "path_rmse_m",
        "speed_rmse_m_s",
        "time_s",
        "terminal",
        "valid_policy_steps",
        "valid_lower_samples",
        "same_slot_zero_V",
        "delta_V_from_zero",
        "reward_closure_max",
        "raw",
        "series",
        "knots_npz",
        "knots_lane",
    ]
    table(out / "episodes.csv", [{k: r.get(k) for k in fields} for r in rows], fields)
    write_step_csvs(out)
    lines = [
        "# A1 · 测得的控制序列和接触相位",
        "",
        "## 实际轨迹与原始参考",
        "",
        "|开发工况（物理组 401）|XY|逐步及累计奖励|峰值路径误差与速度 RMSE|控制与接触时序|",
        "|---|---|---|---|---|",
    ]
    for c in cases:
        n = c["case"]
        timeline = f"[时序]({n}/timeline.png)" if c["selected"] else "尚无完整搜索代"
        lines.append(
            f"|{n}|[实际与参考]({n}/xy.png) · [每一候选]({n}/population.md)|[奖励]({n}/reward.png)|[观测前沿]({n}/frontier.png) · [同槽前沿]({n}/within_slot_frontier.csv)|{timeline}|"
        )
    lines += [
        "",
        "同一行搜索图使用胜者与其同槽、同世界原点的零偏置 H75 对照。历史五档 H75 和关闭中层的 Zero 单独展示。各档沿实际评分偏好计分，Zero 的对照曲线按本案例冻结偏好重评分。各轨迹按实际终点截止。XY 采用平移到共同路线的米制坐标，每条原始世界原点见明细。",
        "",
        "## 已落盘的样本",
        "",
        "|工况|纯 CEM 样本|满 10 s|末窗保持|Task A/B/C|最低 V|相应槽零偏置 V|独立单槽 C|−1/+1 cm 存活|",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for c in cases:
        p = c["pure"]
        z = c["matched_control"]
        single = c["nominal"]
        off = (
            " / ".join(str(c["shifted"].get(k, {}).get("full", "待测")) for k in ("-1", "1"))
            if c != cases[0]
            else "无脊"
        )

        def fmt(v):
            return "待测" if v is None else f"{v:.4f}"

        lines.append(
            f"|{c['case']}|{p['n']}|{p['full']}|{p['hold']}|{p['A']}/{p['B']}/{p['C']}|{fmt(p['min_V'])}|{fmt(z['V'] if z else None)}|{single['C'] if single['n'] else '待测'}|{off}|"
        )
    lines += [
        "",
        f"状态 \x60{status['stage']}\x60。物理提交 {status['submitted']} 条，完成评分 {len(measured)} 条，完整搜索 {status['complete_search_generations']} 代。分流审计 \x60{verdict}\x60。",
    ]
    if unscored:
        lines += [
            "",
            "以下批次尚无完整评分文件，未进入报告分母："
            + "、".join(r["batch"] for r in unscored)
            + "。",
        ]
    lines += [
        "",
        "每个工况每代 32 个候选占用不同物理槽位与世界原点。第二代起，6 个精英控制序列原样进入下一代。上表统计实际评估次数，包含重复使用的精英。逐槽非支配表保留同一原点内的比较。",
    ]
    if not decision:
        lines += [
            "",
            "所列最低值只来自完整落盘搜索代。确认门需完成单槽及偏移实测，本次未得到同时满足所有确认条件的分流证据。",
        ]
    lines += [
        "",
        "前沿仅使用完整 10 s 且达到末窗保持的非零 CEM 样本，按峰值绝对路径误差和全程速度 RMSE 筛出非支配点。历史参考、同槽零偏置和单槽/移脊验证单列。空前沿仅指此预算下未取得合格样本。",
        "",
        "[全部逐轨索引与真实终点](episodes.csv) · [逐轨每步奖励与分项 CSV](per_rollout_steps.zip) · [确认门与逐位复验](decision_audit.json)",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    counts = [c["pure"]["n"] for c in cases]
    total = sum(counts)
    snippet = "A second development search varied eight speed and eight steering offsets per candidate within 1--4~s, preserving the frozen controller outside that interval. "
    if total:
        ridge_names = ", ".join(s["layout"].split("_s")[1] for s in reg["cases"][1:])
        snippet += f"Across {total} scored nonzero candidate evaluations in four group-401 cases (flat, then ridges centred at {ridge_names}~m), the numbers passing $S_C$ were "
        snippet += ", ".join(f"{c['pure']['C']}/{c['pure']['n']}" for c in cases) + ", respectively. "
    if decision:
        flat = cases[0]
        snippet += f"The best flat candidate had $V={flat['selected']['V']:.3f}$ in its original slot and $V={flat['nominal']['min_V']:.3f}$ in the single-environment replay (positive-control threshold 0.676). "
        counts = ", ".join(str(sum(s["full"] for s in c["shifted"].values())) + "/2" for c in cases[1:])
        snippet += f"Among the two ridge shifts ($\\pm 1$~cm), the selected sequences completed 10~s in {counts} runs, respectively. "
        snippet += f"Of the three ridges, {sum(verified_ridges)} met the nominal $S_C$ and shifted-survival criteria. "
        exact = sum(c["all_recorded_arrays_bitwise"] for c in nominal_checks)
        snippet += f"All recorded arrays matched bit for bit between the packed and single-environment replays in {exact}/4 cases."
    else:
        snippet += "The confirmation sequence was incomplete at termination. Robustness under shifted contact phases remains unresolved."
    if total:
        snippet += r" Figure~\ref{fig:windowed_search} shows the observed nondominated error pairs among sampled sequences with full-window completion and final hold."
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), layout="constrained")
        for i, (c, a) in enumerate(zip(cases, axes.flat, strict=True)):
            sample = [
                r
                for r in measured
                if r["case"] == c["case"] and r["role"] == "pure_search" and r["full"]
            ]
            full_hold, pareto = frontier(sample)
            no_hold = [r for r in sample if not r["hold"]]
            for points, color, marker, label in [
                (no_hold, "0.7", "+", "Full only"),
                (full_hold, "#4a88bb", ".", "Full and hold"),
                (pareto, "#aa4400", "s", "Nondominated"),
            ]:
                if points:
                    a.scatter(
                        [r["peak_path_m"] for r in points],
                        [r["speed_rmse_m_s"] for r in points],
                        color=color,
                        s=11,
                        marker=marker,
                        label=label,
                    )
            a.axhline(GATES["C"][1], color="0.45", lw=0.6, ls="--")
            a.axvline(GATES["C"][0], color="0.45", lw=0.6, ls="--")
            a.set(
                xlabel=r"$E_p$ (m)",
                ylabel=r"$E_v$ (m/s)",
                title=("Flat" if i == 0 else reg["cases"][i]["layout"].split("_s")[1] + " m ridge"),
            )
            a.grid(alpha=0.15)
        handles, labels = [], []
        for a in axes.flat:
            for h, label in zip(*a.get_legend_handles_labels()):
                if label not in labels:
                    handles.append(h)
                    labels.append(label)
        if handles:
            fig.legend(handles, labels, loc="outside lower center", ncols=3, fontsize=7)
        fig.savefig(out / "windowed_search_frontier.pdf", bbox_inches="tight")
        plt.close(fig)
    (out / "a1_results.tex").write_text(snippet + "\n")
    print(
        json.dumps(
            dict(verified_route=verdict, completed_scores=len(measured), report=str(out / "README.md"))
        )
    )


def prepare_paper(source, out):
    """Bind this run's measured section to the current paper and rebuild the private source package."""
    paper = out / "paper"
    paper.mkdir(exist_ok=True)
    tex = (source / "PCRP.tex").read_text()
    section_start = "% BEGIN measured encounter-window analysis"
    section_end = "% END measured encounter-window analysis"
    base = re.sub(r"\n\n" + section_start + r".*?" + section_end, "", tex, flags=re.DOTALL)
    anchor = "We also tested whether an offline cross-entropy search"
    if base.count(anchor) != 1:
        raise ValueError(
            "The existing full-horizon search paragraph changed; review the paper insertion point"
        )
    i = base.index("\n\n", base.index(anchor))
    fragment = (out / "a1_results.tex").read_text().strip()
    audit = read(out / "decision_audit.json")
    if audit["source_stage"]["stage"] == "complete":
        ridges = audit["cases"][1:]
        count = len(ridges)
        plus_full = sum(c["shifted"]["1"]["full"] for c in ridges)
        minus_task = sum(c["shifted"]["-1"]["C"] for c in ridges)
        plus_task = sum(c["shifted"]["1"]["C"] for c in ridges)
        fragment += f" At the $+1$~cm shift, {plus_full}/{count} selected sequences completed the window. $S_C$ passed in {minus_task}/{count} runs at $-1$~cm and {plus_task}/{count} at $+1$~cm."
        rows = read_rows(out / "episodes.csv")
        for case in ridges:
            single = next(
                r for r in rows if r["case"] == case["case"] and r["role"] == "isolated_replay"
            )
            if case["selected"]["C"] and single["full"] == "0":
                center = case["case"].split("position_s")[1].split("_")[0]
                fragment += f" At {center}~m, the packed winner passed $S_C$ but terminated after {float(single['time_s']):.4f}~s in the single-environment replay."
        if audit["verified_route"] == "nominal_replay_disagreement_no_expansion":
            fragment += " The replay discrepancy triggered the registered stopping rule, so the search ended with these four cases."
    figure = out / "windowed_search_frontier.pdf"
    if figure.is_file():
        fragment += (
            "\n\n"
            + r"""\begin{figure*}[t]\centering
\includegraphics[width=.80\textwidth]{figures/windowed_search_frontier.pdf}
\caption{Encounter-window search in physical group~401 on four development layouts. Each panel pools the candidate evaluations at 32 world origins. Blue points completed 10~s and met final hold. Squares are nondominated for peak absolute lateral error $E_p$ and speed RMSE $E_v$ among these observations. Crosses completed 10~s without final hold. Dashed lines mark the two error bounds of $S_C$.}
\label{fig:windowed_search}
\end{figure*}"""
        )
    result = base[:i] + "\n\n" + section_start + "\n" + fragment + "\n" + section_end + base[i:]
    projected = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", result)
    members = ["PCRP.tex", "references.bib", "IEEEtran.cls", *dict.fromkeys(projected)]
    archive = paper / "PCRP_Overleaf_Bundle.zip"
    with TemporaryDirectory() as stage:
        stage = Path(stage)
        with ZipFile(archive, "w", compression=ZIP_DEFLATED) as z:
            z.writestr("PCRP.tex", result)
            for name in members[1:]:
                z.write(
                    figure if name == "figures/windowed_search_frontier.pdf" else source / name, name
                )
        with ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Corrupt private paper source bundle")
            z.extractall(stage)
        (paper / "PCRP.tex").write_bytes((stage / "PCRP.tex").read_bytes())
        tool = shutil.which("tectonic") or str(Path.home() / ".local/bin/tectonic")
        run = subprocess.run(
            [tool, "--only-cached", "--keep-logs", "--untrusted", "PCRP.tex"],
            cwd=stage,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        (paper / "build.log").write_text(run.stdout)
        run.check_returncode()
        log = (stage / "PCRP.log").read_text()
        if re.search(r"(?:Reference|Citation) .+ undefined|There were undefined references", log):
            raise ValueError("Unresolved references in result manuscript")
        shutil.copyfile(stage / "PCRP.pdf", paper / "PCRP.pdf")
    content = subprocess.check_output(["pdftotext", str(paper / "PCRP.pdf"), "-"], text=True)
    if "A second development search" not in content:
        raise ValueError("Compiled paper lacks the measured paragraph")
    saved(
        paper / "verification.json",
        dict(
            source=str(source),
            input_tex_sha256=hashlib.sha256(tex.encode()).hexdigest(),
            analysis=str(out / "decision_audit.json"),
            figure_members=projected,
            zip_members=members,
            rendered_pages=content.count("\f"),
            undefined_references=0,
            files_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (archive, paper / "PCRP.tex", paper / "PCRP.pdf")
            },
        ),
    )
    index = out / "README.md"
    content = index.read_text()
    if "(paper/PCRP.pdf)" not in content:
        content += "\n[本文结果段与实测前沿 PDF](paper/PCRP.pdf) · [可编辑 Overleaf 包](paper/PCRP_Overleaf_Bundle.zip) · [编译与输入身份](paper/verification.json)\n"
    if (out / "per_rollout_steps.zip").is_file() and "(per_rollout_steps.zip)" not in content:
        content += "\n[逐轨有效步、奖励与分项 CSV](per_rollout_steps.zip)\n"
    index.write_text(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--paper-source", type=Path)
    args = parser.parse_args()
    output = (args.output or args.run / "analysis").resolve()
    analyze(args.run.resolve(), output)
    if args.paper_source:
        prepare_paper(args.paper_source.resolve(), output)
