"""Score matched reference rules and actors from their actual frozen trajectories."""

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "learning/src"))
from sttw_control.isaac_path_reporting import replay_rewards, static_case_metrics
from sttw_control.torch_controller import TorchController
from sttw_control.torch_path_reference import TorchPathReference

EVIDENCE = ROOT / "docs/evidence/pcrp_matched_controls"
CORE = ("s_bend_pulse", "right_bend", "flat", "oblique_2cm", "oblique_4cm")
FAMILIES = ("H75", "Speed cap", "Steering bias", "Specialists", "Shared Go2", "Zero")
STYLE = dict(
    zip(
        FAMILIES,
        zip(
            ("#0072B2", "#A25322", "#328363", "#8A4C90", "#D69615", "#505050"),
            ("o", "s", "^", "D", "P", "x"),
        ),
    )
)
STYLE["Predictive governor"] = ("#C5531A", "v")
BOX = np.array([0.8, 0.3])


def csv_rows(path):
    with path.open() as fp:
        return list(csv.DictReader(fp))


def csv_write(path, rows):
    with path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        w.writeheader()
        w.writerows(rows)


def family(row):
    text = (row["cohort"] + " " + row.get("method", "")).lower()
    if row.get("stage") == "T2_selector" and row["cohort"] in ("selector", "fixed"):
        return "Pre-run selector" if row["cohort"] == "selector" else "Fixed H75"
    if row["cohort"] in ("fixed0", "fixed05", "fixed1"):
        return "Specialists"
    if row["cohort"] == "predictive_governor":
        return "Predictive governor"
    if "7101" in text:
        return "Shared Go2"
    if "zero" in text:
        return "Zero"
    if "bias" in text:
        return "Steering bias"
    if "cap" in text or "governor" in text:
        return "Speed cap"
    if "h75" in text or "pcrp" in text:
        return "H75"
    raise ValueError("Unrecognized method family: " + text)


def zero_reward_slope(contract, data, lane, cfg):
    # Affine difference r(alpha=1)-r(alpha=0) on the original lower-only trajectory.
    c = cfg["path_reward"]
    tr = data["lower_trace"][:, :, lane].reshape(-1, 11)
    speed = float(contract["settings"]["user_speed_m_s"])
    cc = json.loads((ROOT / contract["settings"]["controller_config"]).read_text())
    ratio = TorchController(cc, 1)._system(torch.tensor([speed]))[3]
    with np.load(Path(contract["bundle"]) / "route.npz") as z:
        route = TorchPathReference(z, cfg)
        curv = route.at(torch.from_numpy(data["progress"][:, lane]))[2]
    chi = (
        ((torch.atan(cc["wheelbase"] * curv / math.cos(cc["caster"])) / ratio).abs() - 0.18)
        .div(0.12)
        .clamp(0, 1)
        .numpy()
    )
    huber = lambda x: np.where(abs(x) <= 1, x * x, 2 * abs(x) - 1)
    ev = tr[:, 2] - speed
    caps = c["component_caps"]
    speed1 = np.minimum(10 * huber(ev / c["speed_scale_m_s"]), caps["speed"])
    primary1 = np.minimum(
        c["primary_weight"]
        * chi
        * huber(np.maximum(abs(ev) - c["primary_speed_band_m_s"], 0) / c["primary_speed_scale_m_s"]),
        caps["primary"],
    )
    diff = (
        speed1
        - data["effective_speed"][:, lane]
        + primary1
        - data["effective_primary"][:, lane]
        + (c["path_secondary_ratio"] - 1) * data["effective_path"][:, lane]
    )
    perstep = (-c["scale"] * contract["lower_dt"] * diff * tr[:, 6]).reshape(-1, 4).sum(1)
    return np.where(data["valid"][:, lane] & ~data["failure"][:, lane], perstep, 0)


@lru_cache(maxsize=32)
def load_contract(source):
    return json.loads((source / "contract.json").read_text())


def check_pair(spec, contract, lane):
    if not spec.get("h75_reference"):
        return
    source = Path(spec["h75_reference"])
    reference = load_contract(source)
    other = int(spec["h75_lane"])
    for key in ("physics_dt", "lower_dt", "policy_dt"):
        if contract[key] != reference[key]:
            raise ValueError(f"Timing differs for {spec['raw']}: {key}")
    for key in ("route.npz", "config.json"):
        if contract["identity"]["files"][key] != reference["identity"]["files"][key]:
            raise ValueError(f"Route or task config differs for {spec['raw']}: {key}")
    for key in ("seed", "lower_sha256", "randomization", "initial_state", "user_speed_m_s"):
        if contract["settings"].get(key) != reference["settings"].get(key):
            raise ValueError(f"Case setting differs for {spec['raw']}: {key}")
    pulses = []
    for stream in (contract, reference):
        pulse = stream["settings"].get("evaluation_pulse") or {}
        pulses.append(pulse if any(pulse.get("world_force_N", [])) else {})
    if pulses[0] != pulses[1]:
        raise ValueError(f"Applied pulse differs for {spec['raw']}")
    if contract["terrain"][lane] != reference["terrain"][other]:
        raise ValueError(f"Terrain metadata differs for {spec['raw']}")
    if spec["cohort"] in ("shared7101", "fixed0", "fixed05", "fixed1"):
        np.testing.assert_array_equal(
            contract["lab_physics"]["terrain_layout"]["clone_origins"][lane],
            reference["lab_physics"]["terrain_layout"]["clone_origins"][other],
            err_msg=f"Registered physical clone origin differs for {spec['raw']} lane {lane}",
        )
    for category in ("delay_steps", "randomization_readback"):
        for k, measured in reference[category].items():
            np.testing.assert_array_equal(
                contract[category][k][lane],
                measured[other],
                err_msg=f"{spec['raw']} lane {lane} versus {source} lane {other}, {k}",
            )


def read_traces(index, diagnostics):
    by_file = defaultdict(list)
    for spec in index:
        by_file[spec["raw"]].append(spec)
    traces = []
    flags = ("roll_limit_failure", "domain_exit", "contact_failure", "numerical_failure", "policy_fault")
    for raw, items in by_file.items():
        source = Path(raw)
        contract = load_contract(source)
        bundle = Path(contract["bundle"])
        cfg = json.loads((bundle / "config.json").read_text())
        with np.load(source / "trajectory.npz") as archive:
            keys = {"reward", "valid", "failure", "lower_trace", "progress", "pose", "offsets"} | {
                k for k in archive.files if k.startswith("effective_") or k in flags
            }
            data = {k: archive[k] for k in keys}
        _, parts, closure = replay_rewards(data, cfg)
        tr = data["lower_trace"]
        metrics = static_case_metrics(data, contract)
        with np.load(bundle / "route.npz") as route:
            reference_xy = route["xy"][route["s"] <= float(route["goal"])]
        for spec in items:
            i = int(spec["lane"])
            m = metrics[i]
            f = family(spec)
            check_pair(spec, contract, i)
            a = float(contract["settings"]["alphas"][i])
            if abs(a - float(spec["alpha"])) > 1e-6:
                raise ValueError(f"Index alpha differs from {source} lane {i}")
            name = spec.get("method") or ("alpha_" + format(a, "g"))
            if f in ("H75", "Shared Go2", "Specialists"):
                name = "alpha_" + format(a, ".9g")
            x = tr[:, :, i].reshape(-1, 11)
            ids = np.flatnonzero(x[:, 6])
            t = ids * contract["lower_dt"] + x[ids, 7].astype(float) * contract["physics_dt"]
            valid = data["valid"][:, i].astype(bool)
            reward = data["reward"][valid, i]
            terminal = "+".join(k for k in flags if k in data and data[k][ids, i].any())
            full = not m["failed"] and abs(t[-1] - 10.0) < 1e-4
            if not terminal:
                terminal = "full_window" if full else "unresolved"
            row = {
                "stage": spec.get("stage", "frontier"),
                "condition": spec["condition"],
                "group": int(spec["group"]),
                "family": f,
                "method": name,
                "cohort": spec["cohort"],
                "alpha": a,
                "full": int(full),
                "hold": int(m["final_hold"]),
                "goal_crossed": int(m["goal_crossed"]),
                "terminal": terminal,
                "end_s": float(t[-1]),
                "peak_path_m": float(abs(x[ids, 3]).max()),
                "whole_speed_rmse_m_s": m["speed_rmse_m_s"],
                "whole_path_rmse_m": m["path_rmse_m"],
                "reward_rebuild_max_error": closure,
                "actor": contract["identity"]["files"]["actor.npz"],
                "raw": raw,
                "lane": i,
                "h75_reference": spec.get("h75_reference", raw),
                "h75_lane": spec.get("h75_lane", i),
            }
            row.update(
                {
                    k: int(
                        full and row["hold"] and row["peak_path_m"] <= l and row["whole_speed_rmse_m_s"] <= v
                    )
                    for k, (l, v) in {"A": (0.3, 0.2), "B": (0.6, 0.1), "C": (0.4, 0.15)}.items()
                }
            )
            pulse = contract["settings"].get("evaluation_pulse") or {}
            use_pulse = np.any(pulse.get("world_force_N", [0]))
            d = {
                "row": row,
                "t": t,
                "xy": x[ids, :2],
                "ey": x[ids, 3],
                "ev": x[ids, 2] - m["requested_speed_m_s"],
                "ref": reference_xy,
                "reward": reward,
                "steps": np.flatnonzero(valid) + 1,
                "pulse": (pulse["start_s"], pulse["start_s"] + pulse["duration_s"]) if use_pulse else None,
            }
            if f == "Zero":
                if a != 0:
                    raise ValueError(f"Zero reward slope requires archived alpha=0, got {a} in {source}")
                d["slope"] = zero_reward_slope(contract, data, i, cfg)[valid]
            if diagnostics:
                name_id = f"trace_{len(traces):04d}.npz"
                series = {
                    k: np.asarray(vals).reshape(len(valid), 4, -1)[:, :, i].sum(1)[valid]
                    for k, vals in parts.items()
                }
                np.testing.assert_allclose(sum(series.values()), reward, atol=1e-4, rtol=1e-5)
                np.savez_compressed(
                    diagnostics / name_id,
                    time_s=t,
                    actual_xy=d["xy"],
                    reference_xy=reference_xy,
                    cross_track_m=d["ey"],
                    speed_error_m_s=d["ev"],
                    policy_step=d["steps"],
                    total_reward=reward,
                    cumulative_reward=np.cumsum(reward, dtype=float),
                    **({"zero_reward_alpha_slope": d["slope"]} if "slope" in d else {}),
                    **{"reward_" + k: v for k, v in series.items()},
                )
                row["diagnostic_npz"] = "traces/" + name_id
            traces.append(d)
    return traces


def hypervolume(points):
    """Area between the lower-left nondominated set and the fixed upper-right bound."""
    last_y = BOX[1]
    area = 0.0
    for x, y in sorted(map(tuple, points)):
        if 0 <= x < BOX[0] and 0 <= y < last_y:
            area += (BOX[0] - x) * (last_y - y)
            last_y = y
    return float(area)


def diagnostics(case, items, out, family_names=FAMILIES[:-1]):
    cond, group = case
    zero = next(d for d in items if d["row"]["family"] == "Zero")
    prefix = f"{cond}_g{group}"
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.plot(*zero["ref"].T, "k:", lw=1.5, label="Original path")
    for f in (*family_names, "Zero"):
        lines = [d for d in items if d["row"]["family"] == f]
        color = STYLE.get(f, ("#29749D" if f == "Fixed H75" else "#8D4878", None))[0]
        for i, d in enumerate(lines):
            ax.plot(*d["xy"].T, color=color, lw=0.85, alpha=0.8, label=f if i == 0 else None)
            if not d["row"]["full"]:
                ax.scatter(*d["xy"][-1], color=color, s=12, marker="x")
    ax.set(
        xlabel="World x (m)",
        ylabel="World y (m)",
        aspect="equal",
        title=f"{cond} | physical group {group} | observed trajectories",
    )
    ax.legend(fontsize=7, ncol=3, frameon=False)
    overview = prefix + "_all_XY.png"
    fig.savefig(out / overview, dpi=135)
    plt.close(fig)
    links = [overview]
    for f in family_names:
        members = sorted(
            [d for d in items if d["row"]["family"] == f],
            key=lambda d: (d["row"]["alpha"], d["row"]["method"]),
        )
        for start in range(0, len(members), 6):
            subset = members[start : start + 6]
            colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(subset)))
            fig, axs = plt.subplots(2, 2, figsize=(10, 6.0), layout="constrained")
            ax, rw, cum, err = axs.flat
            ax.plot(*zero["ref"].T, "k:", lw=1.6, label="Original path")
            ax.plot(*zero["xy"].T, color=".4", lw=1.0, label="Zero")
            if not zero["row"]["full"]:
                ax.scatter(*zero["xy"][-1], marker="x", c=".4")
            err.plot(zero["t"], zero["ev"], color=".4", lw=1)
            for d, col in zip(subset, colors):
                r = d["row"]
                name = r["method"]
                a = r["alpha"]
                ax.plot(*d["xy"].T, color=col, lw=1.1, label=name)
                if not r["full"]:
                    ax.scatter(*d["xy"][-1], marker="x", color=col)
                rw.plot(d["steps"], d["reward"], color=col, lw=0.8)
                cum.plot(d["steps"], np.cumsum(d["reward"], dtype=float), color=col, lw=1.0)
                ref = zero["reward"] + (a - zero["row"]["alpha"]) * zero["slope"]
                rw.plot(zero["steps"], ref, "--", color=col, lw=0.7, alpha=0.75)
                cum.plot(zero["steps"], np.cumsum(ref, dtype=float), "--", color=col, lw=0.7, alpha=0.75)
                err.plot(d["t"], d["ev"], color=col, lw=1.0)
            ax.set(xlabel="World x (m)", ylabel="World y (m)", aspect="equal")
            rw.set(
                xlabel="Policy step",
                ylabel="Reward per step",
                yscale="symlog",
                title="Dashed: Zero at the same preference",
            )
            cum.set(xlabel="Policy step", ylabel="Cumulative reward", yscale="symlog")
            err.set(xlabel="Observed time (s)", ylabel="Forward speed error (m/s)")
            if zero["pulse"]:
                t0, t1 = zero["pulse"]
                err.axvspan(t0, t1, color=".7", alpha=0.3)
                for a_ in (rw, cum):
                    a_.axvspan(t0 / 0.02, t1 / 0.02, color=".7", alpha=0.3)
            fig.suptitle(f"{cond} | physical group {group} | {f}")
            ax.legend(fontsize=6, ncol=2, frameon=False)
            fname = prefix + "_" + f.replace(" ", "_") + f"_{start // 6 + 1}.png"
            fig.savefig(out / fname, dpi=135)
            plt.close(fig)
            links.append(fname)
    return links


def common_window(items):
    end = min(d["t"][-1] for d in items)
    # All policies use exactly the same complete lower ticks. Drop the final
    # partial contact interval for every method, preserving it in raw metrics.
    n = int(np.floor((end + 1e-7) / 0.005))
    for d in items:
        r = d["row"]
        np.testing.assert_allclose(d["t"][:n], np.arange(1, n + 1) * 0.005, atol=1e-7, rtol=0)
        r.update(
            common_s=0.005 * n,
            common_path_rmse_m=math.sqrt(np.mean(np.square(d["ey"][:n], dtype=float))),
            common_speed_rmse_m_s=math.sqrt(np.mean(np.square(d["ev"][:n], dtype=float))),
        )


def summarize_frontier(traces, output):
    groups = defaultdict(list)
    for d in traces:
        r = d["row"]
        if r["condition"] in CORE:
            groups[(r["condition"], r["group"])].append(d)
    if set(groups) != {(s, i) for s in CORE for i in range(101, 106)}:
        raise ValueError("Incomplete five-condition/five-group panel")
    aggregates = defaultdict(list)
    gallery = []
    for case, items in groups.items():
        counts = Counter(d["row"]["family"] for d in items)
        if (
            counts["H75"] != 13
            or counts["Speed cap"] != 18
            or counts["Steering bias"] != 12
            or counts["Zero"] != 1
        ):
            raise ValueError(f"Incomplete frontier in {case}: {counts}")
        if counts["Specialists"] != counts["Shared Go2"] or counts["Specialists"] not in (0, 3):
            raise ValueError(f"Incomplete Go2 controls in {case}: {counts}")
        common_window(items)
        for d in items:
            r = d["row"]
            aggregates[(r["condition"], r["family"], r["method"])].append(r)
        gallery.extend((case[0], case[1], p) for p in diagnostics(case, items, output / "panels"))
    points = []
    for (cond, f, m), rr in sorted(aggregates.items()):
        if len(rr) != 5 or len({r["group"] for r in rr}) != 5:
            raise ValueError(f"Expected one observation per paired group for {cond}/{f}/{m}")
        points.append(
            {
                "condition": cond,
                "family": f,
                "method": m,
                "groups": 5,
                "path_rmse_m": float(np.mean([r["common_path_rmse_m"] for r in rr])),
                "speed_rmse_m_s": float(np.mean([r["common_speed_rmse_m_s"] for r in rr])),
                "full": sum(r["full"] for r in rr),
                "hold": sum(r["hold"] for r in rr),
                "window_min_s": min(r["common_s"] for r in rr),
                "window_max_s": max(r["common_s"] for r in rr),
            }
        )
    ratios = []
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.7), layout="constrained")
    for cond, ax in zip(CORE, axes.flat):
        rows = [r for r in points if r["condition"] == cond]
        pcrp = np.array([[r["path_rmse_m"], r["speed_rmse_m_s"]] for r in rows if r["family"] == "H75"])
        for f in FAMILIES:
            ff = [r for r in rows if r["family"] == f]
            if not ff:
                continue
            xy = np.array([[r["path_rmse_m"], r["speed_rmse_m_s"]] for r in ff])
            dominated = [np.any(np.all(pcrp <= v, axis=1) & np.any(pcrp < v, axis=1)) for v in xy]
            area = hypervolume(xy)
            color, marker = STYLE[f]
            ratios.append(
                {
                    "condition": cond,
                    "family": f,
                    "points": len(ff),
                    "inside_box": int(np.sum(np.all(xy < BOX, axis=1))),
                    "hypervolume": area,
                    "normalized": area / float(np.prod(BOX)),
                    "dominated_by_h75": sum(dominated),
                    "dominance_denominator": len(ff),
                    "dominated_fraction": np.mean(dominated),
                }
            )
            for r, v in zip(ff, xy):
                ax.scatter(*v, color=color, marker=marker, s=23, alpha=0.9 if r["full"] == 5 else 0.38)
            if f != "Zero":
                order = sorted(set(map(tuple, xy)))
                front = []
                last = np.inf
                for x, y in order:
                    if y < last:
                        front.append((x, y))
                        last = y
                ax.plot(*np.asarray(front).T, color=color, lw=0.8)
        ax.set(title=cond.replace("_", " "), xlabel="Cross-track RMSE (m)", ylabel="Speed RMSE (m/s)")
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    for f in [f for f in FAMILIES if any(r["family"] == f for r in points)]:
        color, marker = STYLE[f]
        axes.flat[-1].scatter([], [], marker=marker, color=color, label=f)
    axes.flat[-1].legend(loc="center", frameon=False)
    fig.savefig(output / "frontiers.pdf")
    fig.savefig(output / "frontiers.png", dpi=145)
    plt.close(fig)
    csv_write(output / "points.csv", points)
    csv_write(output / "hypervolume.csv", ratios)
    return gallery, ratios


def nominal_results(rows):
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    totals = []
    for condition, items in sorted(by_condition.items()):
        groups = {r["group"] for r in items}
        members = defaultdict(list)
        for r in items:
            members[(r["family"], r["method"])].append(r)
        for key, rr in members.items():
            if len(rr) != len(groups) or {r["group"] for r in rr} != groups:
                raise ValueError(f"Incomplete nominal treatment {condition} {key}")
        candidates = {}
        for family_name, n in [("H75", 5), ("Speed cap", 15)]:
            eligible = {method: rr for (f, method), rr in members.items() if f == family_name}
            if len(eligible) != n:
                raise ValueError(f"{condition} expected {n} {family_name} methods, got {sorted(eligible)}")
            # A single rule is chosen retrospectively for the whole layout. Apply
            # the identical lexicographic binary endpoint to both candidate sets.
            winner = min(
                eligible,
                key=lambda m: (
                    -sum(x["full"] for x in eligible[m]),
                    -sum(x["hold"] for x in eligible[m]),
                    m,
                ),
            )
            candidates[family_name] = (winner, eligible[winner])
        labels = [
            ("Fixed H75 " + a, "alpha_" + a, members[("H75", "alpha_" + a)]) for a in ("0", "0.5", "1")
        ]
        labels.append(("Frozen cap", "cap_L3_a2", members[("Speed cap", "cap_L3_a2")]))
        labels.extend((family_name + " layout oracle", m, rr) for family_name, (m, rr) in candidates.items())
        for label, m, rr in labels:
            totals.append(
                dict(
                    condition=condition,
                    comparison=label,
                    selected_method=m,
                    n=len(rr),
                    full=sum(r["full"] for r in rr),
                    hold=sum(r["hold"] for r in rr),
                    **{task: sum(r[task] for r in rr) for task in "ABC"},
                )
            )
    if sum(r["n"] for r in totals if r["comparison"] == "Frozen cap") != 40:
        raise ValueError("Expected 40 paired nominal cases")
    return totals


def selector_results(rows):
    by_case = defaultdict(dict)
    for r in rows:
        key = r["condition"] + "_seed" + str(r["group"])
        if r["method"] in by_case[key]:
            raise ValueError(f"Duplicated {key} {r['method']}")
        by_case[key][r["method"]] = r
    historical = csv_rows(ROOT / "docs/evidence/h75_theory_iteration/S4_layout/results/paired_choices.csv")
    expected = {(r["case"], r["task"]): r for r in historical if r["baseline"] == "fixed0"}
    if len(by_case) != 176 or set(by_case) != {k[0] for k in expected}:
        raise ValueError("Incomplete matched selection manifest")
    paired = []
    summary = []
    for task in "ABC":
        for case, mm in sorted(by_case.items()):
            selected = mm["mechanism_" + task]
            cap = mm["cap_L3_a2"]
            e = expected[(case, task)]
            if (float(e["selected_alpha"]), int(e["selected_success"])) != (
                selected["alpha"],
                selected[task],
            ):
                raise ValueError(f"Historical selector outcome differs at {case}/{task}")
            if int(e["baseline_success"]) != mm["fixed0"][task]:
                raise ValueError(f"Historical constant-alpha outcome differs at {case}/{task}")
            for end in ("full", "hold", task):
                s = selected[end]
                c = cap[end]
                paired.append(
                    {
                        "case": case,
                        "group": selected["group"],
                        "task": task,
                        "endpoint": end,
                        "selected_alpha": selected["alpha"],
                        "selector": s,
                        "cap": c,
                        "gain": int(s > c),
                        "loss": int(s < c),
                        "selector_raw": selected["raw"],
                        "selector_lane": selected["lane"],
                        "cap_raw": cap["raw"],
                        "cap_lane": cap["lane"],
                    }
                )
        for end in ("full", "hold", task):
            pp = [r for r in paired if (r["task"], r["endpoint"]) == (task, end)]
            n = len(pp)
            s = sum(r["selector"] for r in pp)
            c = sum(r["cap"] for r in pp)
            g = sum(r["gain"] for r in pp)
            l = sum(r["loss"] for r in pp)
            pvalue = float(binomtest(g, g + l, 0.5).pvalue) if g + l else 1.0
            s_ci = binomtest(s, n).proportion_ci(method="wilson")
            c_ci = binomtest(c, n).proportion_ci(method="wilson")
            summary.append(
                {
                    "task": task,
                    "endpoint": end,
                    "n": n,
                    "selector": s,
                    "selector_low": s_ci.low,
                    "selector_high": s_ci.high,
                    "cap": c,
                    "cap_low": c_ci.low,
                    "cap_high": c_ci.high,
                    "gain": g,
                    "loss": l,
                    "exact_mcnemar_p": pvalue,
                }
            )
    return paired, summary


def write_report(out, gallery, hv, nominal, paired, specialists):
    lines = [
        "# Paired performance of reference corrections",
        "",
        "![Matched actual and reference trajectories, S bend, physical group 101](panels/s_bend_pulse_g101_all_XY.png)",
        "",
        (
            "[Original XY, speed error and policy-step rewards](panels.csv) · [Trajectory-level outcomes](episodes.csv) · "
            "[Five error frontiers](frontiers.pdf) · [Frontier coordinates and completion](points.csv)"
        ),
        "",
        (
            "The five conditions each use physical groups 101–105. All compared policies retain the original route, "
            "requested speed, lower controller and pulse. Within each condition and physical group, cross-track "
            "and forward-speed errors use the complete 5 ms ticks observed by every method through the earliest "
            "termination. RMSE is calculated separately for each group. The five group RMSEs have equal weight "
            "in each plotted point. The per-run table also reports the whole observed trajectory, the "
            "termination cause, 10 s completion and the original final-hold criterion."
        ),
        "",
        "The H75 series comprises 13 frozen preferences. Analytical points represent frozen parameter settings. "
        + (
            "The three specialists each received 50 updates at one preference. Their pooled transitions match the "
            "150-update shared Go2 run (seed 7101). H75 was trained separately."
            if specialists
            else "The fixed-preference training and the budget-matched Go2 actor have a separate evaluation manifest."
        ),
        "",
        "## Five-condition frontiers",
        "",
        "![Cross-track and speed errors](frontiers.png)",
        "",
        (
            "Lower errors are preferred. Pale points include an early termination in at least one group. Each line joins "
            "the measured nondominated settings within a family. Dominated area is computed relative to (0.8 m, "
            "0.3 m/s). A point outside the box contributes zero area. The normalized area divides by 0.24. "
            "The final column counts points weakly worse on both errors and strictly worse on at least one than "
            "at least one H75 point, including points outside the box."
        ),
        "",
        "| Condition | Family | Raw area | Normalized area | Dominated by H75 |",
        "|---|---|---:|---:|---:|",
    ]
    for r in hv:
        lines.append(
            f"| {r['condition']} | {r['family']} | {r['hypervolume']:.5f} | {r['normalized']:.4f} | "
            f"{r['dominated_by_h75']}/{r['dominance_denominator']} |"
        )
    lines += [
        "",
        "[Hypervolume and denominators](hypervolume.csv) · [Completion and final hold by setting](points.csv)",
        "",
        "## Fixed deployment choices and retrospective selection",
        "",
        (
            "The same eight physical groups are paired within each nominal ridge placement and flat condition. "
            "Each gravel layout has four paired groups. The frozen cap uses 3 m preview and a 2 m/s² acceleration "
            "limit. Each oracle selects one candidate for the entire layout, maximizing 10 s completion, then "
            "final hold, then sorting tied candidate names. H75 has five candidate preferences and the cap has 15 "
            "candidate settings. Each displayed pair gives completions / final holds."
        ),
        "",
        "| Condition | n | Fixed 0 | Fixed .5 | Fixed 1 | Frozen cap | H75 layout oracle | Cap layout oracle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in sorted({r["condition"] for r in nominal}):
        nn = [r for r in nominal if r["condition"] == condition]
        pairs = [f"{r['full']}/{r['hold']}" for r in nn]
        lines.append("| " + " | ".join([condition, str(nn[0]["n"]), *pairs]) + " |")
    lines += [
        "",
        "[Selected methods and per-layout task counts](nominal.csv)",
        "",
        "## Pre-run choice paired with the frozen cap",
        "",
        (
            "The mechanism fixes an episode-constant preference before rollout using task, terrain family, height and "
            "requested speed. These are the original 176 held-out cases (22 conditions, two terrain layouts and four "
            "physical groups). The frozen cap is replayed on the same cases. Matching is by condition, layout and "
            "physical group, using each selected episode's own recorded lane. Case-wise Wilson intervals and exact "
            "two-sided McNemar tests are descriptive because the layouts and physical groups repeat."
        ),
        "",
        "| Task | Endpoint | Selector / 176 [95% CI] | Cap / 176 [95% CI] | Gains / losses | Exact p |",
        "|---|---|---:|---:|---:|---:|",
    ]

    def ci(r, method):
        return f"{r[method]}/176 [{100 * r[method + '_low']:.1f}–{100 * r[method + '_high']:.1f}%]"

    for r in paired:
        lines.append(
            f"| {r['task']} | {r['endpoint']} | {ci(r, 'selector')} | {ci(r, 'cap')} | "
            f"{r['gain']}/{r['loss']} | {r['exact_mcnemar_p']:.4g} |"
        )
    lines += [
        "",
        "[Paired case outcomes](selector_pairs.csv) · [Wilson intervals and exact paired tests](selector_summary.csv)",
        "",
        "## Trajectories and rewards",
        "",
        (
            "Each panel shows recorded XY against the original path and Zero, per-step and cumulative reward through "
            "the real terminal transition, and speed error. Dashed rewards are Zero rescored at the displayed actor "
            "preference. A cross marks an early termination. Shading marks the applied force pulse. The trajectory-level "
            "file links each archived lane and its compact diagnostic NPZ containing the observed signals and signed reward "
            "components. The rewards are reconstructed at policy cadence (20 ms), including any terminal replacement."
        ),
        "",
    ]
    by_case = defaultdict(list)
    for condition, group, image in gallery:
        by_case[(condition, group)].append(image)
    for (condition, group), images in sorted(by_case.items()):
        links = " · ".join(
            f"[{Path(f).stem.split('_g' + str(group) + '_')[-1]}](panels/{f})" for f in images
        )
        lines.append(f"- {condition}, group {group} — " + links)
    (out / "README.md").write_text("\n".join(lines) + "\n")


def specialist_panel(root, reference_only):
    """Report the registered three preferences without waiting on the rule sweeps."""
    manifest = root / "episodes_index.csv"
    specs = csv_rows(manifest)
    contract = json.loads((root / "protocol.json").read_text())
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != contract["eval_index_sha256"]:
        raise ValueError("The specialist manifest differs from the registered design")
    families = ("H75",) if reference_only else ("H75", "Specialists", "Shared Go2")
    output = root / ("reference" if reference_only else "comparison")
    panels = output / "panels"
    traces_dir = output / "traces"
    archives = []
    for row in specs:
        if row["cohort"] != "shared7101":
            continue
        raw = row["h75_reference"]
        archives.append(dict(row, cohort="h75", raw=raw, lane=row["h75_lane"]))
        if float(row["alpha"]) == 0:
            zero, = np.flatnonzero(load_contract(Path(raw))["settings"]["zero_middle"])
            archives.append(dict(row, cohort="zero", method="Zero", raw=raw, lane=zero, h75_lane=zero))
    expected = {"H75": load_contract(Path(archives[0]["raw"]))["identity"]["files"]["actor.npz"]}
    if not reference_only:
        for job in json.loads((root / "eval_jobs.json").read_text()):
            status = json.loads((Path(job["job"]) / "batch/status.json").read_text())
            if status["stage"] != "complete" or status["policy_steps"] != 500:
                raise ValueError(f"Incomplete specialist evaluation batch: {job['job']}")
        for tag in {r["cohort"] for r in specs}:
            run = Path(contract["shared_completed_run"]) if tag == "shared7101" else root / tag
            bundle = run.with_name(run.name + ("_final_bundle" if tag == "shared7101" else "_bundle"))
            identity = json.loads((bundle / "identity.json").read_text())
            expected[tag] = identity["files"]["actor.npz"]
            expected_updates = 150 if tag == "shared7101" else 50
            checkpoint = run / "training" / f"model_{expected_updates}.pt"
            if identity["native_checkpoint"]["sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                raise ValueError(f"{tag} exported bundle differs from the registered final checkpoint")
    panels.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(exist_ok=True)
    traces = read_traces(archives if reference_only else archives + specs, traces_dir)
    by_case = defaultdict(list)
    for trace in traces:
        r = trace["row"]
        required_actor = expected.get(r["cohort"], expected["H75"])
        if r["actor"] != required_actor:
            raise ValueError(f"Unexpected actor for {r['cohort']} at {r['raw']}")
        by_case[(r["condition"], r["group"])].append(trace)
    if set(by_case) != {(c, i) for c in CORE for i in range(101, 106)}:
        raise ValueError("The registered five conditions and five physical groups must all be present")

    gallery = []
    pairs = []
    by_setting = defaultdict(list)
    for case, items in sorted(by_case.items()):
        actual = Counter((d["row"]["family"], d["row"]["alpha"]) for d in items)
        wanted = Counter((family, a) for family in families for a in (0, .5, 1))
        wanted["Zero", 0] = 1
        if actual != wanted:
            raise ValueError(f"Missing or duplicate preference in {case}: {actual}")
        common_window(items)
        if not reference_only:
            by_actor = {(d["row"]["family"], d["row"]["alpha"]): d["row"] for d in items}
            for a in (0, .5, 1):
                specialist, shared = (by_actor[(f, a)] for f in ("Specialists", "Shared Go2"))
                pair = dict(condition=case[0], group=case[1], alpha=a, common_s=shared["common_s"],
                            specialist_raw=specialist["raw"], specialist_lane=specialist["lane"],
                            shared_raw=shared["raw"], shared_lane=shared["lane"])
                for field in ("full", "hold", "A", "B", "C", "common_speed_rmse_m_s", "common_path_rmse_m"):
                    pair["specialist_" + field] = specialist[field]
                    pair["shared_" + field] = shared[field]
                    pair["specialist_minus_shared_" + field] = specialist[field] - shared[field]
                pairs.append(pair)
        case_panels = diagnostics(case, items, panels, families)
        gallery.append(f"- {case[0]}, group {case[1]} " + " · ".join(
            f"[{label}](panels/{name})" for label, name in zip(("XY", *families), case_panels)))
        for d in items:
            r = d["row"]
            by_setting[(r["condition"], r["family"], r["alpha"])].append(r)
    rows = []
    for (condition, method, alpha), group in sorted(by_setting.items()):
        rows.append(dict(condition=condition, method=method, alpha=alpha, physical_groups=len(group),
                         full=sum(r["full"] for r in group), hold=sum(r["hold"] for r in group),
                         **{k: sum(r[k] for r in group) for k in ("A", "B", "C")},
                         common_path_rmse_m=np.mean([r["common_path_rmse_m"] for r in group]),
                         common_speed_rmse_m_s=np.mean([r["common_speed_rmse_m_s"] for r in group]),
                         minimum_common_s=min(r["common_s"] for r in group),
                         maximum_common_s=max(r["common_s"] for r in group)))
    csv_write(output / "settings.csv", rows)
    csv_write(output / "episodes.csv", [d["row"] for d in traces])
    lines = ["# " + ("Archived reference panel for the fixed-preference study" if reference_only else
                      "Fixed-preference and shared-policy comparison"), "",
             "[Recorded XY for the first group](panels/" + sorted(panels.glob("*_all_XY.png"))[0].name +
             ") · [All trajectories and terminal causes](episodes.csv) · [Per-setting counts and errors](settings.csv)", "",
             ("This reference panel reuses 75 H75 trajectories and 25 frozen lower-only trajectories." if reference_only else
              "The panel includes 75 specialist and 75 shared Go2 trajectories, plus 75 archived H75 and 25 lower-only references. Each specialist received 50 updates. Their total transition budget matches the 150 updates for the shared Go2 policy with the same initial weights."),
             "", "Each condition uses five previously examined physical groups. The following errors use the common observed prefix of all displayed methods and preferences within each group, then average those five group RMSE values with equal weight. Each panel has its own common window. Completion and hold use each original trajectory through its recorded endpoint. The lower-only policy takes zero middle correction. Reward plots rescore its recorded trajectory at each displayed preference.", "",
             "| Condition | Method | α | Full 10 s | Final hold | A | B | C | Cross-track RMSE (m) | Speed RMSE (m/s) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['condition']} | {r['method']} | {r['alpha']:g} | {r['full']}/5 | {r['hold']}/5 | {r['A']}/5 | {r['B']}/5 | {r['C']}/5 | {r['common_path_rmse_m']:.4f} | {r['common_speed_rmse_m_s']:.4f} |")
    if pairs:
        csv_write(output / "paired_differences.csv", pairs)
        lines.extend(["", "[75 same-preference paired differences, specialist minus shared](paired_differences.csv)"])
    lines.extend(["", "## Recorded XY and per-step rewards", "", *gallery, ""])
    (output / "README.md").write_text("\n".join(lines))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=["analyze", "specialist-reference", "specialist-comparison"])
    cli.add_argument("--output", type=Path, required=True, help="Registered run directory for the selected study")
    cli.add_argument("--specialists", type=Path, help="Completed dedicated/shared Go2 episodes_index.csv")
    args = cli.parse_args()
    torch.set_num_threads(1)
    if args.stage.startswith("specialist-"):
        specialist_panel(args.output, args.stage == "specialist-reference")
        return
    rows = csv_rows(args.output / "episodes_index.csv")
    if args.specialists:
        extras = csv_rows(args.specialists)
        if len(extras) != 150:
            raise ValueError("Specialist manifest must contain 75 dedicated and 75 shared rollouts")
        rows.extend(extras)
    for source in {Path(r["raw"]) for r in rows}:
        for name in ("contract.json", "trajectory.npz"):
            if not (source / name).is_file():
                raise FileNotFoundError(source / name)
    out = args.output / "analysis"
    panels = out / "panels"
    tracedir = out / "traces"
    panels.mkdir(parents=True, exist_ok=True)
    tracedir.mkdir(exist_ok=True)
    traces = read_traces(rows, tracedir)
    gallery, hv = summarize_frontier(traces, out)
    episodes = [d["row"] for d in traces]
    nominal = nominal_results([r for r in episodes if r["stage"] == "T2_nominal"])
    paired, summary = selector_results([r for r in episodes if r["stage"] == "T2_selector"])
    ridge_groups = defaultdict(list)
    for d in traces:
        if d["row"]["stage"] in ("T2_nominal", "T2_selector"):
            ridge_groups[(d["row"]["stage"], d["row"]["condition"], d["row"]["group"])].append(d)
    for (stage, condition, group), items in sorted(ridge_groups.items()):
        case = (condition, group)
        if stage == "T2_nominal":
            families = ("H75", "Speed cap")
        else:
            families = ("Pre-run selector", "Fixed H75", "Speed cap")
        gallery.extend(
            (stage + " " + case[0], case[1], f)
            for f in diagnostics((stage + "_" + case[0], case[1]), items, panels, families)
        )
    csv_write(out / "episodes.csv", episodes)
    csv_write(out / "nominal.csv", nominal)
    csv_write(out / "selector_pairs.csv", paired)
    csv_write(out / "selector_summary.csv", summary)
    csv_write(
        out / "panels.csv", [{"condition": c, "group": g, "panel": "panels/" + f} for c, g, f in gallery]
    )
    write_report(out, gallery, hv, nominal, summary, bool(args.specialists))
    (out / "receipt.json").write_text(
        json.dumps(
            {
                "episodes": len(episodes),
                "panels": len(gallery),
                "frontier_manifest": str(args.output / "episodes_index.csv"),
                "specialists": str(args.specialists) if args.specialists else None,
                "historical_h75_points": 13,
                "nominal_cases": 40,
                "prerun_cases": 176,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
