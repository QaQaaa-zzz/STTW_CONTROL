"""V3 six-episode review from saved 5 ms traces; no simulator or learner calls."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


CASES = ("main", "random")
METHODS = ("B0", "pi_alpha0", "pi_alpha1")
COLORS = {"B0": "black", "pi_alpha0": "tab:blue", "pi_alpha1": "tab:orange"}
COMPONENTS = ("speed", "steer", "heading", "roll", "roll_rate", "overspeed",
              "low_speed", "correction", "command_change")


def _array(trace, name, length, default=0.0):
    if name not in trace:
        return np.full(length, default)
    value = np.asarray(trace[name])
    if value.ndim == 0:
        return np.full(length, value.item())
    if len(value) != length:
        raise ValueError(f"{name} has {len(value)} entries, expected {length}")
    return value


def _active(trace):
    time = np.asarray(trace["time"], dtype=float)
    active = _array(trace, "active_tick", len(time), True).astype(bool)
    return time[active], {key: np.asarray(value)[active]
                          for key, value in trace.items()
                          if np.asarray(value).ndim >= 1 and len(value) == len(time)}


def _rmse(value):
    value = np.asarray(value, dtype=float)
    return float(np.sqrt(np.mean(value * value))) if len(value) and np.isfinite(value).all() else None


def _finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _window(trace, start, end, dt):
    time = np.asarray(trace["time"], dtype=float)
    active = _array(trace, "active_tick", len(time), True).astype(bool)
    mask = active & (time >= start - dt * 1e-4) & (time < end - dt * 1e-4)
    expected = round((end - start) / dt)
    selected = time[mask]
    if (len(selected) != expected or not np.allclose(
            selected, start + np.arange(expected) * dt, atol=dt * 1e-3, rtol=0)):
        return None
    raw = np.asarray(trace["limited_command"])[mask]
    speed_error = np.asarray(trace["actual_forward_speed"])[mask] - raw[:, 0]
    steer_error = np.asarray(trace["actual_delta"])[mask] - raw[:, 1]
    result = {
        "start_s": float(start), "end_s": float(end), "count": int(expected),
        "speed_rmse_m_s": _rmse(speed_error), "steer_rmse_rad": _rmse(steer_error),
        "heading_rmse_rad": _rmse(np.asarray(trace["e_psi_unwrapped"])[mask]),
        "actual_speed_mean_m_s": float(np.mean(np.asarray(trace["actual_forward_speed"])[mask])),
        "actual_abs_steer_mean_rad": float(np.mean(np.abs(np.asarray(trace["actual_delta"])[mask]))),
        "raw_abs_steer_mean_rad": float(np.mean(np.abs(raw[:, 1]))),
    }
    if any(result[key] is None or not np.isfinite(result[key]) for key in (
            "speed_rmse_m_s", "steer_rmse_rad", "heading_rmse_rad",
            "actual_speed_mean_m_s", "actual_abs_steer_mean_rad", "raw_abs_steer_mean_rad")):
        return None
    return result


def _recovery(trace, case, spec, state):
    e = spec["evaluation"]["claim_thresholds"]
    dt = spec["plant"]["control_dt_s"]
    return_time = 4.5 if case == "main" else 8.0
    time = np.asarray(trace["time"], dtype=float)
    active = _array(trace, "active_tick", len(time), True).astype(bool)
    raw = np.asarray(trace["limited_command"])
    eligible = np.flatnonzero(active & (time >= return_time - dt * 1e-4) &
                              (np.abs(raw[:, 1]) <= .005))
    result = {"raw_return_target_s": return_time, "raw_steer_settled_s": None,
              "first_recovery_s": None, "seconds_from_raw_steer_settled": None,
              "final_hold_met": False}
    if not len(eligible):
        return result
    start = int(eligible[0])
    result["raw_steer_settled_s"] = float(time[start])
    speed_error = np.abs(np.asarray(trace["actual_forward_speed"]) - raw[:, 0])
    steer_error = np.abs(np.asarray(trace["actual_delta"]) - raw[:, 1])
    heading_error = np.abs(np.asarray(trace["e_psi_unwrapped"]))
    good = active & (heading_error <= e["recovery_heading_rad"]) & (
        speed_error <= e["recovery_speed_error_m_s"]) & (
        steer_error <= e["recovery_steer_error_rad"])
    hold = round(e["recovery_hold_s"] / dt)
    for i in range(start, len(time) - hold + 1):
        if np.all(good[i:i + hold]) and np.allclose(
                time[i:i + hold], time[i] + np.arange(hold) * dt,
                atol=dt * 1e-3, rtol=0):
            result["first_recovery_s"] = float(time[i])
            result["seconds_from_raw_steer_settled"] = float(time[i] - time[start])
            break
    # An earlier transient recovery cannot substitute for a safe final hold.
    if state == "complete" and len(time) >= hold:
        result["final_hold_met"] = bool(np.all(good[-hold:]))
    return result


def analyze_trace(trace, spec, case):
    """Analyze one observed prefix. Missing fixed windows remain ``None``."""
    if case not in CASES:
        raise ValueError(f"unknown review case: {case}")
    required = ("time", "limited_command", "actual_forward_speed", "actual_delta",
                "phi", "e_psi_unwrapped", "scored_tick_reward")
    missing = [key for key in required if key not in trace]
    if missing:
        return {"state": "invalid", "reason": f"missing fields: {', '.join(missing)}"}
    try:
        time, values = _active(trace)
        n = len(time)
        dt = spec["plant"]["control_dt_s"]
        total = round(spec["commands"]["episode_seconds"] / dt)
        if n == 0 or not np.isfinite(time).all() or not np.allclose(
                time, np.arange(n) * dt, atol=dt * 1e-3, rtol=0):
            return {"state": "invalid", "reason": "active ticks are not a contiguous 5 ms prefix"}
        if np.asarray(values["limited_command"]).shape != (n, 2):
            return {"state": "invalid", "reason": "limited_command must have shape [N,2]"}
        for key in required[2:]:
            if np.asarray(values[key]).shape != (n,):
                return {"state": "invalid", "reason": f"{key} must have shape [N]"}
        physical = bool(np.any(_array(values, "physical_failure", n).astype(bool)) or
                        np.any(_array(values, "nonfinite", n).astype(bool)))
        fault = bool(np.any(_array(values, "policy_fault", n).astype(bool)))
        task_end = bool(np.any(_array(values, "finite_task_end", n).astype(bool)))
        if fault:
            state = "policy_fault"
        elif physical:
            state = "physical_failure"
        elif n == total and task_end:
            state = "complete"
        else:
            state = "partial"
        result = {
            "state": state, "observed_ticks": n, "declared_ticks": total,
            "last_observed_time_s": float(time[-1]),
            "checkpoint_update": int(np.asarray(trace["checkpoint_update"]).item())
            if "checkpoint_update" in trace and np.asarray(trace["checkpoint_update"]).ndim == 0 else None,
            "physical_failure": physical, "policy_fault": fault,
            "finite_task_end": task_end,
            "peak_roll_abs_rad": _finite_float(np.max(np.abs(values["peak_roll"])))
                                  if "peak_roll" in values else _finite_float(np.max(np.abs(values["phi"]))),
            "working_roll_exceed_seconds": float(np.sum(np.abs(values["phi"]) >
                                                       spec["limits"]["working_roll_rad"]) * dt),
            "reward_sum": _finite_float(np.sum(values["scored_tick_reward"])),
            "main_conflict": _window(values, 2.5, 4.5, dt) if case == "main" else None,
            "last_two_seconds": _window(values, 14.0, 16.0, dt),
        }
        chi = _array(values, "chi", n)
        gate = _array(values, "g", n)
        cap = _array(values, "cap_fraction", n)
        phase_masks = {"ordinary": (chi <= 0) & (gate <= 0),
                       "conflict": (chi > 0) & (gate <= 0), "recovery": gate > 0}
        result["cost_cap"] = {name: {"ticks": int(np.sum(mask)),
                                       "fraction": _finite_float(np.mean(cap[mask])) if np.any(mask) else None}
                              for name, mask in phase_masks.items()}
        result["reward_resolution_warning"] = any(
            result["cost_cap"][name]["fraction"] is not None and
            result["cost_cap"][name]["fraction"] > .05 for name in ("ordinary", "recovery"))
        result["costs"] = {}
        result["phase"] = {}
        raw = np.asarray(values["limited_command"])
        ev = np.asarray(values["actual_forward_speed"]) - raw[:, 0]
        ed = np.asarray(values["actual_delta"]) - raw[:, 1]
        for name, mask in phase_masks.items():
            result["phase"][name] = {
                "ticks": int(np.sum(mask)),
                "speed_rmse_m_s": _rmse(ev[mask]),
                "steer_rmse_rad": _rmse(ed[mask]),
                "heading_rmse_rad": _rmse(np.asarray(values["e_psi_unwrapped"])[mask]),
                "raw_cost_integral": _finite_float(np.sum(values["raw_cost"][mask]) * dt)
                if "raw_cost" in values else None,
                "effective_cost_integral": _finite_float(np.sum(values["effective_cost"][mask]) * dt)
                if "effective_cost" in values else None,
            }
        for prefix in ("raw_cost", "effective_cost"):
            if prefix in values:
                result["costs"][prefix] = _finite_float(np.sum(values[prefix]) * dt)
            for component in COMPONENTS:
                name = f"{prefix}_{component}"
                if name in values:
                    result["costs"][name] = _finite_float(np.sum(values[name]) * dt)
        result["clip_fraction"] = {
            name: float(np.mean(_array(values, name, n).astype(bool)))
            for name in ("residual_clipped", "final_command_clipped",
                         "reference_rate_clipped", "reference_reference_clipped")
            if name in values}
        result["recovery"] = _recovery(values, case, spec, state)
        return result
    except (ValueError, TypeError, IndexError, KeyError) as error:
        return {"state": "invalid", "reason": str(error)}


def _main_claims(methods, traces, spec):
    unavailable = {"main_preference": "unavailable", "reason": "complete same-window traces required"}
    if any(methods[m]["state"] != "complete" for m in METHODS):
        return unavailable
    windows = {m: methods[m]["main_conflict"] for m in METHODS}
    if any(w is None for w in windows.values()):
        return unavailable
    e = spec["evaluation"]["claim_thresholds"]
    b, a0, a1 = (windows[m] for m in METHODS)
    # A baseline that already tracks both requested channels safely offers no
    # observed conflict from which to infer a preference tradeoff.
    informative = not (b["speed_rmse_m_s"] <= e["alpha1_speed_rmse_max_m_s"] and
                       b["steer_rmse_rad"] <= e["alpha0_steer_rmse_max_rad"] and
                       methods["B0"]["peak_roll_abs_rad"] is not None and
                       methods["B0"]["peak_roll_abs_rad"] <= e["roll_peak_max_rad"])
    steer_ratio = a0["steer_rmse_rad"] / a1["steer_rmse_rad"] if a1["steer_rmse_rad"] > 0 else None
    speed_ratio = a1["speed_rmse_m_s"] / a0["speed_rmse_m_s"] if a0["speed_rmse_m_s"] > 0 else None
    baseline_speed = _window(traces["pi_alpha0"], 1.0, 1.5, spec["plant"]["control_dt_s"])
    drop = None
    sustained_drop = False
    if baseline_speed is not None:
        time = np.asarray(traces["pi_alpha0"]["time"])
        speed = np.asarray(traces["pi_alpha0"]["actual_forward_speed"])
        mask = (time >= 2.5) & (time < 4.5)
        deficits = baseline_speed["actual_speed_mean_m_s"] - speed[mask]
        drop = float(np.mean(deficits))
        length = round(e["speed_drop_hold_s"] / spec["plant"]["control_dt_s"])
        sustained_drop = bool(np.any(np.convolve(
            (deficits >= e["alpha0_actual_speed_drop_m_s"]).astype(int),
            np.ones(length, dtype=int), mode="valid") == length))
    steer_reduction = a1["raw_abs_steer_mean_rad"] - a1["actual_abs_steer_mean_rad"]
    safety = all(not methods[m]["physical_failure"] and
                 methods[m]["peak_roll_abs_rad"] is not None and
                 methods[m]["peak_roll_abs_rad"] <= e["roll_peak_max_rad"] for m in METHODS)
    directional = a0["steer_rmse_rad"] < a1["steer_rmse_rad"] and a1["speed_rmse_m_s"] < a0["speed_rmse_m_s"]
    separation = (a1["steer_rmse_rad"] - a0["steer_rmse_rad"] >= e["steer_rmse_separation_min_rad"] and
                  a0["speed_rmse_m_s"] - a1["speed_rmse_m_s"] >= e["speed_rmse_separation_min_m_s"])
    quality = (a0["steer_rmse_rad"] <= e["alpha0_steer_rmse_max_rad"] and
               a1["speed_rmse_m_s"] <= e["alpha1_speed_rmse_max_m_s"])
    ratios = steer_ratio is not None and speed_ratio is not None and (
        steer_ratio <= e["primary_ratio_max"] and speed_ratio <= e["primary_ratio_max"])
    recovery = all(methods[m]["recovery"]["final_hold_met"] for m in METHODS)
    return {"main_preference": "not_informative_case" if not informative else
            ("criteria_met" if all((directional, separation, quality, ratios, sustained_drop,
                                    drop is not None and drop >= e["alpha0_actual_speed_drop_m_s"],
                                    steer_reduction >= e["alpha1_mean_steer_reduction_rad"],
                                    safety, recovery)) else "criteria_not_met"),
            "informative_baseline": informative, "directional_errors": directional,
            "absolute_separation": separation, "primary_error_ratios": ratios,
            "absolute_quality": quality, "common_safety": safety,
            "final_hold_all_methods": recovery,
            "alpha0_speed_drop_mean_m_s": drop,
            "alpha0_speed_drop_sustained": sustained_drop,
            "alpha1_mean_steer_reduction_rad": float(steer_reduction),
            "steer_rmse_ratio_alpha0_over_alpha1": steer_ratio,
            "speed_rmse_ratio_alpha1_over_alpha0": speed_ratio}


def _step_csv(path, trace):
    time, values = _active(trace)
    n = len(time)
    columns = {"time": time}
    for name, value in values.items():
        if name == "time":
            continue
        value = np.asarray(value)
        if value.ndim == 1:
            columns[name] = value
        elif value.ndim > 1:
            flat = value.reshape(n, -1)
            for i in range(flat.shape[1]):
                columns[f"{name}_{i}"] = flat[:, i]
    if "scored_tick_reward" in columns:
        columns["cumulative_actual_reward"] = np.cumsum(columns["scored_tick_reward"])
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))


def _plot_case(out, case, traces, spec):
    if not traces:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(14, 17), constrained_layout=True)
    plots = axes.ravel()
    for method, original in traces.items():
        time, x = _active(original)
        if len(time) == 0:
            continue
        color = COLORS[method]
        for ax, name in ((plots[0], "actual_forward_speed"), (plots[1], "actual_delta"),
                         (plots[2], "phi"), (plots[3], "e_psi_unwrapped")):
            if name in x:
                ax.plot(time, x[name], color=color, label=method)
        if "applied_residual" in x:
            plots[4].plot(time, x["applied_residual"][:, 0], color=color, label=method)
            plots[5].plot(time, x["applied_residual"][:, 1], color=color, label=method)
        if "actual_xy" in x:
            plots[6].plot(x["actual_xy"][:, 0], x["actual_xy"][:, 1], color=color, label=method)
            if np.any(_array(x, "physical_failure", len(time)).astype(bool)):
                plots[6].scatter(*x["actual_xy"][-1], color=color, marker="x", s=50)
        if "scored_tick_reward" in x:
            plots[7].plot(time, np.cumsum(x["scored_tick_reward"]), color=color, label=method)
        if np.any(_array(x, "physical_failure", len(time)).astype(bool)):
            for ax in (plots[0], plots[1], plots[2], plots[3], plots[4], plots[5], plots[7]):
                ax.axvline(time[-1], color=color, linestyle=":", alpha=.6)
    source = traces.get("B0", next(iter(traces.values())))
    time, x = _active(source)
    if len(time):
        if "limited_command" in x:
            plots[0].plot(time, x["limited_command"][:, 0], "--", color="gray", label="raw command")
            plots[1].plot(time, x["limited_command"][:, 1], "--", color="gray", label="raw command")
        if "reference_xy" in x:
            plots[6].plot(x["reference_xy"][:, 0], x["reference_xy"][:, 1], "--",
                          color="gray", label="reference (display only)")
    plots[2].axhline(spec["limits"]["working_roll_rad"], color="red", linestyle="--", linewidth=.7)
    plots[2].axhline(-spec["limits"]["working_roll_rad"], color="red", linestyle="--", linewidth=.7)
    labels = ("Forward speed (m/s)", "Steer angle (rad)", "Roll (rad)",
              "Heading debt (rad)", "Front bounded additive residual (rad/s)",
              "Rear bounded additive residual (rad/s)", "XY (m); no position criterion",
              "Cumulative actual scored reward")
    for ax, label in zip(plots, labels):
        ax.set_title(label)
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize=8)
        if ax is not plots[6]:
            ax.set_xlabel("Time (s)")
    plots[6].set_aspect("equal", adjustable="datalim")
    plots[6].set_xlabel("X (m)")
    plots[6].set_ylabel("Y (m)")
    if case == "main":
        for ax in plots[:6]:
            ax.axvspan(2.5, 4.5, color="gray", alpha=.1)
    fig.suptitle(f"V3 {case} review | observed physical prefixes | methods actually available")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{case}_comparison.{suffix}", dpi=150)
    plt.close(fig)


def generate_report(output, spec):
    """Read review NPZ files, write per-step CSV, plots, metrics and report."""
    root = Path(output)
    review = root / "review"
    review.mkdir(parents=True, exist_ok=True)
    result = {"schema": "sttw_direct_command_v3_review", "cases": {}, "claims": {}}
    all_traces = {}
    for case in CASES:
        case_dir = review / case
        methods = {}
        traces = {}
        for method in METHODS:
            path = case_dir / f"{method}.npz"
            if not path.exists():
                methods[method] = {"state": "missing"}
                continue
            try:
                with np.load(path, allow_pickle=False) as archive:
                    trace = {name: archive[name] for name in archive.files}
                methods[method] = analyze_trace(trace, spec, case)
                if methods[method]["state"] != "invalid":
                    traces[method] = trace
                    _step_csv(case_dir / f"{method}_steps.csv", trace)
            except (OSError, ValueError, KeyError) as error:
                methods[method] = {"state": "invalid", "reason": str(error)}
        result["cases"][case] = {"methods": methods,
                                 "all_complete": all(methods[m]["state"] == "complete" for m in METHODS)}
        all_traces[case] = traces
        _plot_case(review, case, traces, spec)
    main = result["cases"]["main"]["methods"]
    result["claims"] = _main_claims(main, all_traces["main"], spec)
    result["claims"]["random_evidence"] = (
        "complete_three_method_case" if result["cases"]["random"]["all_complete"] else "incomplete")
    (review / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Direct command V3 review", "",
             f"Main preference: **{result['claims']['main_preference']}**. ",
             "Comparative criteria require complete, matched 5 ms windows for B0, α0 and α1.",
             "The main conflict window is [2.5, 4.5) s; final quality uses [14, 16) s.",
             "Recovery starts when the raw steer reaches |δc|≤0.005 after the final return target.",
             "Physical failure invalidates final hold. XY is displayed without a position criterion.", ""]
    for case in CASES:
        lines += [f"## {case}", "", "| Method | Status | Observed ticks | Physical failure | Final hold |",
                  "|---|---|---:|---|---|"]
        for method in METHODS:
            metric = result["cases"][case]["methods"][method]
            lines.append(f"| {method} | {metric['state']} | {metric.get('observed_ticks', '—')} | "
                         f"{metric.get('physical_failure', '—')} | "
                         f"{metric.get('recovery', {}).get('final_hold_met', '—')} |")
        lines += ["", "| Method | Conflict speed RMSE (m/s) | Conflict steer RMSE (rad) | "
                  "Last 2 s speed RMSE (m/s) | Last 2 s steer RMSE (rad) | "
                  "First recovery (s) | Peak |φ| (rad) |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            metric = result["cases"][case]["methods"][method]
            conflict = metric.get("main_conflict") or {}
            tail = metric.get("last_two_seconds") or {}
            recovery = metric.get("recovery") or {}
            def shown(value):
                return "—" if value is None else str(round(value, 5))
            lines.append(f"| {method} | {shown(conflict.get('speed_rmse_m_s'))} | "
                         f"{shown(conflict.get('steer_rmse_rad'))} | "
                         f"{shown(tail.get('speed_rmse_m_s'))} | "
                         f"{shown(tail.get('steer_rmse_rad'))} | "
                         f"{shown(recovery.get('first_recovery_s'))} | "
                         f"{shown(metric.get('peak_roll_abs_rad'))} |")
        lines.append("")
        if (review / f"{case}_comparison.png").exists():
            lines += [f"[Metrics](metrics.json) · [Plot PNG]({case}_comparison.png) · "
                      f"[Plot PDF]({case}_comparison.pdf)", ""]
    lines += ["A complete six-episode review is one prepared state and one random sequence; "
              "it does not establish generalization or hard safety.", ""]
    (review / "REPORT.md").write_text("\n".join(lines))
    return result
