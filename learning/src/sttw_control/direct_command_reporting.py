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
            for name in ("reference_rate_clipped", "reference_reference_clipped")
            if name in values}
        for name in ("residual_clipped", "final_command_clipped"):
            result["clip_fraction"][f"logged_{name}"] = (
                float(np.mean(_array(values, name, n).astype(bool))) if name in values else None)
        scales = np.asarray((spec["limits"]["steer_residual_rad_s"],
                             spec["limits"]["rear_residual_rad_s"]), dtype=float)
        requested = np.asarray(values["requested_residual"], dtype=float) if "requested_residual" in values else None
        if requested is not None and requested.shape != (n, 2):
            return {"state": "invalid", "reason": "requested_residual must have shape [N,2]"}
        result["clip_fraction"]["residual_clipped"] = (
            float(np.mean(np.any(np.abs(requested) > scales, axis=1)))
            if requested is not None and np.isfinite(requested).all() else None)
        prelimit = np.asarray(values["u_prelimit"], dtype=float) if "u_prelimit" in values else None
        final = np.asarray(values["final_command"], dtype=float) if "final_command" in values else None
        if any(array is not None and array.shape != (n, 2) for array in (prelimit, final)):
            return {"state": "invalid", "reason": "u_prelimit/final_command must have shape [N,2]"}
        if prelimit is not None and final is not None and np.isfinite(prelimit).all() and np.isfinite(final).all():
            limits = np.asarray((spec["limits"]["final_steer_rate_rad_s"],
                                 spec["limits"]["final_rear_rate_rad_s"]), dtype=float)
            # Float32 divide/multiply and additions may differ by a few ulps;
            # count a physical command limit or a material actuator change.
            tolerance = np.maximum(1e-5, 8 * np.finfo(np.float32).eps *
                                   np.maximum(np.abs(prelimit), limits))
            constrained = (np.abs(prelimit) > limits) | (
                np.abs(final - prelimit) > tolerance)
            result["clip_fraction"]["final_command_clipped"] = float(np.mean(np.any(constrained, axis=1)))
        else:
            result["clip_fraction"]["final_command_clipped"] = None
        result["clip_fraction_method"] = (
            "residual: any abs(requested_residual)>[1.5,10]; final: command exceeds [3,60] "
            "or final differs materially from u_prelimit using float32 rounding tolerance; "
            "logged flags retained separately")
        # The final-command difference includes downstream actuator clipping.
        # Keep it distinct from the bounded additive request at the ECBC port.
        result["residuals"] = {}
        for name, source in (("realized_final_command_change", "actual_normalized_residual"),
                             ("bounded_additive_request", "applied_residual")):
            if source not in values:
                result["residuals"][name] = None
                continue
            residual = np.asarray(values[source], dtype=float)
            if residual.shape != (n, 2):
                return {"state": "invalid", "reason": f"{source} must have shape [N,2]"}
            if source == "actual_normalized_residual":
                residual = residual * np.asarray((spec["limits"]["steer_residual_rad_s"],
                                                  spec["limits"]["rear_residual_rad_s"]))
            result["residuals"][name] = {
                "front_rmse_rad_s": _rmse(residual[:, 0]),
                "rear_rmse_rad_s": _rmse(residual[:, 1]),
                "front_peak_abs_rad_s": _finite_float(np.max(np.abs(residual[:, 0]))),
                "rear_peak_abs_rad_s": _finite_float(np.max(np.abs(residual[:, 1]))),
            }
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
    recovery = all(methods[m]["recovery"]["first_recovery_s"] is not None
                   for m in ("pi_alpha0", "pi_alpha1"))
    return {"main_preference": "not_informative_case" if not informative else
            ("criteria_met" if all((directional, separation, quality, ratios, sustained_drop,
                                    steer_reduction >= e["alpha1_mean_steer_reduction_rad"],
                                    safety, recovery)) else "criteria_not_met"),
            "informative_baseline": informative, "directional_errors": directional,
            "absolute_separation": separation, "primary_error_ratios": ratios,
            "absolute_quality": quality, "common_safety": safety,
            "qualifying_recovery_both_policies": recovery,
            "final_hold_by_method": {m: methods[m]["recovery"]["final_hold_met"]
                                     for m in METHODS},
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


def _huber(value):
    magnitude = np.abs(value)
    return np.where(magnitude <= 1., value * value, 2. * magnitude - 1.)


def rescore_baseline_alpha1(trace, spec):
    """Rescore one B0 physical prefix under alpha=1 without replaying physics.

    Audits the original alpha=0 component/score identity first. The 20 ms
    physical-failure replacement is alpha independent and replaces all normal
    tick scores in that policy interval, including the failing tick.
    """
    time, x = _active(trace)
    n = len(time)
    dt = spec["plant"]["control_dt_s"]
    if not n or not np.allclose(time, np.arange(n) * dt, atol=dt * 1e-3, rtol=0):
        raise ValueError("B0 trace is not a contiguous active 5 ms prefix")
    required = ("limited_command", "actual_forward_speed", "actual_delta", "chi", "g",
                "scored_tick_reward") + tuple(f"raw_cost_{name}" for name in COMPONENTS)
    missing = [name for name in required if name not in x]
    if missing:
        raise ValueError(f"B0 rescore missing {', '.join(missing)}")
    raw = np.asarray(x["limited_command"], dtype=float)
    if raw.shape != (n, 2):
        raise ValueError("B0 limited_command must have shape [N,2]")
    r = spec["reward"]
    chi = np.asarray(x["chi"], dtype=float)
    gate = np.asarray(x["g"], dtype=float)
    ev = np.asarray(x["actual_forward_speed"], dtype=float) - raw[:, 0]
    ed = np.asarray(x["actual_delta"], dtype=float) - raw[:, 1]
    if not np.isfinite(np.stack((chi, gate, ev, ed))).all():
        raise ValueError("B0 rescore input contains nonfinite values")
    gap = r["priority_high"] - r["priority_low"]
    speed_base = _huber(ev / r["speed_error_scale_m_s"])
    steer_base = _huber(ed / r["steer_error_scale_rad"])
    alpha0_speed = (r["priority_high"] - gap * chi) * speed_base
    alpha0_steer = ((1. - gate) * r["priority_high"] +
                    gate * r["recovery_steer_weight"]) * steer_base
    raw_parts = {name: np.asarray(x[f"raw_cost_{name}"], dtype=float)
                 for name in COMPONENTS}
    if any(part.shape != (n,) or not np.isfinite(part).all() for part in raw_parts.values()):
        raise ValueError("B0 raw component arrays must be finite [N] vectors")
    if not np.allclose(raw_parts["speed"], alpha0_speed, rtol=2e-4, atol=2e-4) or not np.allclose(
            raw_parts["steer"], alpha0_steer, rtol=2e-4, atol=2e-4):
        raise ValueError("B0 original alpha0 speed/steer components do not reconstruct")
    original_raw = sum(raw_parts.values())
    if "raw_cost" in x and not np.allclose(x["raw_cost"], original_raw, rtol=2e-4, atol=2e-4):
        raise ValueError("B0 original raw cost does not equal component sum")
    original_normal = -r["scale"] * dt * np.minimum(original_raw, r["cost_rate_cap"])
    original_score = np.asarray(x["scored_tick_reward"], dtype=float)
    failed = np.flatnonzero(_array(x, "physical_failure", n).astype(bool) |
                            _array(x, "nonfinite", n).astype(bool))
    if len(failed) and int(failed[0]) != n - 1:
        raise ValueError("B0 failure must end the active physical prefix")
    failure_start = (int(failed[0]) // spec["plant"]["control_ticks_per_action"] *
                     spec["plant"]["control_ticks_per_action"]) if len(failed) else n
    if not np.allclose(original_score[:failure_start], original_normal[:failure_start],
                       rtol=2e-4, atol=2e-4):
        raise ValueError("B0 original alpha0 scored reward does not reconstruct")
    if len(failed):
        policy_index = int(failed[0]) // spec["plant"]["control_ticks_per_action"]
        remaining = round(spec["commands"]["episode_seconds"] / spec["plant"]["policy_dt_s"]) - policy_index
        gamma = spec["ppo"]["gamma"]
        replacement = (-r["failure_extra_penalty"] -
                       r["scale"] * spec["plant"]["policy_dt_s"] * r["cost_rate_cap"] *
                       (1. - gamma ** remaining) / (1. - gamma))
        if not np.allclose(original_score[failure_start:-1], 0., atol=2e-4) or not np.isclose(
                original_score[-1], replacement, rtol=2e-4, atol=2e-3):
            raise ValueError("B0 failure replacement does not match frozen 20 ms rule")
    parts = dict(raw_parts)
    parts["speed"] = r["priority_high"] * speed_base
    parts["steer"] = ((1. - gate) * (r["priority_high"] - gap * chi) +
                      gate * r["recovery_steer_weight"]) * steer_base
    total = sum(parts.values())
    factor = np.minimum(1., r["cost_rate_cap"] / np.maximum(total, 1e-12))
    score = -r["scale"] * dt * np.minimum(total, r["cost_rate_cap"])
    if len(failed):
        score[failure_start:] = 0.
        score[-1] = original_score[-1]
    result = dict(x)
    for name, part in parts.items():
        result[f"raw_cost_{name}"] = part
        result[f"effective_cost_{name}"] = part * factor
    result.update(raw_cost=total, effective_cost=np.minimum(total, r["cost_rate_cap"]),
                  cap_fraction=(total > r["cost_rate_cap"]).astype(float),
                  scored_tick_reward=score, cumulative_actual_reward=np.cumsum(score),
                  alpha=np.asarray(1.))
    return result


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
        if "actual_normalized_residual" in x:
            realized = x["actual_normalized_residual"] * np.asarray((
                spec["limits"]["steer_residual_rad_s"], spec["limits"]["rear_residual_rad_s"]))
            plots[4].plot(time, realized[:, 0], color=color, label=method)
            plots[5].plot(time, realized[:, 1], color=color, label=method)
        if "actual_xy" in x:
            plots[6].plot(x["actual_xy"][:, 0], x["actual_xy"][:, 1], color=color, label=method)
            if np.any(_array(x, "physical_failure", len(time)).astype(bool)):
                plots[6].scatter(*x["actual_xy"][-1], color=color, marker="x", s=50)
        if np.any(_array(x, "physical_failure", len(time)).astype(bool)):
            for ax in (plots[0], plots[1], plots[2], plots[3], plots[4], plots[5]):
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
        for name, color in (("chi", "tab:red"), ("g", "tab:green")):
            if name in x:
                plots[7].plot(time, x[name], color=color, label=name)
    plots[2].axhline(spec["limits"]["working_roll_rad"], color="red", linestyle="--", linewidth=.7)
    plots[2].axhline(-spec["limits"]["working_roll_rad"], color="red", linestyle="--", linewidth=.7)
    labels = ("Forward speed (m/s)", "Steer angle (rad)", "Roll (rad)",
              "Heading debt (rad)", "Front realized final-command change (rad/s)",
              "Rear realized final-command change (rad/s)", "XY (m); no position criterion",
              "Raw conflict χ / recovery weight g")
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


def _plot_rewards(out, case, traces, spec, baseline_alpha1=None):
    """Plot each policy against B0 scored with the same alpha weights."""
    if not traces:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    series = []
    for alpha, method in ((0, "pi_alpha0"), (1, "pi_alpha1")):
        if method in traces:
            series.append((method, traces[method], COLORS[method], "-"))
        baseline = traces.get("B0") if alpha == 0 else baseline_alpha1
        if baseline is not None:
            series.append((f"B0 α{alpha}", baseline, COLORS[method], "--"))
    for label, original, color, linestyle in series:
        time, x = _active(original)
        if not len(time) or "scored_tick_reward" not in x:
            continue
        reward = np.asarray(x["scored_tick_reward"])
        cumulative = np.cumsum(reward)
        endpoint_marker = ("x" if np.any(_array(x, "physical_failure", len(time)).astype(bool))
                           else "o" if np.any(_array(x, "finite_task_end", len(time)).astype(bool))
                           else "s")
        for ax, values in zip(axes, (reward, cumulative)):
            ax.plot(time, values, color=color, linestyle=linestyle, label=label)
            ax.plot(time[-1], values[-1], marker=endpoint_marker,
                    color=color, markersize=4)
    axes[0].set_title("Actual scored reward per 5 ms tick; failure replacement included")
    axes[1].set_title("Cumulative actual scored reward; x failure, square partial, circle finite end")
    for ax in axes:
        if case == "main":
            ax.axvspan(2.5, 4.5, color="gray", alpha=.1)
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend()
    fig.suptitle(f"V3 {case} | solid policy, dashed same-alpha B0 | observed prefixes")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{case}_rewards.{suffix}", dpi=150)
    plt.close(fig)


def _plot_components(out, case, method, original, spec):
    """Raw/effective per-component costs and cumulative observed integrals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time, x = _active(original)
    if not len(time):
        return False
    available = {prefix: [component for component in COMPONENTS
                          if f"{prefix}_{component}" in x]
                 for prefix in ("raw_cost", "effective_cost")}
    if not any(available.values()):
        return False
    dt = spec["plant"]["control_dt_s"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for row, prefix in enumerate(("raw_cost", "effective_cost")):
        for component in available[prefix]:
            cost = np.asarray(x[f"{prefix}_{component}"])
            axes[row, 0].plot(time, cost, label=component)
            axes[row, 1].plot(time, np.cumsum(cost * dt), label=component)
        label = "Uncapped raw" if row == 0 else "Capped effective"
        axes[row, 0].set_title(f"{label} component cost rate (cost/s)")
        axes[row, 1].set_title(f"{label} cumulative component cost (cost)")
        for ax in axes[row]:
            if case == "main":
                ax.axvspan(2.5, 4.5, color="gray", alpha=.1)
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=.25)
            if ax.lines:
                ax.legend(fontsize=8, ncol=3)
    endpoint = "physical failure" if np.any(_array(x, "physical_failure", len(time)).astype(bool)) else (
        "finite task end" if np.any(_array(x, "finite_task_end", len(time)).astype(bool)) else "partial prefix")
    fig.suptitle(f"V3 {case} {method} | {endpoint} at {time[-1]:.3f} s | "
                 "normal interval costs; failure reward replacement shown in scored reward plot")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{case}_{method}_components.{suffix}", dpi=150)
    plt.close(fig)
    return True


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
        rescored = None
        if "B0" in traces:
            try:
                rescored = rescore_baseline_alpha1(traces["B0"], spec)
                np.savez_compressed(case_dir / "B0_rescored_alpha1.npz", **rescored)
                _step_csv(case_dir / "B0_rescored_alpha1_steps.csv", rescored)
                result["cases"][case]["baseline_alpha1_rescore"] = {
                    "state": "audited", "source": f"{case}/B0.npz",
                    "derived": f"{case}/B0_rescored_alpha1.npz"}
            except (ValueError, KeyError) as error:
                result["cases"][case]["baseline_alpha1_rescore"] = {
                    "state": "unavailable", "reason": str(error)}
        else:
            result["cases"][case]["baseline_alpha1_rescore"] = {
                "state": "unavailable", "reason": "B0 physical trace missing or invalid"}
        _plot_case(review, case, traces, spec)
        _plot_rewards(review, case, traces, spec, rescored)
        for method, trace in traces.items():
            _plot_components(review, case, method, trace, spec)
    main = result["cases"]["main"]["methods"]
    result["claims"] = _main_claims(main, all_traces["main"], spec)
    result["claims"]["random_evidence"] = (
        "complete_three_method_case" if result["cases"]["random"]["all_complete"] else "incomplete")
    (review / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Direct command V3 review", "",
             ("B0 = ECBC+ESO plus the pinned frozen residual Actor (lower alpha=1); "
              "upper alpha0/1 share that same lower controller. The lower causal governed path "
              "is a transfer adapter, not a reproduction of the source bend task."
              if spec.get('lower_controller') else "B0 = original ECBC+ESO."), "",
             f"Main preference: **{result['claims']['main_preference']}**. ",
             "Comparative criteria require complete, matched 5 ms windows for B0, α0 and α1.",
             "The main conflict window is [2.5, 4.5) s; final quality uses [14, 16) s.",
             "Recovery timing starts when the raw steer reaches |δc|≤0.005 after the final return target; "
             "the criterion is the first qualifying 0.5 s hold for each learned method.",
             "Final-window hold is descriptive. Physical failure invalidates it. "
             "XY is displayed without a position criterion.",
             "Reward plots compare each learned alpha with B0 rescored under that same alpha. "
             "B0 alpha1 uses the same saved physical trace, with audited component reweighting "
             "and the same whole-policy-interval physical-failure replacement.", ""]
    for case in CASES:
        rescore_status = result["cases"][case]["baseline_alpha1_rescore"]
        lines += [f"## {case}", "", "| Method | Status | Observed ticks | Physical failure | Final hold (descriptive) |",
                  "|---|---|---:|---|---|"]
        for method in METHODS:
            metric = result["cases"][case]["methods"][method]
            lines.append(f"| {method} | {metric['state']} | {metric.get('observed_ticks', '—')} | "
                         f"{metric.get('physical_failure', '—')} | "
                         f"{metric.get('recovery', {}).get('final_hold_met', '—')} |")
        lines += ["", f"B0 alpha1 rescore: **{rescore_status['state']}**" +
                  (f" ({rescore_status['reason']})" if "reason" in rescore_status else "."), ""]
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
        lines += ["", "Residual peaks below use rad/s. Realized means final command minus zero-residual final command, "
                  "after actuator clipping; bounded request is before that clip.", "",
                  "| Method | Realized front/rear peak | Bounded request front/rear peak |",
                  "|---|---:|---:|"]
        for method in METHODS:
            residuals = result["cases"][case]["methods"][method].get("residuals") or {}
            realized = residuals.get("realized_final_command_change") or {}
            requested = residuals.get("bounded_additive_request") or {}
            lines.append(f"| {method} | {shown(realized.get('front_peak_abs_rad_s'))} / "
                         f"{shown(realized.get('rear_peak_abs_rad_s'))} | "
                         f"{shown(requested.get('front_peak_abs_rad_s'))} / "
                         f"{shown(requested.get('rear_peak_abs_rad_s'))} |")
        lines += ["", "Clip fractions are reconstructed from saved requested residuals and final commands. "
                  "The original logged flags remain visible because exact float inequality can flag rounding alone.", "",
                  "| Method | Residual true / logged | Final command true / logged |",
                  "|---|---:|---:|"]
        for method in METHODS:
            clips = result["cases"][case]["methods"][method].get("clip_fraction") or {}
            lines.append(f"| {method} | {shown(clips.get('residual_clipped'))} / "
                         f"{shown(clips.get('logged_residual_clipped'))} | "
                         f"{shown(clips.get('final_command_clipped'))} / "
                         f"{shown(clips.get('logged_final_command_clipped'))} |")
        lines.append("")
        if (review / f"{case}_comparison.png").exists():
            lines += [f"[Metrics](metrics.json) · [Plot PNG]({case}_comparison.png) · "
                      f"[Plot PDF]({case}_comparison.pdf) · "
                      f"[Per-tick and cumulative reward]({case}_rewards.png) "
                      f"([PDF]({case}_rewards.pdf))", ""]
        for method in METHODS:
            if (review / f"{case}_{method}_components.png").exists():
                lines.append(f"[{method} raw/effective cost components]({case}_{method}_components.png) "
                             f"([PDF]({case}_{method}_components.pdf))")
        lines.append("")
    lines += ["A complete six-episode review is one prepared state and one random sequence; "
              "it does not establish generalization or hard safety.", ""]
    (review / "REPORT.md").write_text("\n".join(lines))
    return result
