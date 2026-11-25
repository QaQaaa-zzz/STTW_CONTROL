"""Compare offline roll warnings at the post-step timestamps in 760 frozen runs."""
import argparse
import csv
import json
import platform
from collections import defaultdict
from pathlib import Path
from time import perf_counter_ns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sttw_control.roll_warning_inputs import B_NOM, CTRL_CFG, DT, PHI_C, euler_roll_rate
from sttw_control.margin_theory import recovery_margin_batch, recovery_outcome

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/evidence/pcrp_matched_controls/forensic"
COHORTS = ("H75", "H75_P1", "7202_u64", "7202_u160")
DETECTORS = ("angle", "rate", "time_to_bound", "stopping_margin")
LABELS = ("Roll angle", "Roll rate", "Time to bound", "Stopping margin")


def write_csv(path, rows):
    with path.open("w", newline="") as out:
        table = csv.DictWriter(out, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)


def scan(runs, output):
    """Episode ROC; operating thresholds use the full-window controls on this panel."""
    rolls = np.array([r["endpoint"] == "roll" for r in runs])
    survivors = np.array([r["endpoint"] == "full" for r in runs])
    ends = np.array([r["end_s"] for r in runs])
    env_hits = np.array([r["envelope_lead_ms"] > 0 for r in runs])
    fp_budget = np.count_nonzero(survivors & env_hits)
    curves, comparisons = [], []
    for j, name in enumerate(DETECTORS):
        maxima = np.array([r["prefix"][j][-1] for r in runs])
        cap_threshold = np.sort(maxima[survivors])[-fp_budget-1]
        # Include fixed H<0 at zero and both ends of the empirical ROC.
        thresholds = np.unique(np.r_[np.nextafter(maxima.min(), -np.inf), 0, maxima])
        hits = np.full((len(runs), len(thresholds)), np.nan)
        for i, r in enumerate(runs):
            ticks = r["prefix"][j].searchsorted(thresholds, side="right")
            valid = ticks < len(r["times"])
            hits[i, valid] = r["times"][ticks[valid]]
        for k, threshold in enumerate(thresholds):
            valid = np.isfinite(hits[:, k])
            leads = (ends - hits[:, k]) * 1e3
            row = dict(detector=name, score_threshold=float(threshold),
                       roll_alerts=int((valid & rolls).sum()), full_alerts=int((valid & survivors).sum()),
                       other_alerts=int((valid & ~(rolls | survivors)).sum()),
                       median_lead_ms=float(np.nanmedian(leads[rolls])) if (valid & rolls).any() else None)
            curves.append(row)
            if threshold == cap_threshold or (name == "stopping_margin" and threshold == 0):
                comparisons.append(dict(setting="matched survivor budget" if threshold == cap_threshold else "H < 0",
                                        **row))
            if threshold == cap_threshold:
                for i, r in enumerate(runs):
                    r[name + "_lead_ms"] = leads[i]
    comparisons.append(dict(setting="frozen", detector="nominal_envelope", score_threshold=0,
                            roll_alerts=int((rolls & env_hits).sum()), full_alerts=int((survivors & env_hits).sum()),
                            other_alerts=int((~(rolls | survivors) & env_hits).sum()),
                            median_lead_ms=float(np.median([r["envelope_lead_ms"] for r in runs if r["endpoint"] == "roll" and r["envelope_lead_ms"] > 0]))))
    write_csv(output / "threshold_sweep.csv", curves)
    write_csv(output / "operating_points.csv", comparisons)
    return curves, comparisons


def crossing_query(r, index, when, states, speed, normal, times, terminal_side):
    state = states[index]
    valid = speed[index] >= .5 and abs(state[2]) <= .8 and abs(state[0]) < PHI_C
    result = recovery_outcome(speed[index], state, CTRL_CFG) if valid else {}
    predicts = result.get("outcome") == "unavoidable_nominal_limit"
    same_side = predicts and np.sign(state[1]) == terminal_side
    actual = (r["end_s"] - times[index])*1000
    predicted = result["event_s"]*1000 if predicts else np.nan
    return dict(cohort=r["cohort"], condition=r["condition"], group=r["group"], alpha=r["alpha"],
                origin=when, time_s=times[index], abs_phi_deg=np.degrees(abs(state[0])),
                abs_omega_deg_s=np.degrees(abs(state[1])), front_force_N=normal[index],
                speed_m_s=speed[index], steering_rad=state[2], nominal_domain=int(valid),
                classification=result.get("outcome", "outside_domain"), same_side=int(same_side),
                actual_lead_ms=actual, nominal_lead_ms=predicted,
                error_ms=(predicted-actual) if same_side else np.nan)


def collect(events):
    batch = defaultdict(list)
    for r in events:
        batch[r["raw"]].append(r)
    runs, queries, timing_states = [], [], []
    for raw, rows in batch.items():
        path = Path(raw)
        contract = json.loads((path / "contract.json").read_text())
        bodies = contract["lab_physics"]["runtime_inertial"]["body_names"]
        with np.load(path / "trajectory.npz") as z:
            m = z["measurement"]
            trace = z["lower_trace"].reshape(*m.shape[:2], 11)
            omega = euler_roll_rate(z["body_state"][:, :, bodies.index("chasis")])
            # Archived net contact-force magnitude, the wheel-load proxy used by the forensic audit.
            front = np.linalg.norm(z["contact_force"][:, :, bodies.index("frontwheel")], axis=-1)
        for old in rows:
            lane = int(old["lane"])
            active = np.flatnonzero(trace[:, lane, 6])
            end = float(old["time_s"])
            full = bool(int(old["full"]))
            is_roll = "roll_limit" in old["term"]
            times = active*DT + trace[active, lane, 7].astype(float)*float(contract["physics_dt"])
            # The diagnostics store rigid-body/rate at the end of a lower step. Partial
            # terminal contact steps cannot supply a strictly pre-termination warning.
            keep = times <= end + 1e-7 if full else times < end - 1e-7
            idx = active[keep]
            t = times[keep]
            phi = m[idx, lane, 0].astype(float)
            w = omega[idx, lane].astype(float)
            speed = trace[idx, lane, 2]
            states = np.column_stack([phi, w, m[idx, lane, 2]])
            env_index = float(old["t_env"])/DT
            env_k = int(round(env_index)) if np.isfinite(env_index) else -1
            raw_lead = (end-t[env_k])*1000 if 0 <= env_k < len(t) else np.nan
            # The legacy first-hit index came from a permissive diagnostic. Apply
            # the theorem's stated domain before counting a crossing as a warning.
            domain = (speed >= .5) & (np.abs(states[:, 2]) <= .8) & (np.abs(phi) < PHI_C)
            if np.isfinite(raw_lead) and not domain[env_k]:
                future = np.flatnonzero(domain & (np.arange(len(t)) > env_k))
                env_k = -1
                for block in range(0, len(future), 32):
                    ii = future[block:block + 32]
                    hits = np.flatnonzero(recovery_margin_batch(speed[ii], states[ii], CTRL_CFG) < 0)
                    if len(hits):
                        env_k = ii[hits[0]]
                        break
            env_lead = (end-t[env_k])*1000 if 0 <= env_k < len(t) else np.nan
            # All scores rise with danger. Negative time to bound avoids singular
            # reciprocal scores when approaching the angle limit.
            distance = PHI_C-np.abs(phi)
            ttb = np.divide(distance, abs(w), out=np.full_like(w, np.inf), where=w != 0)
            stopping = np.minimum(PHI_C-phi-np.maximum(w, 0)**2/(2*B_NOM),
                                  PHI_C+phi-np.maximum(-w, 0)**2/(2*B_NOM))
            scores = np.stack((abs(phi), abs(w), -ttb, -stopping))
            assert np.isfinite(scores[:2]).all() and not np.isnan(scores).any()
            r = dict(cohort=old["source"], condition=old["cond"], group=old["group"], alpha=old["alpha"],
                     raw=raw, lane=lane, terminal=old["term"], endpoint="roll" if is_roll else "full" if full else "other",
                     end_s=end, raw_envelope_lead_ms=raw_lead, envelope_lead_ms=env_lead)
            if is_roll:
                terminal_side = np.sign(m[active[-1], lane, 0])
                qidx = [(env_k, "first_alert")] if np.isfinite(env_lead) else []
                for lag in (200, 100, 50, 25):
                    i = int(np.argmin(abs(end - t - lag / 1000)))
                    if abs(end-t[i]-lag/1000) <= DT/2 + 1e-7:
                        qidx.append((i, str(lag)+"ms"))
                for i, label in qidx:
                    queries.append(crossing_query(r, i, label, states, speed, front[idx,lane], t, terminal_side))
                if np.isfinite(env_lead):
                    timing_states.append((speed[env_k:env_k+1], states[env_k:env_k+1]))
            r.update(prefix=np.maximum.accumulate(scores, axis=1), times=t)
            runs.append(r)
    return runs, queries, timing_states


def binned_errors(queries):
    rows = []
    for origin in ("first_alert", "200ms", "100ms", "50ms", "25ms"):
        selected = [q for q in queries if q["origin"] == origin]
        for key, cuts in [("abs_phi_deg", [0,10,20,30,np.inf]),
                          ("abs_omega_deg_s", [0,50,100,250,np.inf]),
                          ("front_force_N", [0,1,10,30,np.inf])]:
            for lo, hi in zip(cuts, cuts[1:]):
                group = [q for q in selected if lo <= q[key] < hi]
                err = np.array([q["error_ms"] for q in group if np.isfinite(q["error_ms"])])
                rows.append(dict(origin=origin, variable=key, low=lo, high=hi, n=len(group),
                                 in_domain=sum(q["nominal_domain"] for q in group), same_side_forecasts=len(err),
                                 median_error_ms=float(np.median(err)) if len(err) else None,
                                 median_abs_error_ms=float(np.median(abs(err))) if len(err) else None,
                                 q90_abs_error_ms=float(np.quantile(abs(err), .9)) if len(err) else None))
    return rows


def draw(curves, points, queries, path, roll_n, full_n):
    plt.rcParams.update({"font.family":"serif", "font.size":8, "axes.spines.top":False,
                         "axes.spines.right":False, "pdf.fonttype":42, "mathtext.fontset":"stix"})
    fig, ax = plt.subplots(1, 3, figsize=(7.18, 2.55), layout="constrained")
    colors = ["#346182","#BA5729","#35806B","#8D659F"]
    for name, label, color, style in zip(DETECTORS, LABELS, colors, ["-", "--", "-.", ":"]):
        rows = sorted([r for r in curves if r["detector"] == name],key=lambda r:-r["score_threshold"])
        x = np.array([r["full_alerts"]*100/full_n for r in rows])
        y = [r["roll_alerts"]*100/roll_n for r in rows]
        ax[0].plot(x,y,style,color=color,label=label,lw=1.25)
        # At each control-alert count, use the least restrictive threshold
        # implicit in the panel calibration (global maxima can be tied).
        operating = {r["full_alerts"]:r for r in rows}
        selected = [operating[n] for n in sorted(operating) if n<=.2*full_n and operating[n]["roll_alerts"] == roll_n]
        ax[1].plot([r["full_alerts"]*100/full_n for r in selected], [r["median_lead_ms"] for r in selected],
                   style,color=color,lw=1.25)
    ep = next(r for r in points if r["detector"] == "nominal_envelope")
    fpr = ep["full_alerts"]*100/full_n
    ax[0].scatter(fpr,100*ep["roll_alerts"]/roll_n,marker="*",color="black",s=65,zorder=4,label="Envelope")
    ax[1].scatter(fpr,ep["median_lead_ms"],marker="*",color="black",s=65,zorder=4)
    ax[0].set(title="(a) Episode detection",xlabel="Full-window alerts (%)",ylabel="Roll runs warned (%)",xlim=(-1,20),ylim=(-3,106))
    ax[1].set(title="(b) All roll runs warned",xlabel="Full-window alerts (%)",ylabel="Median lead (ms)",xlim=(-1,20))
    ax[0].legend(loc="lower right",fontsize=6.4,frameon=False)
    matched = [q for q in queries if q["origin"] == "first_alert" and q["same_side"]]
    for supported, marker, color, label in [(True,"o","#346182","Front load ≥1 N"),(False,"x","#BA5729","Front load <1 N")]:
        keep = [q for q in matched if (q["front_force_N"]>=1)==supported]
        ax[2].scatter([q["actual_lead_ms"] for q in keep],[q["nominal_lead_ms"] for q in keep],marker=marker,s=9,
                      alpha=.67,lw=.6,color=color,label=label)
    lim = max(max(q["actual_lead_ms"],q["nominal_lead_ms"]) for q in matched)*1.03
    ax[2].plot([0,lim],[0,lim],"--",color=".5",lw=.8)
    ax[2].set(title="(c) First envelope alert",xlabel="Observed time to limit (ms)",ylabel="Nominal time to limit (ms)",xlim=(0,lim),ylim=(0,lim))
    ax[2].legend(loc="lower right",fontsize=6.2,frameon=False)
    for a in ax: a.grid(alpha=.2)
    fig.savefig(path)
    plt.close(fig)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--events", type=Path, default=EVIDENCE/"events.csv")
    args.add_argument("--output", type=Path, default=EVIDENCE/"warning_comparison")
    args.add_argument("--figure", type=Path, default=Path(__file__).parent/"figures/warning_comparison.pdf")
    a = args.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with a.events.open() as f:
        cases = [r for r in csv.DictReader(f) if r["source"] in COHORTS]
    assert len(cases)==760 and len({(r["raw"],r["lane"]) for r in cases})==760
    runs, queries, states = collect(cases)
    curves, points = scan(runs, a.output)
    for r in runs:
        del r["times"], r["prefix"]
    write_csv(a.output/"episodes.csv",runs)
    write_csv(a.output/"crossing_queries.csv",queries)
    write_csv(a.output/"domain_bins.csv",binned_errors(queries))
    cohort = []
    for name in COHORTS:
        x = [r for r in runs if r["cohort"]==name]
        for outcome in ("roll","full","other"):
            subset = [r["envelope_lead_ms"] for r in x if r["endpoint"]==outcome]
            leads = [v for v in subset if np.isfinite(v)]
            cohort.append(dict(cohort=name,endpoint=outcome,n=len(subset),alerts=len(leads),
                               median_lead_ms=float(np.median(leads)) if leads else None))
    write_csv(a.output/"cohorts.csv",cohort)
    # Warm scalar calls on observed first-alert states. Input preparation and disk IO
    # are outside the clock; coefficient calculation and the 4001-point grid are inside.
    for v,x in states[:10]: recovery_margin_batch(v,x,CTRL_CFG)
    us = []
    for _ in range(4):
        for v,x in states:
            start = perf_counter_ns()
            margin = recovery_margin_batch(v,x,CTRL_CFG)
            us.append((perf_counter_ns()-start)/1000)
            assert margin[0]<0
    cpu = next(s.split(":",1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name"))
    timing = dict(cpu=cpu,python=platform.python_version(),numpy=np.__version__,
                  implementation="NumPy recovery_margin_batch; 1 state, 1 s horizon, 0.25 ms internal grid",
                  calls=len(us), median_us=np.median(us),p95_us=np.quantile(us,.95))
    (a.output/"timing.json").write_text(json.dumps(timing,indent=2)+"\n")
    draw(curves,points,queries,a.figure,sum(r["endpoint"]=="roll" for r in runs),sum(r["endpoint"]=="full" for r in runs))
    print(points, timing)

if __name__ == "__main__":
    main()
