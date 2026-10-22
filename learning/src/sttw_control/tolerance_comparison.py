"""Matched evaluation layout and bookkeeping without simulator dependencies."""
from itertools import product
import math


def chunk_schedule(onset, duration=20., dt=.02, max_chunk=50):
    """Split scans at the pulse onset, retaining every control interval."""
    if dt <= 0 or max_chunk < 1 or not math.isfinite(onset):
        raise ValueError("invalid timing")
    total = round(duration / dt)
    pulse = round(onset / dt)
    if not math.isclose(total * dt, duration, abs_tol=1e-9) or not math.isclose(pulse * dt, onset, abs_tol=1e-9) or not 0 <= pulse < total:
        raise ValueError("onset and duration must align with control ticks")
    result = []
    start = 0
    while start < total:
        end = min(start + max_chunk, total)
        if start < pulse < end:
            end = pulse
        result.append((start, end - start, start == pulse))
        start = end
    return result


def comparison_layout(protocol):
    lanes = [dict(policy=p, route=r, speed=s, alpha=a, force_N=f)
             for p, r, s, a, f in product(*(protocol[k] for k in
                 ("policies", "routes", "speeds", "endpoints", "forces")))]
    def context(row):
        return tuple(row[k] for k in ("policy", "route", "speed", "alpha"))
    zeros = {}
    for i, row in enumerate(lanes):
        if row["force_N"] == 0:
            key = context(row)
            if key in zeros:
                raise ValueError("duplicate zero-force context")
            zeros[key] = i
    try:
        return lanes, [zeros[context(row)] for row in lanes]
    except KeyError as exc:
        raise ValueError("missing matched zero-force branch") from exc


def paired_counts(rows, baseline, candidate, alpha):
    def collect(policy):
        result = {}
        for row in rows:
            if row["policy"] != policy or row["alpha"] != alpha:
                continue
            key = tuple(row[k] for k in ("route", "speed", "onset_s", "force_N"))
            if key in result:
                raise ValueError("duplicate paired case")
            if row["success"] not in (True, False):
                raise ValueError("missing binary outcome")
            result[key] = bool(row["success"])
        return result
    a, b = collect(baseline), collect(candidate)
    if not a or a.keys() != b.keys():
        raise ValueError("incomplete paired outcomes")
    improved = sum(not a[k] and b[k] for k in a)
    regressed = sum(a[k] and not b[k] for k in a)
    return dict(n=len(a), baseline_success=sum(a.values()), candidate_success=sum(b.values()),
                improved=improved, regressed=regressed, difference=improved-regressed)
