"""Repeat a disclosed TEST with calibrated GPU decision delays on its original assignments."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from report_blind_platform import summarize
from run_blind_platform import execute


def repeat(baseline, benchmark, output):
    original = baseline / 'test_freeze_manifest.json'
    plan = json.loads(original.read_text())
    rows = pd.read_csv(benchmark).query("method == 'phase2_relative'").set_index('candidates').loc[[9, 3]]
    error = rows[['max_pose_abs', 'max_velocity_abs', 'max_domain_abs']].to_numpy()
    if not ((error < 1e-5).all() and (rows.decision_mismatches == 0).all()
            and (rows.horizon_steps == 255).all() and (rows.n >= 1000).all()):
        raise ValueError('Benchmark failed the forecast, decision or timing protocol.')
    output.mkdir(exist_ok=True, parents=True)
    plan.update(role='Post-TEST paired latency ablation', parent_manifest=str(original),
                parent_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                latency_source=str(benchmark), timings_ms={str(n): t for n, t in rows.p95_ms.items()},
                latency_ticks={'start': math.ceil(rows.at[9, 'p95_ms'] / 5),
                               'departure': math.ceil(rows.at[3, 'p95_ms'] / 5)})
    for layout in plan['maps']:
        folder = output / layout
        folder.mkdir(exist_ok=True)
        for name in ('terrain.npz', 'source_prior.csv', 'rb_run1_prior.csv',
                     'source_initializer.npz', 'rb_run1_initializer.npz'):
            target = folder / name
            if not target.exists():
                target.symlink_to((baseline / layout / name).resolve())
    receipt = output / 'test_freeze_manifest.json'
    if receipt.exists() and json.loads(receipt.read_text()) != plan:
        raise ValueError('Existing ablation uses a different protocol')
    receipt.write_text(json.dumps(plan, indent=2) + '\n')
    execute(output)
    summarize(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('benchmark', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    repeat(args.baseline, args.benchmark, args.output)
