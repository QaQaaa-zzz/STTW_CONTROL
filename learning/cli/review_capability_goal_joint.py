"""Authenticate historical low-speed90 and separate arrival from physical survival.

No simulation, fitting, or reward changes. Outputs refuse existing directories.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from sttw_control.goal_arrival_analysis import analyze, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    grid = args.worktree / 'runs/preference_capability_grid_v1'
    table = grid / 'analysis/all_cases.csv'
    source = list(csv.DictReader(table.open()))
    if len(source) != 90:
        raise ValueError('Expected the sealed historical90 panel')
    audits = []
    for row in source:
        if digest(row['raw_file']) != row['raw_sha256']:
            raise ValueError('Historical raw identity changed')
        with np.load(row['raw_file'], allow_pickle=False) as z:
            active = z['active_tick'].astype(bool)
            ticks = z['physical_tick'][active]
            if not np.array_equal(ticks, np.arange(len(ticks))):
                raise ValueError('Incomplete physical clock prefix')
            contact = bool(np.any(active & z['contact_failure'].astype(bool)))
            fault = bool(np.any(active & (z['nonfinite'] | z['lower_fault'] | z['policy_fault'])))
            outcome = 'unknown' if fault else 'contact' if contact else 'survived' if len(ticks) == 2000 else 'unknown'
            if outcome != row['physical_outcome']:
                raise ValueError('Physical outcome disagrees with authenticated source')
            audits.append(dict(case_id=row['case_id'], outcome=outcome,
                               observed_ticks=len(ticks), raw_sha256=row['raw_sha256']))
    analyze([('force_grid', table, grid / 'evaluation/plan.json')], args.output,
            radius_m=1., radius_authority='Current user-approved original-world-goal radius1m, applied retrospectively to the old10s panel; historical labels preserved.')
    rows = list(csv.DictReader((args.output / 'cases.csv').open()))
    rows = [r for r in rows if float(r['goal_radius_m']) == 1.]
    groups = {}
    for row in rows:
        key = (float(row['alpha']), float(row['force_N']))
        group = groups.setdefault(key, {})
        if row['role'] in group:
            raise ValueError('Duplicate role in paired event')
        group[row['role']] = row
    if len(groups) != 45 or any(set(g) != {'task_only', 'task_constant'} for g in groups.values()):
        raise ValueError('Complete45 paired events required')
    pairs = []
    counts = {}
    for (alpha, force), group in sorted(groups.items()):
        a, b = group['task_only'], group['task_constant']
        arrival = lambda r: r['goal_arrived'] == 'True'
        survives = lambda r: r['physical_outcome'] == 'survived'
        joint = lambda r: arrival(r) and survives(r)
        pair = dict(alpha=alpha, force_N=force,
                    original_arrived=arrival(a), support_arrived=arrival(b),
                    original_survived=survives(a), support_survived=survives(b),
                    arrival_delta=int(arrival(b))-int(arrival(a)),
                    survival_delta=int(survives(b))-int(survives(a)),
                    joint_delta=int(joint(b))-int(joint(a)),
                    original_min_goal_m=float(a['minimum_upright_goal_distance_m']),
                    support_min_goal_m=float(b['minimum_upright_goal_distance_m']))
        pairs.append(pair)
    for role in ['task_only', 'task_constant']:
        for split in ['all', 'nominal', 'disturbed']:
            rr = [r for r in rows if r['role'] == role and
                  (split == 'all' or bool(float(r['force_N'])) == (split == 'disturbed'))]
            counts[role + '/' + split] = dict(n=len(rr), arrived=sum(arrival(r) for r in rr),
                survived=sum(survives(r) for r in rr), joint=sum(joint(r) for r in rr))
    transitions = {}
    for key in ['arrival_delta', 'survival_delta', 'joint_delta']:
        transitions[key] = dict(gain=sum(p[key] == 1 for p in pairs),
                                loss=sum(p[key] == -1 for p in pairs))
    gains = [p for p in pairs if p['survival_delta'] == 1]
    result = dict(counts=counts, paired_transitions=transitions, physical_rescue_cases=gains,
        physical_rescues_also_arriving=sum(p['support_arrived'] for p in gains),
        physical_raw_audits=audits, new_physics_steps=0, optimizer_updates=0,
        interpretation='Retrospective shared-ancestor DEV. Constant support changes pre-disturbance entry state; not same-state rescue, not a learned selector, not independent confirmation.10s cannot establish20s outcomes. No force-grid expansion or threshold fitting.',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / 'joint_review.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output / 'paired.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0])); writer.writeheader(); writer.writerows(pairs)
    print(json.dumps({k: v for k, v in result.items() if k not in ['physical_raw_audits', 'physical_rescue_cases']}, indent=2))


if __name__ == '__main__':
    main()
