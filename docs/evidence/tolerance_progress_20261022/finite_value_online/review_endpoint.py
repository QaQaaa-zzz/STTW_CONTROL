"""Read-only decision audit; no simulation, fitting, or checkpoint selection.

Usage: PYTHONPATH=learning/src python review_endpoint.py /path/to/evaluation
Writes additional review artifacts without modifying the original analysis.
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from sttw_control.rejoin_comparison import geometric_task_trace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(root):
    root = Path(root).resolve()
    out = root / 'analysis'
    status = json.loads((out / 'status.json').read_text())
    assert status['phase'] == 'completed' and status['completed_cases'] == 300
    rows = json.loads((out / 'results.json').read_text())
    assert len(rows) == len({r['case_id'] for r in rows}) == 300
    config = json.loads((root / 'common_environment_config.json').read_text())
    bands = config['stage_B_baseline']
    dt = config['timing']['lower_dt_s']
    plan = json.loads((root / 'plan.json').read_text())
    paths = {}
    for context, item in plan['contexts'].items():
        assert sha(Path(item['path'])) == item['path_sha256']
        with np.load(item['path'], allow_pickle=False) as z:
            paths[context] = {k: z[k] for k in z.files}
    indices = {r['case_id']: r for r in json.loads((root / 'physical/case_index.json').read_text())}
    groups = defaultdict(dict)
    aggregates = []
    for row in rows:
        groups[(row['context'], row['alpha'], row['event_id'])][row['role']] = row
    assert len(groups) == 50 and all(len(v) == 6 for v in groups.values())
    for split in ['all', 'nominal', 'disturbed']:
        for role in groups[next(iter(groups))]:
            rr = [r for r in rows if r['role'] == role and
                  (split == 'all' or bool(r['force_N']) == (split == 'disturbed'))]
            aggregates.append(dict(split=split, role=role, n=len(rr),
                tasks=sum(r['category'] == 'safe_task' for r in rr),
                contacts=sum(r['category'] == 'contact' for r in rr),
                unknown=sum(r['category'] == 'unknown' for r in rr),
                mean_cost=float(np.mean([r['scored_cost'] for r in rr])),
                mean_path_rmse_m=float(np.mean([r['task_path_rmse_m'] for r in rr])),
                median_tail_path_m=float(np.median([r['tail_path_max_m'] for r in rr])),
                median_tail_heading_rad=float(np.median([r['tail_heading_error_max_rad'] for r in rr])),
                crossed_goal=sum(r['goal_crossed_while_task_valid'] for r in rr),
                full_pulse_exposure=sum(r['full_pulse_exposure'] for r in rr)))
    ranking = dict(groups_with_task=0, minimum_cost_task=0, minimum_cost_task_failed=0)
    for group in groups.values():
        known = [r for r in group.values() if r['category'] in ['safe_task', 'safe_task_failed']]
        if any(r['category'] == 'safe_task' for r in known):
            ranking['groups_with_task'] += 1
            winner = min(known, key=lambda r: r['scored_cost'])
            ranking['minimum_cost_task' if winner['category'] == 'safe_task'
                    else 'minimum_cost_task_failed'] += 1
    releases = []
    for row in rows:
        if row['role'] != 'parent_causal':
            continue
        file = Path(row['raw_file'])
        assert sha(file) == row['raw_sha256']
        with np.load(file, allow_pickle=False) as z:
            raw = {k: z[k] for k in z.files}
        active = raw['active_tick'].astype(bool)
        gate = active & raw['relay_active'].astype(bool)
        ticks = np.flatnonzero(gate)
        if not ticks.size:
            continue
        geometry = geometric_task_trace(raw, paths[row['context']], config,
                                        initial_xy=indices[row['case_id']]['initial_xy'])
        # End of the last physically executed support tick, not an invented
        # hybrid carry or an independently replayable mid-episode snapshot.
        last = int(ticks[-1])
        after = active & (raw['physical_tick'] > raw['physical_tick'][last])
        assert not gate[after].any()
        path = float(geometry['path_cross_track'][last])
        heading = float(geometry['path_heading_error'][last])
        speed_error = float(raw['actual_forward_speed'][last] - row['requested_speed_m_s'])
        releases.append(dict(case_id=row['case_id'], category=row['category'],
            alpha=row['alpha'], context=row['context'],
            release_time_s=float((raw['physical_tick'][last] + 1) * dt),
            remaining_s=float(after.sum() * dt),
            gate_duration_s=float(gate.sum() * dt),
            release_path_m=path, release_heading_rad=heading,
            release_roll_rad=float(raw['phi'][last]),
            release_roll_rate_rad_s=float(raw['phi_dot'][last]),
            release_speed_error_m_s=speed_error,
            release_in_task_bands=bool(abs(path) <= bands['final_y_error_m'] and
                abs(heading) <= bands['final_path_heading_error_rad'] and
                abs(speed_error) <= bands['final_speed_error_m_s']),
            # .4 is the existing relay's off threshold, not a stability claim.
            release_rate_within_existing_off_threshold=bool(abs(raw['phi_dot'][last]) <= .4),
            after_release_actor_legal_fraction=float(raw['actor_action_valid'][after].mean())
                if after.any() else None,
            tail_path_max_m=row['tail_path_max_m'],
            tail_heading_max_rad=row['tail_heading_error_max_rad']))
    fail = [r for r in releases if r['category'] == 'safe_task_failed']
    findings = dict(source_results_sha256=sha(out / 'results.json'),
        source_config_sha256=sha(root / 'common_environment_config.json'),
        source_analysis_status=status, aggregate=aggregates, finite_library_ranking=ranking,
        causal_releases=dict(n=len(releases), failed_tasks=len(fail),
            failed_outside_task_bands=sum(not r['release_in_task_bands'] for r in fail),
            failed_rate_within_existing_off_threshold=sum(r['release_rate_within_existing_off_threshold'] for r in fail),
            failed_with_some_actor_unavailability_after_release=sum(
                r['after_release_actor_legal_fraction'] < 1 for r in fail),
            median_failed_actor_legal_fraction=float(np.median([
                r['after_release_actor_legal_fraction'] for r in fail])) if fail else None,
            median_failed_absolute_release_path_m=float(np.median([
                abs(r['release_path_m']) for r in fail])) if fail else None,
            median_failed_absolute_release_heading_rad=float(np.median([
                abs(r['release_heading_rad']) for r in fail])) if fail else None,
            median_failed_remaining_s=float(np.median([r['remaining_s'] for r in fail])) if fail else None),
        physics_steps=0, optimizer_updates=0,
        limitations=['one matched training repeat; two old contexts and correlated branches',
            'all300 survived: this panel cannot rank fall prevention',
            'finite-library cost ranking is descriptive, not deployable oracle',
            'release values are observations; no counterfactual tail or stability certificate'])
    (out / 'decision_review.json').write_text(json.dumps(findings, indent=2) + '\n')
    if releases:
        with (out / 'causal_release_review.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(releases[0]))
            writer.writeheader(); writer.writerows(releases)
    print(json.dumps({k: v for k, v in findings.items() if k not in ['aggregate', 'source_analysis_status']}, indent=2))


if __name__ == '__main__':
    review(sys.argv[1])
