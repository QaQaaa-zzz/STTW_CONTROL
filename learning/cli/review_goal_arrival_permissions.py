"""Reconstruct observed goal entries and annotate actual logged Actor permission.

This is a temporal description, not a counterfactual feedback-necessity test.
No policy evaluation or simulator is imported.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_table(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def segments(labels, dt):
    cuts = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]] + [len(labels)]
    return [dict(phase=labels[a], start_tick=a, end_tick_exclusive=b,
                 interval_start_s=a * dt, interval_end_s=b * dt)
            for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def compare_independent(records, other):
    bykey = {r['key']: r for r in other['records']}
    maximum, arrivals, phases = 0., 0, 0
    names = {'original_permission': 'active_before_exit', 'reentry_permission': 'active_reentry'}
    for r in records:
        s = bykey[r['key']]
        assert r['raw_sha256'] == s['raw_sha256'] and r['goal'] == s['goal_arrived']
        maximum = max(maximum, abs(r['minimum_goal_distance_m'] - s['minimum_prefatal_goal_distance_m']))
        assert r['first_contact_posttick_elapsed_s'] == s['first_contact_posttick_elapsed_s']
        if r['goal']:
            a = s['first_arrival']
            assert r['first_arrival_index'] == a['task_tick']
            maximum = max(maximum, abs(r['first_arrival_posttick_elapsed_s'] - a['posttick_elapsed_s']))
            arrivals += 1
            for k, f in [('arrival_actor_permission', 'actor_action_valid'),
                         ('arrival_reentry_permission', 'reentry_actor_active'),
                         ('arrival_ever_exit', 'ever_domain_exit'),
                         ('arrival_current_invalid', 'domain_invalid'), ('arrival_zero_fallback', 'zero_fallback')]:
                assert r[k] == a['flags'][f]
            if r['role'] != 'zero_middle':
                assert names[r['arrival_permission_phase']] == a['phase']
                phases += 1
    assert maximum < 1e-8
    return dict(records=len(records), arrival_samples_compared=arrivals,
                nonzero_role_permission_phases_compared=phases, maximum_numeric_difference=maximum,
                all_arrival_raw_flags_match=True, new_physics_ticks=0, new_training_steps=0)


def plot_summary(root, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for r in summary if r['role'] != 'zero_middle']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    original = [r['arrival_phases'].get('original_permission', 0) for r in rows]
    reentry = [r['arrival_phases'].get('reentry_permission', 0) for r in rows]
    missed = [r['n'] - r['arrived'] for r in rows]
    y = np.arange(len(rows))
    axes[0].barh(y, original, color='#009E73', label='Arrived: before first exit')
    axes[0].barh(y, reentry, left=original, color='#E69F00', hatch='//', label='Arrived: reentry permission')
    left = np.asarray(original) + np.asarray(reentry)
    axes[0].barh(y, missed, left=left, color='#dddddd', label='Not arrived')
    for i, (a, b) in enumerate(zip(original, reentry)):
        axes[0].text(a / 2, i, str(a), ha='center', va='center')
        axes[0].text(a + b / 2, i, str(b), ha='center', va='center')
    axes[0].set(yticks=y, yticklabels=[r['role'] for r in rows], xlim=(0, 70),
                xlabel='Trajectory records (70 per role)', title='All observed goal entries, radius 1 m')
    axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.14), fontsize=8)
    gain_a = [r['gain_arrival_phases'].get('original_permission', 0) for r in rows]
    gain_b = [r['gain_arrival_phases'].get('reentry_permission', 0) for r in rows]
    losses = [r['goal_losses'] for r in rows]
    axes[1].barh(y, -np.asarray(losses), color='#bbbbbb', hatch='..', label='Goal losses vs zero')
    axes[1].barh(y, gain_a, color='#009E73')
    axes[1].barh(y, gain_b, left=gain_a, color='#E69F00', hatch='//')
    axes[1].axvline(0, color='black', linewidth=.7)
    for i, r in enumerate(rows):
        axes[1].text(-r['goal_losses'] - .5, i, str(r['goal_losses']), ha='right', va='center')
        axes[1].text(r['goal_gains'] + .5, i, str(r['goal_gains']), ha='left', va='center')
    axes[1].set(xlim=(-27, 27), xlabel='Goal losses (left) / gains (right), paired records',
                title='All 420 logical comparisons to zero')
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.14), fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('Temporal attribution only: permitted Actor output is not proof of causal necessity\n14 zero trajectories reused: 6 reached the goal; related DEV, no independent training repeats')
    fig.savefig(root / 'arrival_permission_summary.png', dpi=160)
    plt.close(fig)


def main(root):
    plan = json.loads((root / 'plan.json').read_text())
    for file, sha in plan['sources_sha256'].items():
        assert digest(file) == sha, file
    source_paths = list(plan['sources_sha256'])
    source = table(source_paths[0])
    old = {r['key']: r for r in table(source_paths[1])}
    goals, route_identity = {}, {}
    for file in source_paths[3:]:
        p = Path(file)
        route = p.parent.name.split('_')[0]
        contract = json.loads(p.read_text())
        assert contract['lower_dt_s'] == .005 and contract['horizon_s'] == 10.
        path_file = Path(contract['path_file'])
        assert digest(path_file) == contract['path_sha256']
        with np.load(path_file) as path:
            goals[route] = np.array([np.interp(float(path['goal']), path['s'], path['xy'][:, i]) for i in (0, 1)])
        route_identity[route] = dict(path_file=str(path_file), path_sha256=contract['path_sha256'], goal_xy_m=goals[route].tolist())
        # Verify the archived, actually executed logging and permission definitions.
        for name in ('anticipatory_reference.py', 'reference_reentry.py', 'path_command_env.py'):
            archived = [f for f in (p.parent / 'source_snapshot').iterdir() if f.name.partition('_')[2] == name]
            sha = [v for k, v in contract['source_sha256'].items() if Path(k).name == name]
            assert len(archived) == len(sha) == 1 and digest(archived[0]) == sha[0]
    records = []
    dt = .005
    for r in source:
        assert digest(r['raw_file']) == r['raw_sha256']
        with np.load(r['raw_file']) as z:
            raw = {k: z[k] for k in z.files}
        active = raw['active_tick'].astype(bool)
        n = int(active.sum())
        assert np.array_equal(np.flatnonzero(active), np.arange(n))
        assert np.array_equal(raw['physical_tick'][active], np.arange(n))
        xy = np.asarray(raw['actual_xy'][:n], float)
        contact = raw['contact_failure'][:n].astype(bool)
        fault = np.logical_or.reduce([raw[k][:n].astype(bool) for k in ('nonfinite', 'lower_fault', 'policy_fault')]) | ~np.isfinite(xy).all(axis=1)
        fatal = np.flatnonzero(contact | fault)
        stop = int(fatal[0]) if len(fatal) else n
        distances = np.linalg.norm(xy[:stop] - goals[r['route']], axis=1)
        hits = np.flatnonzero(distances <= 1.)
        hit = int(hits[0]) if len(hits) else None
        arrived = hit is not None
        assert not fault.any(), 'This source panel is known, do not silently coerce unknown'
        assert arrived == (old[r['key']]['goal'] == 'True')
        minimum = float(distances.min()) if len(distances) else None
        assert minimum is not None and abs(minimum - float(old[r['key']]['distance'])) < 1e-8
        valid = raw['actor_action_valid'][:n].astype(bool)
        reentry = raw['reentry_actor_active'][:n].astype(bool)
        exited = raw['ever_domain_exit'][:n].astype(bool)
        fallback = raw['zero_fallback'][:n].astype(bool)
        invalid = raw['domain_invalid'][:n].astype(bool)
        zero = r['role'] == 'zero_middle'
        assert not np.any(reentry & ~valid) and not np.any(fallback & valid)
        assert not zero or not valid.any()
        labels = [('deliberate_zero' if zero else 'no_actor_permission' if not valid[i] else
                   'reentry_permission' if reentry[i] else 'first_exit_block_permission' if exited[i] else
                   'original_permission') for i in range(n)]
        perm_segments = segments(labels, dt)
        first_exit = np.flatnonzero(exited)
        first_invalid = np.flatnonzero(invalid)
        first_no_permission = np.flatnonzero(~valid) if not zero else np.array([], dtype=int)
        pulse_indices = np.flatnonzero(np.linalg.norm(raw['applied_force_impulse'][:n], axis=1) > 0)
        row = dict(key=r['key'], role=r['role'], route=r['route'],
                   alpha=None if zero else float(r['alpha']), event_id=r['event_id'],
                   force_N=float(r['force_N']), onset=int(r['onset_tick']),
                   raw_file=r['raw_file'], raw_sha256=r['raw_sha256'], active_ticks=n,
                   goal=arrived, survival=old[r['key']]['survival'] == 'True',
                   minimum_goal_distance_m=minimum, first_arrival_index=hit,
                   first_arrival_posttick_elapsed_s=(hit + 1) * dt if arrived else None,
                   first_contact_posttick_elapsed_s=(int(np.flatnonzero(contact)[0]) + 1) * dt if contact.any() else None,
                   first_exit_observed_posttick_s=(int(first_exit[0]) + 1) * dt if len(first_exit) else None,
                   first_invalid_observed_posttick_s=(int(first_invalid[0]) + 1) * dt if len(first_invalid) else None,
                   first_no_permission_interval_start_s=int(first_no_permission[0]) * dt if len(first_no_permission) else None,
                   first_logged_force_interval_start_s=int(pulse_indices[0]) * dt if len(pulse_indices) else None,
                   last_logged_force_interval_end_s=(int(pulse_indices[-1]) + 1) * dt if len(pulse_indices) else None,
                   arrival_permission_phase=labels[hit] if arrived else 'not_arrived',
                   arrival_actor_permission=bool(valid[hit]) if arrived else None,
                   arrival_reentry_permission=bool(reentry[hit]) if arrived else None,
                   arrival_ever_exit=bool(exited[hit]) if arrived else None,
                   arrival_current_invalid=bool(invalid[hit]) if arrived else None,
                   arrival_zero_fallback=bool(fallback[hit]) if arrived else None,
                   permission_segments=perm_segments)
        for k in ('issued_latent', 'original_actor_latent', 'reference', 'nominal_reference', 'filtered_offset'):
            present = arrived and (k != 'original_actor_latent' or valid[hit])
            row['arrival_' + k] = np.asarray(raw[k][hit], float).tolist() if present else None
        row['arrival_reference_delta_from_previous_lower'] = (np.asarray(raw['reference'][hit], float) -
            np.asarray(raw['reference'][hit - 1], float)).tolist() if arrived and hit > 0 else None
        row['arrival_issued_minus_original_actor'] = (np.asarray(raw['issued_latent'][hit], float) -
            np.asarray(raw['original_actor_latent'][hit], float)).tolist() if arrived and valid[hit] else None
        if arrived:
            # A temporal diagnostic, not a "constant reference" causal gate.
            start = max(0, hit - 199)
            window = np.asarray(raw['reference'][start:hit + 1], float)
            row['prearrival_reference_window_s'] = len(window) * dt
            row['prearrival_reference_range_speed_m_s'] = float(np.ptp(window[:, 0]))
            row['prearrival_reference_range_steer_rad'] = float(np.ptp(window[:, 1]))
            row['prearrival_actor_permission_fraction'] = float(valid[start:hit + 1].mean())
            row['reentry_segments_before_arrival'] = sum(s['phase'] == 'reentry_permission' and s['start_tick'] <= hit for s in perm_segments)
            actor_window = np.asarray(raw['original_actor_latent'][start:hit + 1], float)[valid[start:hit + 1]]
            row['prearrival_original_actor_range'] = np.ptp(actor_window, axis=0).tolist() if len(actor_window) else None
        else:
            for k in ('prearrival_reference_window_s', 'prearrival_reference_range_speed_m_s',
                      'prearrival_reference_range_steer_rad', 'prearrival_actor_permission_fraction', 'reentry_segments_before_arrival',
                      'prearrival_original_actor_range'):
                row[k] = None
        records.append(row)
    assert len(records) == 434
    zero = {(r['route'], r['event_id']): r for r in records if r['role'] == 'zero_middle'}
    pairs = []
    for r in records:
        if r['role'] == 'zero_middle':
            continue
        z = zero[r['route'], r['event_id']]
        pairs.append(dict(key=r['key'], zero_key=z['key'], role=r['role'], route=r['route'], alpha=r['alpha'],
                          event_id=r['event_id'], force_N=r['force_N'], goal_delta=int(r['goal']) - int(z['goal']),
                          survival_delta=int(r['survival']) - int(z['survival']),
                          learned_arrival_phase=r['arrival_permission_phase'], zero_arrival_phase=z['arrival_permission_phase'],
                          learned_arrival_posttick_s=r['first_arrival_posttick_elapsed_s'], zero_arrival_posttick_s=z['first_arrival_posttick_elapsed_s']))
    assert len(pairs) == 420 and len(zero) == 14
    roles = list(dict.fromkeys(r['role'] for r in records))
    summary = []
    for role in roles:
        selected = [r for r in records if r['role'] == role]
        pair = [r for r in pairs if r['role'] == role]
        summary.append(dict(role=role, n=len(selected), arrived=sum(r['goal'] for r in selected),
                            arrival_phases=dict(Counter(r['arrival_permission_phase'] for r in selected if r['goal'])),
                            goal_gains=sum(p['goal_delta'] == 1 for p in pair), goal_losses=sum(p['goal_delta'] == -1 for p in pair),
                            gain_arrival_phases=dict(Counter(p['learned_arrival_phase'] for p in pair if p['goal_delta'] == 1)),
                            nominal=dict(gains=sum(p['goal_delta'] == 1 for p in pair if not p['force_N']),
                                         losses=sum(p['goal_delta'] == -1 for p in pair if not p['force_N'])),
                            disturbed=dict(gains=sum(p['goal_delta'] == 1 for p in pair if p['force_N']),
                                           losses=sum(p['goal_delta'] == -1 for p in pair if p['force_N']))))
    review = dict(summary=summary, original_world_goals=route_identity, raw_records=434,
                  distinct_raw_sha256=len({r['raw_sha256'] for r in records}), pairs=420,
                  old_1m_goal_labels_reproduced=434, old_minimum_distances_reproduced=434,
                  new_physics_ticks=0, new_training_steps=0, limits=plan['limits'])
    (root / 'review.json').write_text(json.dumps(review, indent=2) + '\n')
    (root / 'records.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')
    write_table(root / 'records.csv', records)
    write_table(root / 'zero_pairs.csv', pairs)
    if (root / 'independent_review.json').exists():
        agreement = compare_independent(records, json.loads((root / 'independent_review.json').read_text()))
        (root / 'independent_agreement.json').write_text(json.dumps(agreement, indent=2) + '\n')
    plot_summary(root, summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
