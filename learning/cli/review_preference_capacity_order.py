"""Read-only ordering audit of authenticated held-reference capability branches.

No environment construction, pickle deserialization, predictor fitting or simulation.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    src = args.source.resolve()
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    inputs = {}

    def read(name):
        path = src / name
        inputs[str(path)] = digest(path)
        return json.loads(path.read_text())

    status = read('scan/status.json')
    assert status['phase'] == 'completed'
    plan = read('scan/scan_plan.json')
    manifest = read('scan/manifest.json')
    historical = read('scan/analysis.json')
    prefixes = read('scan/prefixes.json')['cases']
    results = read('scan/results.json')['cases']
    bundle = read('prefix_bundle.json')['cases']
    receipt = read('capture_receipt.json')
    assert manifest['plan'] == plan
    assert manifest['no_upper_inference'] and manifest['complete_lower_history_preserved']
    assert manifest['termination_contract'] == 'contact_only_v1'
    assert manifest['roll_limit_is_terminal'] is False
    assert plan['observation_s'] == 4 and plan['pulse_duration_s'] == .1
    assert plan['pulse_delay_s'] == 0 and 'reference_profiles_file' not in plan
    assert receipt['all_first_control_crossings'] and receipt['full_history_preserved']
    # Authenticate archived runtime against launch seals, rather than current dirty code.
    sealed = read('plan.json')['source_sha256']
    archive = {}
    for original, expected in sealed.items():
        matches = [p for p in (src / 'source_at_launch').iterdir()
                   if p.name.split('_', 1)[-1] == Path(original).name]
        assert len(matches) == 1 and digest(matches[0]) == expected, original
        archive[str(matches[0])] = expected
    raw_hashes = {Path(x['path']).name: x['sha256'] for x in historical['source_sha256']}
    by_prefix = {p['state_id']: p for p in prefixes}
    assert len(prefixes) == len(by_prefix) == 8
    for p, b in zip(prefixes, bundle):
        assert b['state_id'] == p['state_id'] + '_phase0.5'
        assert digest(p['snapshot']) == b['snapshot_sha256'] == plan['source_sha256'][p['snapshot']]
        assert b['fraction'] == p['phase_fraction'] == .5
        assert b['previous_control_progress_m'] < b['target_progress_m'] <= b['actual_progress_m']
        assert p['future_middle_actor_active'] is False
        assert b['complete_upper_history']
    expected_ticks = round(plan['observation_s'] / .005)
    rows = []
    directions = {}
    assert len(results) == len(plan['cases']) == len(plan['forces_N']) == 72
    for i, (case, old) in enumerate(zip(plan['cases'], results)):
        assert case['force_N'] == old['force_N'] == plan['forces_N'][i]
        assert manifest['case_snapshot_paths'][i] == by_prefix[case['state_id']]['snapshot']
        assert case['reference'] == manifest['case_references_m_s_rad'][i]
        path = src / 'scan' / (old['name'] + '.npz')
        assert digest(path) == raw_hashes[path.name]
        inputs[str(path)] = raw_hashes[path.name]
        with np.load(path, allow_pickle=False) as z:
            active = z['active_tick'].astype(bool)
            n = int(active.sum())
            assert np.array_equal(active, np.arange(len(active)) < n)
            t = z['time'][active]
            np.testing.assert_allclose(t, np.arange(n) * .005, atol=3e-7, rtol=0)
            contact = bool(z['contact_failure'][active].any())
            fault = bool(z['nonfinite'][active].any() or z['lower_fault'][active].any())
            fault |= not bool(np.isfinite(z['peak_roll'][active]).all())
            label = 'UNKNOWN' if fault else 'CONTACT' if contact else 'PASS' if n == expected_ticks else 'UNKNOWN'
            np.testing.assert_array_equal(z['reference'][active], np.broadcast_to(np.asarray(case['reference'], np.float32), (n, 2)))
            impulse = z['applied_force_impulse'].sum(0)
            np.testing.assert_allclose(impulse, old['impulse_Ns'], atol=2e-5)
        mapping = {'FLOOR_CONTACT': 'CONTACT', 'NO_FLOOR_CONTACT': 'PASS', 'ENGINEERING_ERROR': 'UNKNOWN', 'UNKNOWN': 'UNKNOWN'}
        assert label == mapping[old['balance_label']] and n == old['active_ticks']
        if case['force_N']:
            direction = np.asarray(old['expected_impulse_Ns']) / (case['force_N'] * plan['pulse_duration_s'])
            np.testing.assert_allclose(np.linalg.norm(direction), 1, atol=1e-6)
            if case['state_id'] in directions:
                np.testing.assert_allclose(direction, directions[case['state_id']], atol=1e-7)
            directions[case['state_id']] = direction
        rows.append(dict(case, label=label, active_ticks=n, raw=str(path)))
    assert sum(x['active_ticks'] for x in rows) == status['active_lower_ticks']
    states = []
    for sid, p in by_prefix.items():
        rr = {r['force_N']: r for r in rows if r['state_id'] == sid}
        assert set(rr) == {0., -16., 16., -32., 32., -48., 48., -64., 64.}
        capacity = {}
        for sign in (-1, 1):
            last = 0. if rr[0]['label'] == 'PASS' else None
            blocked = last is None
            first_nonpass = None
            reversal = []
            for a in (16., 32., 48., 64.):
                label = rr[a * sign]['label']
                if blocked and label == 'PASS':
                    reversal.append(a)
                if not blocked:
                    if label == 'PASS':
                        last = a
                    else:
                        first_nonpass = dict(amplitude_N=a, label=label)
                        blocked = True
            capacity[str(sign)] = dict(grid_prefix_N=last, first_nonpass=first_nonpass,
                                       later_passes_N=reversal, top_grid_pass=rr[64. * sign]['label'] == 'PASS')
        states.append(dict(state_id=sid, route=next(r['route'] for r in rows if r['state_id'] == sid),
                           requested_speed=next(r['requested_speed'] for r in rows if r['state_id'] == sid),
                           alpha=p['alpha'], true_speed=p['true_forward_speed'], task_time_s=p['task_time_s'],
                           held_reference=p['held_reference'], sensor=p['sensor'], eso=p['eso'],
                           world_force_direction=directions[sid].tolist(), capacity=capacity))
    pairs = []
    for route, speed in sorted({(s['route'], s['requested_speed']) for s in states}):
        a, b = sorted([s for s in states if (s['route'], s['requested_speed']) == (route, speed)], key=lambda s: s['alpha'])
        assert (a['alpha'], b['alpha']) == (0., 1.)
        ar = {r['force_N']: r['label'] for r in rows if r['state_id'] == a['state_id'] and r['force_N']}
        br = {r['force_N']: r['label'] for r in rows if r['state_id'] == b['state_id'] and r['force_N']}
        gains = [f for f in ar if ar[f] == 'CONTACT' and br[f] == 'PASS']
        losses = [f for f in ar if ar[f] == 'PASS' and br[f] == 'CONTACT']
        angle = float(np.rad2deg(np.arccos(np.clip(np.dot(a['world_force_direction'], b['world_force_direction']), -1, 1))))
        pairs.append(dict(route=route, requested_speed=speed, alpha0=a['state_id'], alpha1=b['state_id'],
                          speed_delta=b['true_speed']-a['true_speed'], world_direction_difference_deg=angle,
                          pass_gains_alpha1=gains, pass_losses_alpha1=losses,
                          prefix_delta_N={d:b['capacity'][d]['grid_prefix_N']-a['capacity'][d]['grid_prefix_N'] for d in ('-1','1')}))
    ordered = sorted(states, key=lambda s:s['true_speed'])
    # Post-hoc compatibility check, not a trained speed classifier or causal test.
    comparisons, reversals = 0, []
    for a, b in itertools.combinations(ordered, 2):
        for sign in ('-1', '1'):
            x, y = a['capacity'][sign]['grid_prefix_N'], b['capacity'][sign]['grid_prefix_N']
            if x is not None and y is not None:
                comparisons += 1
                if x > y:
                    reversals.append([a['state_id'], b['state_id'], sign])
    result = dict(states=states, alpha_pairs=pairs, rows=rows,
                  counts={label:sum(r['label']==label for r in rows) for label in ('PASS','CONTACT','UNKNOWN')},
                  disturbed_counts={label:sum(r['force_N']!=0 and r['label']==label for r in rows) for label in ('PASS','CONTACT','UNKNOWN')},
                  speed_order=dict(state_direction_comparisons=comparisons, reversals=reversals,
                                   ordered_state_ids=[s['state_id'] for s in ordered]),
                  historical_active_ticks=sum(r['active_ticks'] for r in rows),
                  source_sha256=inputs, archived_source_sha256=archive,
                  analysis_script_sha256=digest(__file__), new_physics_steps=0, optimizer_updates=0,
                  status='completed', role='Exploratory reused DEV, 8 related states and dependent force branches; no fitted predictor or new control method.')
    (out/'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('counts','disturbed_counts','speed_order','historical_active_ticks')}, indent=2))


if __name__ == '__main__':
    main()
