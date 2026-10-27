"""Join 24 new simple controls to 24 authenticated H75 branches, without reruns."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    source = Path(plan['source_run'])
    for name, expected in plan['source_sha256'].items():
        assert sha(name) == expected, name
    assert sha(source / 'plan.json') == plan['source_plan_sha256']
    status = json.loads((root / 'status.json').read_text())
    assert status['phase'] == 'completed' and status['completed_cases'] == 24
    raw_rows = json.loads((root / 'results.json').read_text())['cases']
    assert len(raw_rows) == len(plan['cases']) == 24
    assert (json.loads((root / 'manifest.json').read_text())['physical_identity'] ==
            json.loads((source / 'manifest.json').read_text())['physical_identity'])
    cached = json.loads((source / 'analysis_verified/review.json').read_text())['records']
    old_rows = json.loads((source / 'results.json').read_text())['cases']
    old_plan = json.loads((source / 'plan.json').read_text())
    for field in ('source_declaration', 'path_evidence_directory', 'sensor_contract',
                  'pulse_duration_s', 'observation_s', 'direction_contract', 'termination'):
        assert plan[field] == old_plan[field], field
    assert len(old_rows) == len(old_plan['cases']) == 24
    for observed, declared in zip(old_rows, old_plan['cases']):
        assert all(observed[key] == value for key, value in declared.items())
    cache_map = {(r['state_id'], r['future_alpha'], r['force_N']): r for r in cached}
    records, traces = [], {}
    pending = []
    for actual, declared in zip(raw_rows, plan['cases']):
        assert all(actual[key] == value for key, value in declared.items())
        pending.append((actual, root / (actual['name'] + '.npz'), actual['baseline_role'], False))
    for row in old_rows:
        cached_record = cache_map[(row['state_id'], row['future_alpha'], row['force_N'])]
        file = Path(cached_record['raw_file'])
        assert sha(file) == cached_record['raw_sha256']
        pending.append((row, file, f"future{row['future_alpha']:g}", True))
    from sttw_control.reference_reserve_gate import GateConfig
    assert vars(GateConfig()) == plan['gate_config']
    boost = np.float32(plan['gate_config']['boost_latent'])
    for row, file, role, reused in pending:
        with np.load(file) as z:
            trace = {key: z[key] for key in z.files}
        active = trace['active_tick'].astype(bool)
        t = trace['time'][active]
        assert len(t) and np.allclose(t, np.arange(len(t)) * .005, atol=3e-6)
        contact = bool(trace['contact_failure'][active].any())
        fault = bool(row['final_fault']) or any(bool(trace[k][active].any()) for k in ('nonfinite', 'lower_fault', 'policy_fault'))
        label = 'FAULT' if fault else 'CONTACT' if contact else 'PASS' if len(t) == 800 else 'UNKNOWN'
        assert label == {'NO_FLOOR_CONTACT': 'PASS', 'FLOOR_CONTACT': 'CONTACT',
                         'ENGINEERING_ERROR': 'FAULT', 'UNKNOWN': 'UNKNOWN'}[row['balance_label']]
        upper = active & trace['upper_active'].astype(bool)
        original = trace['original_policy_latent'][upper]
        executed = trace['executed_policy_latent'][upper]
        if role == 'zero_latent':
            np.testing.assert_array_equal(original, np.zeros_like(original))
            np.testing.assert_array_equal(executed, np.zeros_like(executed))
        elif role == 'speed_support':
            np.testing.assert_array_equal(executed[:, 1], original[:, 1])
            np.testing.assert_allclose(executed[:, 0], np.maximum(original[:, 0], boost), atol=1e-6, rtol=0)
            assert trace['gate_active'][upper].all()
        impulse = trace['applied_force_impulse'].sum(axis=0)
        if len(t) >= 20:
            np.testing.assert_allclose(impulse, row['force_N'] * .1 * np.asarray(row['world_direction']), atol=.04, rtol=.01)
        key = (row['state_id'], row['force_N'], role)
        assert key not in traces
        traces[key] = trace
        records.append(dict(state_id=row['state_id'], route=row['route'], entry_alpha=row['entry_alpha'],
                            force_N=row['force_N'], role=role, label=label, reused=reused,
                            active_ticks=int(active.sum()), upper_active_ticks=int(upper.sum()),
                            fallback_ticks=int((active & trace['fallback_active'].astype(bool)).sum()),
                            full_pulse_exposure=len(t) >= 20,
                            raw_file=str(file), raw_sha256=sha(file)))
    roles = ('future0', 'future1', 'zero_latent', 'speed_support')
    summary = {role: dict(n=12, outcomes=dict(Counter(r['label'] for r in records if r['role'] == role)),
                         disturbed=dict(Counter(r['label'] for r in records if r['role'] == role and r['force_N']))) for role in roles}
    event_rows = []
    for sid, force in sorted({(r['state_id'], r['force_N']) for r in records}):
        outcomes = {role: next(r['label'] for r in records if (r['state_id'], r['force_N'], r['role']) == (sid, force, role)) for role in roles}
        event_rows.append(dict(state_id=sid, force_N=force, outcomes=outcomes,
                               original_rescue=outcomes['future0'] == 'CONTACT' and outcomes['future1'] == 'PASS'))
    paired = {}
    for baseline in ('zero_latent', 'speed_support'):
        paired[baseline] = dict(
            future1_pass_baseline_contact=[e['state_id']+f"/F{e['force_N']:g}" for e in event_rows if e['outcomes']['future1'] == 'PASS' and e['outcomes'][baseline] == 'CONTACT'],
            baseline_pass_future1_contact=[e['state_id']+f"/F{e['force_N']:g}" for e in event_rows if e['outcomes']['future1'] == 'CONTACT' and e['outcomes'][baseline] == 'PASS'],
            original_three_rescues_covered=sum(e['original_rescue'] and e['outcomes'][baseline] == 'PASS' for e in event_rows))
        paired[baseline]['unknown_or_fault_pairs'] = [e['state_id']+f"/F{e['force_N']:g}" for e in event_rows
            if e['outcomes']['future1'] in ('UNKNOWN', 'FAULT') or e['outcomes'][baseline] in ('UNKNOWN', 'FAULT')]
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(roles, ('#0072B2', '#D55E00', '#009E73', '#CC79A7')))
    styles = dict(zip(roles, ('-', '--', '-.', ':')))
    states = sorted({r['state_id'] for r in records})
    fig, axes = plt.subplots(4, 3, figsize=(14, 16), layout='constrained')
    for i, sid in enumerate(states):
        sample = next(r for r in records if r['state_id'] == sid)
        with np.load(root / (sample['route'] + '_path.npz')) as z:
            path_xy = z['xy']
        diag, dax = plt.subplots(5, 3, figsize=(13, 13), layout='constrained')
        channels = [('phi', None, 'Roll (rad)'), ('actual_forward_speed', None, 'Actual speed (m/s)'),
                    ('governed', 0, 'Published speed (m/s)'), ('actual_delta', None, 'Actual steer (rad)'),
                    ('governed', 1, 'Published steer (rad)')]
        for j, force in enumerate((0., -32., 32.)):
            ax, observed = axes[i, j], []
            for role in roles:
                trace = traces[(sid, force, role)]
                record = next(r for r in records if (r['state_id'], r['force_N'], r['role']) == (sid, force, role))
                active = trace['active_tick'].astype(bool)
                xy = trace['actual_xy'][active]
                observed.append(xy)
                ax.plot(xy[:, 0], xy[:, 1], color=colors[role], ls=styles[role], label=role)
                ax.scatter(*xy[-1], marker='x' if record['label'] == 'CONTACT' else 'o', color=colors[role], s=25)
                for line, (key, channel, label) in enumerate(channels):
                    values = trace[key][active]
                    if channel is not None:
                        values = values[:, channel]
                    dax[line, j].plot(trace['time'][active], values, color=colors[role], ls=styles[role], label=role)
                    dax[line, j].set(xlabel='Suffix time (s)', ylabel=label)
            all_xy = np.concatenate(observed)
            lo, hi = all_xy.min(axis=0)-.5, all_xy.max(axis=0)+.5
            ax.plot(path_xy[:, 0], path_xy[:, 1], ':', color='0.6', lw=1, label='original path')
            ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), aspect='equal',
                   title=f"{sample['route']} entry{sample['entry_alpha']:g}, F={force:g} N", xlabel='World X (m)', ylabel='World Y (m)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
            dax[0, j].set_title(f'F={force:g} N')
            for line in range(5):
                dax[line, j].axvspan(0., .1, color='0.8', alpha=.3)
                dax[line, j].grid(alpha=.2)
        dax[0, 0].legend(fontsize=7)
        diag.suptitle(sid + ': four complete suffix rules, not task completion evaluation')
        diag.savefig(out / (sid + '_diagnostics.png'), dpi=120)
        plt.close(diag)
    fig.suptitle('Matched H75 and simple controls: 48 traces (24 new + 24 cached)\n'
                 'x = real contact; o = observed endpoint; original future path view clipped')
    fig.savefig(out / 'all_xy.png', dpi=150)
    plt.close(fig)
    result = dict(summary=summary, events=event_rows, paired=paired, records=records,
                  new_lower_ticks=sum(r['active_ticks'] for r in records if not r['reused']),
                  inherited_lower_ticks=sum(r['active_ticks'] for r in records if r['reused']),
                  new_allocated_slots=21600, training_steps=0,
                  limitations=['Four related entry states, not independent training repeats',
                               'No original goal or user-alpha task-cost evaluation',
                               'Suffix effect includes domain-exit/held reference; support failure is not proof of neural necessity'])
    assert result['new_lower_ticks'] == status['active_lower_ticks']
    (out / 'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('summary', 'paired', 'new_lower_ticks')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
