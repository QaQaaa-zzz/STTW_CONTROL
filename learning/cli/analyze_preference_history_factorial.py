"""Read all 24 branches, with shared-world-force XY and intervention diagnostics."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    rows = json.loads((root / 'results.json').read_text())['cases']
    plan = json.loads((root / 'plan.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    assert status['phase'] == 'completed' and status['completed_cases'] == 24
    assert len(rows) == len(plan['cases']) == 24
    identity_fields = ('state_id', 'route', 'entry_alpha', 'future_alpha', 'force_N', 'snapshot', 'world_direction')
    for observed, declared in zip(rows, plan['cases']):
        assert all(observed[k] == declared[k] for k in identity_fields)
    out = args.output.resolve() if args.output else root / 'analysis'
    out.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    routes = sorted({r['route'] for r in rows})
    fig, axes = plt.subplots(2, 3, figsize=(13, 9), layout='constrained')
    counts, records, traces = {}, [], {}
    for row in rows:
        file = root / (row['name'] + '.npz')
        with np.load(file) as z:
            trace = {k: z[k] for k in z.files}
        active = trace['active_tick'].astype(bool)
        t = trace['time'][active]
        assert len(t) and abs(float(t[0])) < .006
        assert np.allclose(np.diff(t), .005, atol=3e-6)
        contact = bool(trace['contact_failure'][active].any())
        trace_fault = bool(np.logical_or.reduce([trace[k][active].astype(bool) for k in
                                                ('nonfinite', 'lower_fault', 'policy_fault')]).any())
        fault = trace_fault or bool(row['final_fault'])
        label = 'FAULT' if fault else 'CONTACT' if contact else 'PASS' if len(t) == 800 else 'UNKNOWN'
        expected = dict(NO_FLOOR_CONTACT='PASS', FLOOR_CONTACT='CONTACT', ENGINEERING_ERROR='FAULT', UNKNOWN='UNKNOWN')
        assert label == expected[row['balance_label']]
        impulse = trace['applied_force_impulse'].sum(axis=0)
        target = row['force_N'] * .1 * np.asarray(row['world_direction'])
        # Contact can terminate the pulse: preserve incomplete exposure explicitly.
        full_pulse = len(t) >= 20
        if full_pulse:
            np.testing.assert_allclose(impulse, target, atol=.04, rtol=.01)
        upper = active & trace['upper_active'].astype(bool)
        record = {k: row[k] for k in ('state_id', 'route', 'entry_alpha', 'future_alpha', 'force_N')}
        record.update(label=label, active_ticks=int(active.sum()), upper_active_ticks=int(upper.sum()),
                      trace_fault=trace_fault, latched_final_fault=bool(row['final_fault']),
                      fallback_ticks=int((active & trace['fallback_active'].astype(bool)).sum()),
                      full_pulse_exposure=full_pulse, impulse_Ns=impulse.tolist(),
                      raw_sha256=hashlib.sha256(file.read_bytes()).hexdigest(), raw_file=str(file))
        records.append(record)
        traces[(row['route'], row['entry_alpha'], row['future_alpha'], row['force_N'])] = trace
        ax = axes[routes.index(row['route']), [0., -32., 32.].index(row['force_N'])]
        xy = trace['actual_xy'][active]
        color = '#0072B2' if row['entry_alpha'] == 0 else '#D55E00'
        ax.plot(xy[:, 0], xy[:, 1], color=color, ls='-' if row['future_alpha'] == 0 else '--',
                label=f"entry {row['entry_alpha']:g} / future {row['future_alpha']:g}")
        ax.scatter(*xy[-1], marker='x' if contact else 's' if fault else 'o', color=color, s=28)
    for i, route in enumerate(routes):
        with np.load(root / (route + '_path.npz')) as z:
            xy = z['xy']
        for j, force in enumerate((0., -32., 32.)):
            ax = axes[i, j]
            ax.plot(xy[:, 0], xy[:, 1], ':', color='0.55', lw=1, label='original path')
            observed = np.concatenate([traces[(route, a, b, force)]['actual_xy'][
                traces[(route, a, b, force)]['active_tick'].astype(bool)]
                for a in (0., 1.) for b in (0., 1.)])
            lo, hi = observed.min(axis=0)-.5, observed.max(axis=0)+.5
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set(title=f'{route}; F={force:g} N', xlabel='World X (m)', ylabel='World Y (m)', aspect='equal')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle('Frozen H75: entry history x future alpha; 4 s contact-only diagnostic\n'
                 'Line style = future alpha; color = entry alpha; x = contact; o = endpoint; reference view clipped')
    fig.savefig(out / 'all_xy.png', dpi=150)
    plt.close(fig)
    contrasts = []
    for route in routes:
        for entry in (0., 1.):
            for force in (0., -32., 32.):
                aa, bb = [traces[(route, entry, future, force)] for future in (0., 1.)]
                common = (aa['active_tick'].astype(bool) & bb['active_tick'].astype(bool)
                          & aa['upper_active'].astype(bool) & bb['upper_active'].astype(bool))
                finite = common & np.isfinite(aa['executed_policy_latent']).all(axis=1) & np.isfinite(bb['executed_policy_latent']).all(axis=1)
                delta = np.max(np.abs(aa['executed_policy_latent'][finite] - bb['executed_policy_latent'][finite]), axis=0).tolist() if finite.any() else None
                ref = np.max(np.abs(aa['governed'][common] - bb['governed'][common]), axis=0).tolist() if common.any() else None
                pair = [next(r for r in records if (r['route'], r['entry_alpha'], r['future_alpha'], r['force_N']) == (route, entry, future, force)) for future in (0., 1.)]
                contrasts.append(dict(route=route, entry_alpha=entry, force_N=force,
                                      future0=pair[0]['label'], future1=pair[1]['label'],
                                      common_upper_ticks=int(common.sum()), max_abs_latent_difference=delta,
                                      max_abs_reference_difference=ref))
    for entry in (0., 1.):
        for future in (0., 1.):
            counts[f'entry{entry:g}_future{future:g}'] = dict(Counter(r['label'] for r in records if r['entry_alpha'] == entry and r['future_alpha'] == future))
    factorial = []
    for route in routes:
        for force in (0., -32., 32.):
            labels = {(entry, future): next(r['label'] for r in records if
                      (r['route'], r['entry_alpha'], r['future_alpha'], r['force_N']) ==
                      (route, entry, future, force)) for entry in (0., 1.) for future in (0., 1.)}
            known = all(label in ('PASS', 'CONTACT') for label in labels.values())
            values = {k: int(label == 'PASS') for k, label in labels.items()} if known else None
            factorial.append(dict(route=route, force_N=force,
                                  outcomes={f'entry{a:g}_future{b:g}': v for (a, b), v in labels.items()},
                                  entry_effect_by_future=[values[(1., b)]-values[(0., b)] for b in (0., 1.)] if known else None,
                                  future_effect_by_entry=[values[(a, 1.)]-values[(a, 0.)] for a in (0., 1.)] if known else None,
                                  interaction=values[(1., 1.)]-values[(1., 0.)]-values[(0., 1.)]+values[(0., 0.)] if known else None))
        diag, dax = plt.subplots(5, 3, figsize=(13, 13), layout='constrained')
        channels = [('phi', None, 'Roll (rad)'), ('actual_forward_speed', None, 'Actual speed (m/s)'),
                    ('governed', 0, 'Published speed (m/s)'), ('actual_delta', None, 'Actual steer (rad)'),
                    ('governed', 1, 'Published steer (rad)')]
        for col, force in enumerate((0., -32., 32.)):
            for entry in (0., 1.):
                for future in (0., 1.):
                    trace = traces[(route, entry, future, force)]
                    active = trace['active_tick'].astype(bool)
                    for line, (key, channel, label) in enumerate(channels):
                        value = trace[key][active]
                        if channel is not None:
                            value = value[:, channel]
                        dax[line, col].plot(trace['time'][active], value,
                                            color='#0072B2' if entry == 0 else '#D55E00',
                                            ls='-' if future == 0 else '--',
                                            label=f'entry {entry:g} / future {future:g}')
                        dax[line, col].set(xlabel='Suffix time (s)', ylabel=label)
                        dax[line, col].axvspan(0., .1, color='0.8', alpha=.3)
                        dax[line, col].grid(alpha=.2)
            dax[0, col].set_title(f'F={force:g} N; pulse 0–0.1 s')
        dax[0, 0].legend(fontsize=7)
        diag.suptitle(route + ': observed contact/endpoint stops each trace; no tail extrapolation')
        diag.savefig(out / (route + '_physical_diagnostics.png'), dpi=120)
        plt.close(diag)
    result = dict(summary=counts, records=records, same_state_contrasts=contrasts,
                  factorial_descriptive_contrasts=factorial,
                  actual_lower_ticks=sum(r['active_ticks'] for r in records), training_steps=0,
                  allocated_lower_slots=21600,
                  scope='DEV mechanism only; four related complete entry states, no independent training repeat',
                  limitations=['No task completion claim or floor/zero advantage comparison',
                               'Entry state difference includes speed/pose/observer/history',
                               'Future alpha intervention is diagnostic, not user preference redefinition'])
    (out / 'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(summary=counts, ticks=result['actual_lower_ticks']), ensure_ascii=False))


if __name__ == '__main__':
    main()
