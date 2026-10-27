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
    args = parser.parse_args()
    root = args.run.resolve()
    rows = json.loads((root / 'results.json').read_text())['cases']
    plan = json.loads((root / 'plan.json').read_text())
    assert len(rows) == len(plan['cases']) == 24
    out = root / 'analysis'
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
        fault = bool(np.logical_or.reduce([trace[k][active].astype(bool) for k in
                                          ('nonfinite', 'lower_fault', 'policy_fault')]).any())
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
            ax.set(title=f'{route}; F={force:g} N', xlabel='World X (m)', ylabel='World Y (m)', aspect='equal')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
    fig.suptitle('Frozen H75: entry history x future alpha; 4 s contact-only diagnostic\n'
                 'Line style = future alpha; color = entry alpha; x = contact; o = observed endpoint')
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
    result = dict(summary=counts, records=records, same_state_contrasts=contrasts,
                  actual_lower_ticks=sum(r['active_ticks'] for r in records), training_steps=0,
                  scope='DEV mechanism only; four related complete entry states, no independent training repeat',
                  limitations=['No task completion claim or floor/zero advantage comparison',
                               'Entry state difference includes speed/pose/observer/history',
                               'Future alpha intervention is diagnostic, not user preference redefinition'])
    (out / 'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(summary=counts, ticks=result['actual_lower_ticks']), ensure_ascii=False))


if __name__ == '__main__':
    main()
