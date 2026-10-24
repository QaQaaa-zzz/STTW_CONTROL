"""Measure force-induced speed changes against each candidate's own zero branch."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    status = json.loads((root / 'status.json').read_text())
    rows = json.loads((root / 'results.json').read_text())['cases']
    if status['phase'] != 'completed' or len(rows) != 45:
        raise ValueError('requires the completed original 45-case opportunity panel')
    cache, hashes = {}, {}
    for row in rows:
        file = root / (row['name'] + '.npz')
        with np.load(file) as data:
            cache[row['name']] = {k: data[k] for k in data.files}
        hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
    states = sorted({r['state_id'] for r in rows})
    colors = dict(zip(['continue_original', 'speed_plus', 'speed_minus', 'steer_plus', 'steer_minus'],
                      ['black', 'tab:red', 'tab:blue', 'tab:green', 'tab:orange']))
    figure, axes = plt.subplots(3, 2, figsize=(15, 12), layout='constrained')
    wheel_figure, wheel_axes = plt.subplots(3, 2, figsize=(15, 12), layout='constrained')
    result = []
    for row in rows:
        if row['force_N'] == 0:
            continue
        zero = next(r for r in rows if r['state_id'] == row['state_id'] and
                    r['candidate'] == row['candidate'] and r['force_N'] == 0)
        if zero['snapshot_sha256'] != row['snapshot_sha256']:
            raise ValueError('different physical prefix')
        trace, baseline = cache[row['name']], cache[zero['name']]
        count = min(len(trace['time']), len(baseline['time']))
        np.testing.assert_allclose(trace['relative_time'][:count], baseline['relative_time'][:count], atol=2e-6, rtol=0)
        delta_speed = trace['actual_forward_speed'][:count] - baseline['actual_forward_speed'][:count]
        delta_wheel = trace['wheel_speed_proxy'][:count] - baseline['wheel_speed_proxy'][:count]
        measurements = []
        for seconds in [.1, .2, .4]:
            index = round(seconds / .005) - 1
            if index >= count:
                measurements.append(dict(seconds=seconds, available=False))
                continue
            measurements.append(dict(seconds=seconds, available=True,
                                     actual_speed_m_s=float(trace['actual_forward_speed'][index]),
                                     own_zero_speed_m_s=float(baseline['actual_forward_speed'][index]),
                                     force_induced_speed_change_m_s=float(delta_speed[index]),
                                     wheel_speed_proxy_m_s=float(trace['wheel_speed_proxy'][index]),
                                     own_zero_wheel_proxy_m_s=float(baseline['wheel_speed_proxy'][index]),
                                     force_induced_wheel_proxy_change_m_s=float(delta_wheel[index]),
                                     actual_steer_rad=float(trace['actual_delta'][index]),
                                     own_zero_steer_rad=float(baseline['actual_delta'][index])))
        result.append(dict(state_id=row['state_id'], candidate=row['candidate'], force_N=row['force_N'],
                           measurements=measurements, balance_label=row['balance_label'], task=row['task_recovered']))
        ax = axes[states.index(row['state_id']), 0 if row['force_N'] < 0 else 1]
        show = trace['relative_time'][:count] <= 1.2
        t = trace['relative_time'][:count][show]
        ax.plot(t, delta_speed[show], color=colors[row['candidate']], label=row['candidate'])
        if row['event_contact'] and show[-1]:
            ax.scatter(t[-1], delta_speed[show][-1], marker='x', color=colors[row['candidate']])
        ax.set_title(f"{row['state_id']} F={row['force_N']:+g} N", fontsize=9)
        wheel_ax = wheel_axes[states.index(row['state_id']), 0 if row['force_N'] < 0 else 1]
        wheel_ax.plot(t, delta_wheel[show], color=colors[row['candidate']], label=row['candidate'])
        if row['event_contact'] and show[-1]:
            wheel_ax.scatter(t[-1], delta_wheel[show][-1], marker='x', color=colors[row['candidate']])
        wheel_ax.set_title(f"{row['state_id']} F={row['force_N']:+g} N", fontsize=9)
    for ax in [*axes.flat, *wheel_axes.flat]:
        ax.axhline(0, color='gray', ls='--')
        ax.axvspan(0, .1, color='gray', alpha=.1)
        ax.axvline(.4, color='gray', ls=':')
        ax.set(xlabel='seconds after same complete state', ylabel='change versus own zero-force branch (m/s)')
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    figure.suptitle('Observed longitudinal response; x=contact. Each candidate compared with its own zero branch.')
    wheel_figure.suptitle('Rear-wheel speed proxy response (model input); x=contact. Not true body speed.')
    args.output.mkdir(exist_ok=False)
    figure.savefig(args.output / 'force_induced_speed_change.png')
    wheel_figure.savefig(args.output / 'force_induced_wheel_speed_change.png')
    plt.close(figure)
    plt.close(wheel_figure)
    report = dict(cases=result, source_raw_sha256=hashes, new_physics=0, fits=0,
                  interpretation='Descriptive matched counterfactual response. Policy/steering/drive all interact; does not isolate velocity as the sole cause or identify a new dynamics model. Measurement .1/.2/.4s uses the last completed lower tick of that interval. Domain-hold physical data remain actual; no task reward inferred.')
    (args.output / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    for row in result:
        if row['state_id'].startswith('right') and '_v1.5_' in row['state_id'] and row['force_N'] < 0:
            print(json.dumps(row))


if __name__ == '__main__':
    main()
