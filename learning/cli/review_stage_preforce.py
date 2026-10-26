"""Exploratory paired pre-force consequences, without fitting a risk gate.

Use the already audited00/11 endpoints of the fixed stage experiment.  Outcome
strata are retrospective descriptions, never controller inputs or test labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = ['actual_forward_speed', 'phi', 'phi_dot', 'actual_delta',
          'path_cross_track', 'path_heading_error', 'reference',
          'nominal_reference', 'correction_offset']


def sample_prefix(raw, onset, dt=.005, window=.5):
    """Use logged post-tick states before any measured force impulse.

    The planned onset defines a retrospective common sampling clock. It is
    unavailable to the proposed deployment controller. Reject contamination,
    rather than silently moving the measurement time using future force data.
    """
    ticks = np.asarray(raw['physical_tick'])
    active = np.asarray(raw['active_tick'], bool)
    before = active & (ticks < onset) & (ticks >= onset - round(window / dt))
    idx = np.flatnonzero(before)
    if len(idx) != round(window / dt) or ticks[idx[-1]] != onset - 1:
        raise ValueError('complete fixed pre-onset window required')
    impulse = np.asarray(raw['applied_force_impulse'])
    if np.any(impulse[active & (ticks < onset)] != 0):
        raise ValueError('measured force contaminates planned pre-onset window')
    force = np.flatnonzero(active & np.any(impulse != 0, axis=-1))
    result = dict(sample_task_tick=int(ticks[idx[-1]]),
                  first_measured_force_task_tick=int(ticks[force[0]]) if len(force) else None,
                  window_s=window, samples=len(idx),
                  tv_sample_intervals=len(idx)-1,
                  tv_sample_span_s=float((ticks[idx[-1]]-ticks[idx[0]])*dt))
    for name in FIELDS:
        values = np.asarray(raw[name])[idx]
        if not np.all(np.isfinite(values)):
            raise ValueError('nonfinite pre-force measurement')
        result[name + '_endpoint'] = np.asarray(values[-1]).tolist()
    result.update(
        prewindow_peak_abs_roll_rad=float(np.max(np.abs(raw['phi'][idx]))),
        prewindow_peak_abs_roll_rate_rad_s=float(np.max(np.abs(raw['phi_dot'][idx]))),
        prewindow_actual_steer_tv_rad=float(np.abs(np.diff(raw['actual_delta'][idx])).sum()),
        prewindow_reference_steer_tv_rad=float(np.abs(np.diff(raw['reference'][idx, 1])).sum()),
        prewindow_nominal_steer_tv_rad=float(np.abs(np.diff(raw['nominal_reference'][idx, 1])).sum()))
    return result


def contrast(on, off, key):
    return float(on[key]) - float(off[key])


def review(root, output=None):
    root = Path(root).resolve()
    for file in ['dispatch_status.json', 'analysis/status.json']:
        if json.loads((root / file).read_text())['phase'] != 'completed':
            raise ValueError('completed original analysis required')
    plan = json.loads((root / 'plan.json').read_text())
    if plan['experiment'] != 'steering_stage_mechanism_v1':
        raise ValueError('fixed stage experiment required')
    source = root / 'analysis/results.json'
    rows = json.loads(source.read_text())
    groups = {}
    for row in rows:
        if row['role'] not in ['goal_zero_feedback', 'goal_steer_feedback']:
            continue
        key = (row['context'], row['alpha'], row['event_id'])
        cell = groups.setdefault(key, {})
        if row['role'] in cell:
            raise ValueError('duplicate endpoint event')
        cell[row['role']] = row
    if len(groups) != 180:
        raise ValueError('all180 original events required')
    records = []
    for key, cell in groups.items():
        off, on = [cell[role] for role in ['goal_zero_feedback', 'goal_steer_feedback']]
        for field in ['pulse_onset_task_tick', 'force_N', 'world_direction']:
            if off[field] != on[field]:
                raise ValueError('unmatched event')
        measurements = {}
        for name, row in [('off', off), ('on', on)]:
            with np.load(row['raw_file'], allow_pickle=False) as z:
                raw = {k: z[k] for k in FIELDS + ['active_tick', 'physical_tick', 'applied_force_impulse']}
            measurements[name] = sample_prefix(raw, row['pulse_onset_task_tick'])
        survived = [row['physical_outcome'] == 'survived' for row in [off, on]]
        if any(row['physical_outcome'] not in ['survived', 'contact'] for row in [off, on]):
            raise ValueError('retain unknown; do not force outcome group')
        group = ('survival_gain' if survived == [False, True] else
                 'survival_loss' if survived == [True, False] else
                 'both_survive' if all(survived) else 'both_contact')
        records.append(dict(context=key[0], alpha=key[1], event_id=key[2],
            prefix_name=off['prefix_name'], geometry_name=off['geometry_name'],
            force_N=off['force_N'], pulse_onset_task_tick=off['pulse_onset_task_tick'],
            physical_group=group, off_arrived=off['goal_arrived'], on_arrived=on['goal_arrived'],
            off_active_lower_ticks_to_contact=off['physical_ticks'] if not survived[0] else None,
            on_active_lower_ticks_to_contact=on['physical_ticks'] if not survived[1] else None,
            off_raw_sha256=off['raw_sha256'], on_raw_sha256=on['raw_sha256'],
            off_raw_file=off['raw_file'], on_raw_file=on['raw_file'], **measurements))
    metric_keys = ['actual_forward_speed_endpoint', 'phi_endpoint', 'phi_dot_endpoint',
                   'actual_delta_endpoint', 'path_cross_track_endpoint', 'path_heading_error_endpoint',
                   'prewindow_peak_abs_roll_rad', 'prewindow_peak_abs_roll_rate_rad_s',
                   'prewindow_actual_steer_tv_rad', 'prewindow_reference_steer_tv_rad',
                   'prewindow_nominal_steer_tv_rad']
    def summarize(selected):
        return dict(events=len(selected), differences_on_minus_off={k: dict(
            median=float(np.median([contrast(r['on'], r['off'], k) for r in selected])),
            minimum=float(min(contrast(r['on'], r['off'], k) for r in selected)),
            maximum=float(max(contrast(r['on'], r['off'], k) for r in selected)))
            for k in metric_keys} if selected else {})
    disturbed = [r for r in records if r['force_N'] != 0]
    summary = {g: summarize([r for r in disturbed if r['physical_group'] == g])
               for g in ['survival_gain', 'survival_loss', 'both_survive', 'both_contact']}
    out = Path(output).resolve() if output is not None else root / 'analysis/preforce_mechanism'
    out.mkdir(exist_ok=False)
    result = dict(records=records, disturbed_summary=summary,
        nominal_summary=summarize([r for r in records if r['force_N'] == 0]),
        source_results_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physics_steps=0, optimizer_updates=0, visual_review_complete=False,
        scope='Exploratory after endpoint outcomes were seen. Fixed0.5s pre-onset window with100 samples and99 TV intervals spanning0.495s,180 paired original00/11 events; no trajectory selection for measurement, smoothing, fitted threshold, classifier, gate, counterfactual state swap or new simulation. Physical fields post-tick; references pre-tick. Active lower ticks to contact are transition counts, not zero-origin log timestamps. Original onset and outcome groups are privileged retrospective labels. Actual forward speed is simulation telemetry, not a claim of deployment observability or actor input availability. Shared bank3 ancestor and three correlated alpha, no independent confidence intervals; association of individual state fields is not causal attribution. Missing/nonfinite windows cause an explicit error, not imputation.')
    (out / 'review.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    render(out, disturbed)
    print(json.dumps(summary, ensure_ascii=False))


def render(out, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = [('actual_forward_speed_endpoint', 'Speed (m/s)'),
               ('phi_endpoint', 'Roll (rad)'), ('phi_dot_endpoint', 'Roll rate (rad/s)'),
               ('actual_delta_endpoint', 'Steering angle (rad)'),
               ('prewindow_actual_steer_tv_rad', 'Actual steering TV over0.5s (rad)'),
               ('prewindow_reference_steer_tv_rad', 'Reference steering TV over0.5s (rad)')]
    styles = [('both_survive', '#777777', '.', 16), ('both_contact', '#0072B2', 's', 27),
              ('survival_gain', '#009E73', '^', 65), ('survival_loss', '#D55E00', 'x', 65)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), layout='constrained')
    for ax, (key, label) in zip(axes.flat, metrics):
        allvalues = [float(r[arm][key]) for r in records for arm in ['off', 'on']]
        low, high = min(allvalues), max(allvalues)
        pad = max((high - low) * .08, .001)
        ax.plot([low-pad, high+pad], [low-pad, high+pad], 'k--', lw=.8, label='Equality')
        for group, color, marker, size in styles:
            selected = [r for r in records if r['physical_group'] == group]
            ax.scatter([r['off'][key] for r in selected], [r['on'][key] for r in selected],
                color=color, marker=marker, s=size, label=f'{group} n={len(selected)}', alpha=.8)
        ax.set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad),
            xlabel='Zero neural: '+label, ylabel='Neural steering: '+label)
        ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Paired pre-force measurements,144 disturbed events (correlated DEV)\n'
                 'Outcome-colored retrospective diagnosis; no threshold fitting or risk predictor')
    fig.savefig(out / 'paired_preforce_state.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    review(args.run, args.output)
