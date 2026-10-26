"""Read-only timing diagnostics; observations do not establish intervention effects."""
import json
from pathlib import Path

import numpy as np

from .preference_best import digest, atomic_json


def trace_diagnostics(raw, series, *, onset_tick, dt, horizon_ticks):
    active = np.asarray(raw['active_tick'], bool)
    ticks = np.asarray(raw['physical_tick'])[active]
    if not np.array_equal(ticks, np.arange(len(ticks))):
        raise ValueError('complete clock prefix required')
    if len(series['path_error']) != len(ticks):
        raise ValueError('derived geometry must match executed ticks')
    gate = np.asarray(raw['relay_active'], bool)[active]
    legal = np.asarray(raw['actor_action_valid'], bool)[active]
    rate = np.asarray(raw['phi_dot'])[active]
    path = np.asarray(series['path_error'])
    heading = np.asarray(series['heading_error'])
    post = ticks >= onset_tick
    supported = np.flatnonzero(gate)
    # Each saved pose is the END of its executed tick. A post-tick snapshot
    # must never be represented as a deployable pre-decision observation.
    def snapshot(i):
        if i is None or i < 0 or i >= len(ticks):
            return None
        after = ticks > ticks[i]
        return dict(end_of_tick=int(ticks[i]), time_s=float((ticks[i]+1)*dt),
                    path_error_m=float(path[i]), heading_error_rad=float(heading[i]),
                    roll_rate_rad_s=float(rate[i]),
                    remaining_observed_s=float(after.sum()*dt),
                    later_actor_legal_fraction=float(legal[after].mean()) if after.any() else None)
    first = int(supported[0]) if supported.size else None
    last = int(supported[-1]) if supported.size else None
    # Last active support tick can be the end of observation; that is censored,
    # not an observed release. Also keep all bouts rather than inventing a
    # single contiguous rescue window.
    release_observed = last is not None and last+1 < len(ticks)
    bouts = int(np.sum(gate & ~np.r_[False, gate[:-1]])) if len(ticks) else 0
    return dict(observed_ticks=len(ticks), complete_horizon=len(ticks)==horizon_ticks,
                pre_pulse=snapshot(onset_tick-1),
                support_bouts=bouts, support_gate_s=float(gate.sum()*dt),
                legal_support_s=float((gate & legal).sum()*dt),
                first_support_preceding_pose=snapshot(first-1) if first is not None else None,
                first_support_time_s=float(ticks[first]*dt) if first is not None else None,
                last_support_pose=snapshot(last), release_observed=release_observed,
                post_pulse_actor_legal_fraction=float(legal[post].mean()) if post.any() else None,
                post_pulse_peak_abs_roll_rate_rad_s=float(abs(rate[post]).max()) if post.any() else None,
                post_pulse_max_abs_path_m=float(abs(path[post]).max()) if post.any() else None,
                post_pulse_max_abs_heading_rad=float(abs(heading[post]).max()) if post.any() else None)


def review(root):
    root = Path(root).resolve()
    analysis = root/'analysis'
    status = json.loads((analysis/'status.json').read_text())
    if status['phase'] != 'completed':
        raise ValueError('completed analysis required')
    source = analysis/'results.json'
    rows = json.loads(source.read_text())
    plan = json.loads((root/'plan.json').read_text())
    if plan.get('supported_candidate') != 'free_GAE':
        raise ValueError('declared single supported candidate required')
    selected = [r for r in rows if r['role']=='free_GAE']
    if len(selected)!=30 or len({r['case_id'] for r in selected})!=30:
        raise ValueError('thirty unique declared candidate episodes required')
    groups = {}
    for r in rows:
        key = (r['context'], r['alpha'], r['event_id'])
        if key not in groups:
            groups[key] = len(groups)
    diagnostics = []
    for row in selected:
        file = Path(row['raw_file'])
        if digest(file)!=row['raw_sha256']:
            raise ValueError('source trajectory changed')
        key = (row['context'], row['alpha'], row['event_id'])
        derived = analysis/f'series_{groups[key]:03d}_free_GAE.npz'
        with np.load(file, allow_pickle=False) as z:
            raw = {k:z[k] for k in z.files}
        with np.load(derived, allow_pickle=False) as z:
            series = {k:z[k] for k in ['path_error','heading_error']}
        d = trace_diagnostics(raw, series, onset_tick=row['pulse_onset_task_tick'],
                              dt=plan['lower_dt_s'], horizon_ticks=2000)
        diagnostics.append(dict(case_id=row['case_id'], context=row['context'], alpha=row['alpha'],
            force_N=row['force_N'], arrived=row['goal_arrived'], physical_outcome=row['physical_outcome'],
            minimum_goal_distance_m=row['minimum_upright_goal_distance_m'],
            source_raw_sha256=row['raw_sha256'], source_geometry_sha256=digest(derived), **d))
    out = analysis/'continuation_diagnostics'
    out.mkdir(exist_ok=False)
    atomic_json(out/'cases.json', diagnostics)
    summaries = []
    for arrived in [True, False]:
        for supported in [True, False]:
            group = [d for d in diagnostics if d['force_N'] and d['arrived'] is arrived
                     and (d['support_bouts']>0)==supported]
            summaries.append(dict(arrived=arrived, support_activated=supported, n=len(group),
                incomplete=sum(not d['complete_horizon'] for d in group),
                some_actor_unavailability=sum(d['post_pulse_actor_legal_fraction']<1 for d in group),
                median_post_pulse_actor_legal_fraction=float(np.median([d['post_pulse_actor_legal_fraction'] for d in group])) if group else None,
                median_goal_distance_m=float(np.median([d['minimum_goal_distance_m'] for d in group])) if group else None,
                median_post_pulse_peak_abs_roll_rate_rad_s=float(np.median([d['post_pulse_peak_abs_roll_rate_rad_s'] for d in group])) if group else None))
    result = dict(source_results_sha256=digest(source), cases=len(diagnostics), groups=summaries,
                  physics_steps=0, optimizer_updates=0,
                  interpretation='Descriptive paired timing; no causal handback claim or recoverability certificate. No threshold changed.')
    atomic_json(out/'summary.json', result)
    return result
