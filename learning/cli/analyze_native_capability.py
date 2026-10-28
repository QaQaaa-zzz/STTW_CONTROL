"""Offline, full-panel native H75 S-path capability report (no simulation)."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from analyze_native_frozen_validation import digest, score


def dump_csv(file, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with file.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def bend_landmarks(path):
    """Return signed-curvature lobe peaks, separating the two S bends."""
    k = np.asarray(path['curvature'])
    active = np.flatnonzero(abs(k) > max(1e-6, abs(k).max() * .01))
    if not len(active):
        return np.array([])
    cuts = np.flatnonzero((np.diff(active) > 1) | (np.sign(k[active[1:]]) != np.sign(k[active[:-1]]))) + 1
    return np.array([path['s'][group[np.argmax(abs(k[group]))]] for group in np.split(active, cuts)])


def labels(raw, path, dt, horizon):
    """Contact-free prefixes are censored at any native fatal stop."""
    n = len(raw['time'])
    fatal = np.asarray(raw['physical_failure'], bool) | np.asarray(raw['domain_exit'], bool)
    if fatal.any() and np.flatnonzero(fatal)[0] != n - 1:
        raise ValueError('trace must end at first native fatal stop')
    contact = bool(np.asarray(raw['contact_failure']).any())
    complete = n * dt >= horizon - dt / 2 and not fatal.any()
    stop = n - 1 if fatal.any() else n
    goal = np.array([np.interp(float(path['goal']), path['s'], path['xy'][:, i]) for i in (0, 1)])
    distance = np.linalg.norm(raw['actual_xy'][:stop] - goal, axis=1)
    end = max(1, int(round(1 / dt)))
    enough = complete and n >= end
    final = enough and bool(np.all(abs(raw['path_cross_track'][-end:]) <= .1) and
                            np.all(abs(raw['path_heading_error'][-end:]) <= .05) and
                            np.all(abs(raw['actual_forward_speed'][-end:] - raw['v_user'][-end:]) <= .1))
    bends = bend_landmarks(path)
    second_start = float(path.get('second_start_m', bends[1] if len(bends) >= 2 else np.inf))
    return dict(actual_lower_ticks=n, contact_observation='contact' if contact else 'complete_no_contact' if complete else 'censored_no_contact',
                actual_contact=contact, native_domain_stop=bool(np.asarray(raw['domain_exit']).any()),
                native_physical_stop=bool(np.asarray(raw['physical_failure']).any()), complete_native_horizon=complete,
                minimum_original_goal_distance_before_native_stop_m=float(distance.min()) if len(distance) else None,
                goal1m_before_native_stop=bool(np.any(distance <= 1)),
                original_goal_crossed_before_native_stop=bool(np.any((raw['path_progress'][:stop] >= float(path['goal'])) & (raw['goal_section_signed_distance'][:stop] >= 0))),
                completed_second_bend_before_native_stop=bool(np.any(raw['path_progress'][:stop] >= float(path['turn_end']))),
                reached_second_bend_before_native_stop=bool(np.any(raw['path_progress'][:stop] >= second_start)),
                final_1s_thresholds_met=final, final_1s_observation='complete' if enough else 'censored_or_short',
                observed_duration_s=n * dt)


def original_user_score(raw, path, cfg, controller, keys, execution_alpha):
    """Validate the actual reward without changing the fixed original-user objective."""
    own, own_series = score(raw, path, cfg, controller, keys, execution_alpha)
    logged = -float(np.asarray(raw['scored_tick_reward'], float).sum())
    np.testing.assert_allclose(own['scored_cost'], logged, rtol=1e-5, atol=1e-3,
                               err_msg='executed-alpha runtime reward mismatch')
    np.testing.assert_allclose(own_series['scored'], -np.asarray(raw['scored_tick_reward'], float),
                               rtol=2e-5, atol=2e-5, err_msg='executed-alpha per-tick runtime reward mismatch')
    objective, series = score(raw, path, cfg, controller, keys, 1.)
    return objective, series, own['scored_cost'] - logged


def verified_raw(row, root):
    """Missing or changed archived traces are fatal; never substitute a rollout."""
    file = Path(row['raw_file'])
    if not file.is_absolute():
        file = root / file
    if not file.is_file() or digest(file) != row['raw_sha256']:
        raise ValueError('missing or changed cached/raw trace: ' + str(file))
    with np.load(file, allow_pickle=False) as z:
        return dict(z)


def strategy(row):
    if row['policy'] in ('alpha_schedule', 'curvature_speed', 'speed0.5_steer1', 'speed1_steer0.5'):
        return row['policy']
    return 'zero' if row['policy'].lower() == 'zero' else f"fixed{float(row['alpha']):g}"


def sealed_panel(root, expected):
    plan = json.loads((root / 'plan.json').read_text())
    if digest(root / 'plan.json') != json.loads((root / 'seal.json').read_text())['plan_sha256']:
        raise ValueError('sealed plan changed: ' + str(root))
    status = json.loads((root / 'evaluation/status.json').read_text())
    if status['phase'] != 'completed' or status['completed_cases'] != expected:
        raise ValueError('complete declared panel required; no interim outcome analysis')
    manifest = json.loads((root / 'evaluation/manifest.json').read_text())
    if manifest['plan_sha256'] != digest(root / 'plan.json') or manifest['cases'] != plan['cases']:
        raise ValueError('frozen manifest/plan identity mismatch')
    for key in ('config', 'source', 'native_runtime_root'):
        if manifest[key] != plan['initial_source' if key == 'source' else key]:
            raise ValueError('frozen manifest identity mismatch: ' + key)
    rows = json.loads((root / 'evaluation/results.json').read_text())['cases']
    def inventory(items):
        return sorted((r['route'], r['policy'], float(r['alpha']), float(r['speed'])) +
                      ((float(r['speed_alpha']), float(r['steer_alpha'])) if plan.get('mode') == 'native_channel_factorial_v1' else ())
                      for r in items)
    if len(rows) != expected or inventory(rows) != inventory(plan['cases']):
        raise ValueError('result inventory differs from sealed plan')
    return plan, manifest, rows


def scheduling_acceptance(records, scores, traces, paths, acceptance, contexts):
    """Apply the sealed nominal-DEV decision rule, retaining all failed comparisons."""
    # These keys are protocol, not thresholds fitted to observed outcomes.
    relative = acceptance['gentle_cost_max_ratio_to_fixed1'] - 1.
    absolute = acceptance['gentle_cost_max_additive']
    improvements = dict(total_cost=acceptance['aggregate_total_cost_reduction_fraction'],
                        task_channels=acceptance['aggregate_task_channels_reduction_fraction'])
    tol = acceptance['return_alpha_tolerance']
    angles = sorted({c['angle_deg'] for c in contexts})
    if len(angles) != 2:
        raise ValueError('two declared gentle/tight geometry pairs required')
    gentle = {c['route'] for c in contexts if c['angle_deg'] == angles[0]}
    tight = {c['route'] for c in contexts if c['angle_deg'] == angles[1]}
    if gentle & tight or gentle | tight != set(paths) or len(gentle) != 2 or len(tight) != 2:
        raise ValueError('acceptance must explicitly cover two gentle and two tight routes')
    indexed = {(r['route'], r['strategy']): r for r in records}
    scored = {s['analysis_id']: s for s in scores}
    candidates = ['fixed0', 'fixed0.5', 'fixed1', 'curvature_speed']
    aggregates = {}
    for name in candidates + ['alpha_schedule', 'zero']:
        group = [indexed[route, name] for route in paths]
        aggregates[name] = dict(strict_passing_all_routes=all(r['task_recovered'] for r in group),
                                total_cost=sum(scored[r['analysis_id']]['scored_cost'] for r in group),
                                task_channels=sum(scored[r['analysis_id']]['task_channels'] for r in group))
    eligible = [name for name in candidates if aggregates[name]['strict_passing_all_routes']]
    # Each objective must beat its own strongest strictly passing simple comparator.
    best = {metric: min((aggregates[name][metric] for name in eligible), default=None)
            for metric in ('total_cost', 'task_channels')}
    schedule = [indexed[route, 'alpha_schedule'] for route in paths]
    return_details = {}
    for r in schedule:
        raw = traces[r['analysis_id']]
        after = np.flatnonzero(raw['path_progress'] >= float(paths[r['route']]['turn_end']))
        hits = after[np.abs(raw['executed_alpha'][after] - 1.) <= tol]
        returned = bool(len(hits) and np.all(np.abs(raw['executed_alpha'][hits[0]:] - 1.) <= tol))
        active = np.flatnonzero(np.asarray(raw['rule_active'], bool)) if 'rule_active' in raw else np.array([], dtype=int)
        tick_dt = float(np.diff(raw['time'])[0]) if len(raw['time']) > 1 else 0.
        return_details[r['route']] = dict(returned_and_remained_at_1_after_final_curve=returned,
                                         first_postcurve_return_s=float(raw['time'][hits[0]]) if len(hits) else None,
                                         rule_active_duration_s=len(active)*tick_dt,
                                         last_rule_active_s=float(raw['time'][active[-1]]) if len(active) else None)
    gentle_costs = {route: scored[indexed[route, 'alpha_schedule']['analysis_id']]['scored_cost'] <=
                   scored[indexed[route, 'fixed1']['analysis_id']]['scored_cost'] * (1 + relative) + absolute
                   for route in gentle}
    gates = dict(strict_all_four=all(r['task_recovered'] for r in schedule),
                 gentle_original_user_cost_bounds=all(gentle_costs.values()),
                 no_contact_or_domain_or_physical=all(not (r['actual_contact'] or r['native_domain_stop'] or r['native_physical_stop']) for r in schedule),
                 alpha_returns_1_after_final_curve=all(r['returned_and_remained_at_1_after_final_curve'] for r in return_details.values()),
                 simple_strict_comparator_exists=bool(eligible))
    for metric in best:
        gates['aggregate_' + metric + '_improvement'] = bool(best[metric] is not None and
            aggregates['alpha_schedule'][metric] <= best[metric] * (1 - improvements[metric]))
    differences, hindsight = [], {}
    for route in paths:
        current = scored[indexed[route, 'alpha_schedule']['analysis_id']]
        for name in candidates + ['zero']:
            other = scored[indexed[route, name]['analysis_id']]
            differences.append(dict(route=route, comparator=name,
                                    comparator_strict=indexed[route, name]['task_recovered'],
                                    schedule_strict=indexed[route, 'alpha_schedule']['task_recovered'],
                                    total_cost_difference=current['scored_cost'] - other['scored_cost'],
                                    task_channels_difference=current['task_channels'] - other['task_channels']))
        eligible_fixed = [name for name in candidates[:3] if indexed[route, name]['task_recovered']]
        hindsight[route] = {metric: min((dict(strategy=name, cost=scored[indexed[route, name]['analysis_id']][field])
                                      for name in eligible_fixed), key=lambda x:x['cost'], default=None)
                            for metric, field in (('total_cost', 'scored_cost'), ('task_channels', 'task_channels'))}
    max_degradation = {name:{metric:max(d[metric+'_difference'] for d in differences if d['comparator']==name)
                            for metric in ('total_cost','task_channels')} for name in candidates+['zero']}
    return dict(accepted=all(gates.values()), gates=gates, gentle_cost_bounds=gentle_costs,
                per_route_differences= differences, maximum_observed_cost_increase=max_degradation,
                hindsight_per_route_strict_fixed=hindsight,
                hindsight_warning='Descriptive hindsight oracle on familiar DEV, not an executed online controller.',
                alpha_return=return_details, aggregates=aggregates, strictly_passing_simple_controllers=eligible,
                best_strict_simple_costs=best, protocol=acceptance,
                scope='Exploratory familiar DEV nominal screen; no disturbed recovery, tolerance predictor, replanning or generalization claim.')


def verify_channel_composition(raw, row):
    """Check both queried vectors and exact per-channel composition every tick."""
    n = len(raw['time'])
    pair = (float(row['speed_alpha']), float(row['steer_alpha']))
    if pair not in ((.5, 1.), (1., .5)):
        raise ValueError('undeclared channel alpha pair')
    for field in ('queried_latent_alpha_half', 'queried_latent_alpha_one', 'composed_latent', 'channel_alpha_pair', 'latent_z'):
        a = np.asarray(raw[field])
        if a.shape != (n, 2) or not np.isfinite(a).all():
            raise ValueError('invalid channel log: ' + field)
    np.testing.assert_array_equal(raw['channel_alpha_pair'], np.tile(pair, (n, 1)))
    np.testing.assert_array_equal(raw['original_user_alpha'], np.ones(n))
    queries = {.5: raw['queried_latent_alpha_half'], 1.: raw['queried_latent_alpha_one']}
    expected = np.column_stack([queries[pair[0]][:, 0], queries[pair[1]][:, 1]])
    np.testing.assert_allclose(raw['composed_latent'], expected, rtol=0, atol=1e-6,
                               err_msg='channel composition mismatch')
    np.testing.assert_allclose(raw['latent_z'], expected, rtol=0, atol=1e-6,
                               err_msg='applied latent differs from composition')
    return float(np.max(np.abs(raw['composed_latent'] - expected)))


def channel_interpretation(records, scores, traces, paths, contexts):
    """Frozen descriptive comparisons; no adoption or post-hoc threshold search."""
    index = {(r['route'], r['strategy']): r for r in records}
    costs = {r['analysis_id']: r for r in scores}
    hybrid = ('speed0.5_steer1', 'speed1_steer0.5')
    comparisons = []
    tight_angle = max(c['angle_deg'] for c in contexts)
    tight = {c['route'] for c in contexts if c['angle_deg'] == tight_angle}
    for route in paths:
        for name in hybrid:
            a = index[route, name]; ac = costs[a['analysis_id']]
            for baseline in ('fixed0.5', 'fixed1', 'fixed0', 'zero'):
                b = index[route, baseline]; bc = costs[b['analysis_id']]
                av = np.asarray(traces[a['analysis_id']]['actual_forward_speed'])
                bv = np.asarray(traces[b['analysis_id']]['actual_forward_speed'])
                n = min(len(av), len(bv)); delta = av[:n] - bv[:n]
                comparisons.append(dict(route=route, hybrid=name, comparator=baseline,
                    hybrid_strict=a['channel_strict'], comparator_strict=b['channel_strict'],
                    strict_gain=bool(a['channel_strict'] and not b['channel_strict']),
                    strict_loss=bool(b['channel_strict'] and not a['channel_strict']),
                    total_cost_difference=ac['scored_cost']-bc['scored_cost'],
                    task_channels_difference=ac['task_channels']-bc['task_channels'],
                    component_cost_differences={k: ac['components'][k]-bc['components'][k] for k in ac.get('components', {})},
                    total_cost_within_tolerance=ac['scored_cost'] <= 1.05*bc['scored_cost']+.1,
                    task_cost_within_tolerance=ac['task_channels'] <= 1.05*bc['task_channels']+.1,
                    common_observation_ticks=n, hybrid_observation_ticks=len(av), comparator_observation_ticks=len(bv),
                    mean_signed_speed_difference_m_s=float(delta.mean()),
                    mean_abs_speed_difference_m_s=float(abs(delta).mean()),
                    maximum_abs_speed_difference_m_s=float(abs(delta).max())))
    forward = [x for x in comparisons if x['hybrid']==hybrid[0] and x['comparator']=='fixed0.5']
    reverse = [x for x in comparisons if x['hybrid']==hybrid[1] and x['comparator']=='fixed1']
    aggregates = {}
    for name in ('fixed0', 'fixed0.5', 'fixed1', 'zero') + hybrid:
        group = [index[route, name] for route in paths]
        aggregates[name] = dict(strict_count=sum(r['channel_strict'] for r in group), denominator=len(group),
            total_cost=sum(costs[r['analysis_id']]['scored_cost'] for r in group),
            task_channels=sum(costs[r['analysis_id']]['task_channels'] for r in group))
    return dict(per_route_differences=comparisons, aggregates=aggregates,
        speed_half_steer_one_covers_fixed_half=all(x['hybrid_strict'] and x['total_cost_within_tolerance'] and x['task_cost_within_tolerance'] for x in forward),
        reverse_improves_fixed_one=all(x['hybrid_strict'] for x in reverse if x['route'] in tight) and any(x['strict_gain'] for x in reverse if x['route'] in tight) and not any(x['strict_loss'] for x in reverse),
        cost_exceptions=[x for x in comparisons if not (x['total_cost_within_tolerance'] and x['task_cost_within_tolerance'])],
        tolerance='Engineering descriptive bound per route: cost <= 1.05 * comparator + 0.1; not statistical significance.',
        adoption='not_automatically_adopted',
        interpretation='Total closed-loop channel substitution; actual speeds differ. No speed-matched or independent steering effect claim. Both hybrids failing does not establish synergy: composed actions may be out of training distribution.',
        scope='Nominal familiar DEV; no disturbance, replanning or new training; single shared physical ancestor.')


def analyze_scheduling(root):
    started = time.perf_counter()
    plan, manifest, newrows = sealed_panel(root, 8)
    factorial = plan.get('mode') == 'native_channel_factorial_v1'
    source = Path(plan['source'])
    old, oldmanifest, cached = sealed_panel(source, 16)
    # The source records themselves, not merely whichever raw files still exist,
    # are part of the sealed comparison inventory.
    for name in ('plan.json', 'seal.json', 'evaluation/manifest.json', 'evaluation/results.json', 'analysis/review.json'):
        file = str(source / name)
        if plan['input_sha256'].get(file) != digest(file):
            raise ValueError('cached frozen identity missing/changed: ' + file)
    stored = json.loads((root / 'cached_records.json').read_text())
    reviewed = json.loads((source / 'analysis/review.json').read_text())['records']
    if stored != reviewed or len(stored) != 16:
        raise ValueError('cached source records changed or missing')
    by_raw = {r['raw_file']: r for r in stored}
    if len(by_raw) != 16 or any(any(by_raw.get(r['raw_file'], {}).get(k) != v for k, v in r.items()) for r in cached):
        raise ValueError('cached reviewed/source result identity mismatch')
    cached = stored
    for key in ('config', 'initial_source', 'h75', 'contexts', 'native_runtime_root', 'native_source_sha256'):
        if plan[key] != old[key]:
            raise ValueError('cached/native identity mismatch: ' + key)
    for key in ('model_identity', 'reward_components'):
        if manifest[key] != oldmanifest[key]:
            raise ValueError('cached frozen manifest identity mismatch: ' + key)
    cfg, keys = manifest['config'], manifest['reward_components']
    if plan['protocol']['original_user_alpha'] != 1.:
        raise ValueError('fixed original user1 required')
    # Frozen physical/model/checkpoint/normalizer inputs remain checked. Mutable
    # live runner source is not mistaken for an archived dependency.
    for f, sha in plan['input_sha256'].items():
        if digest(f) != sha:
            raise ValueError('sealed input changed: ' + f)
    if factorial:
        initial = json.loads((root / 'evaluation/initial_identity_receipt.json').read_text())
        if not initial['original_paths_exact'] or not initial['all_initial_carry_leaves_exact'] or initial['physics_steps'] != 0:
            raise ValueError('original full initial carry/path identity not established')
    dt = cfg['timing']['lower_dt_s']; horizon = plan['horizon_s']
    paths = {}
    geometry = {r['route']: r for r in json.loads((source / 'evaluation/geometry_receipt.json').read_text())}
    for context in plan['contexts']:
        route = context['route']
        with np.load(source / 'evaluation' / (route + '_path.npz')) as z:
            paths[route] = dict(z)
        with np.load(root / 'evaluation' / (route + '_path.npz')) as z:
            if set(z.files) != set(paths[route]):
                raise ValueError('original path fields changed')
            for key in z.files:
                np.testing.assert_array_equal(z[key], paths[route][key], err_msg='original path changed')
        paths[route]['second_start_m'] = geometry[route]['second_start_m']
    records, scores, traces, series = [], [], {}, {}
    for origin, rows in ((source, cached), (root, newrows)):
        for row in rows:
            raw = verified_raw(row, origin)
            case = row.get('case_id', row.get('name'))
            if not case or case in traces:
                raise ValueError('missing/duplicate trace identifier')
            n = len(raw['time'])
            if not n or not np.asarray(raw['active_tick']).all() or n > round(horizon / dt):
                raise ValueError('invalid actual observation window')
            np.testing.assert_allclose(raw['time'], np.arange(n) * dt, rtol=0, atol=3e-6)
            for field in ('policy_fault', 'lower_fault', 'nonfinite', 'applied_force_impulse'):
                if np.any(raw[field]):
                    raise ValueError('invalid nominal trace: ' + field)
            dynamic = origin == root
            composition_error = None
            if dynamic and factorial:
                if row['policy'] not in ('speed0.5_steer1', 'speed1_steer0.5') or row['alpha'] != 1. or row['original_user_alpha'] != 1.:
                    raise ValueError('invalid channel factorial case identity')
                composition_error = verify_channel_composition(raw, row)
                execution_alpha = 1.
            elif dynamic:
                if row['policy'] not in ('alpha_schedule', 'curvature_speed'):
                    raise ValueError('unexpected scheduling policy')
                if row['alpha'] != 1. or row['original_user_alpha'] != 1.:
                    raise ValueError('scheduling cases must retain original user1')
                for field in ('rule_active', 'rule_preview_max_curvature', 'rule_accel_estimate', 'rule_alpha_target',
                              'rule_speed_target', 'rule_h75_proposed_speed', 'rule_requested_speed',
                              'rule_realizable_speed', 'rule_speed_cap_binding', 'rule_speed_authority_limited'):
                    if field not in raw or np.asarray(raw[field]).shape != (n,) or not np.isfinite(raw[field]).all():
                        raise ValueError('missing/invalid rule trace: ' + field)
                a = np.asarray(raw['executed_alpha'])
                if a.shape != (n,) or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
                    raise ValueError('invalid executed_alpha log')
                np.testing.assert_array_equal(raw['original_user_alpha'], np.ones(n))
                if row['policy'] == 'curvature_speed':
                    np.testing.assert_array_equal(a, np.ones(n))
                execution_alpha = a
            else:
                execution_alpha = float(row['alpha'])
            result, data, difference = original_user_score(raw, paths[row['route']], cfg, plan['initial_source']['controller'], keys, execution_alpha)
            np.testing.assert_allclose(raw['v_user'], row['speed'], rtol=0, atol=1e-6, err_msg='original requested speed changed')
            record = dict(row, analysis_id=case, strategy=strategy(row), source_kind='new8' if dynamic else 'cached16',
                          **labels(raw, paths[row['route']], dt, horizon), own_reward_reconstruction_difference=difference)
            # Retain the native strict label and independently verify its semantics.
            strict = bool(record['complete_native_horizon'] and record['final_1s_thresholds_met'] and
                          record['original_goal_crossed_before_native_stop'] and not record['actual_contact'])
            if strict != bool(row['task_recovered']):
                raise ValueError('native original-task strict label mismatch: ' + case)
            if factorial:
                record['channel_strict'] = strict
                if dynamic:
                    record['maximum_composition_abs_error'] = composition_error
            records.append(record); traces[case] = raw; series[case, 1.] = data
            scores.append(dict(analysis_id=case, route=row['route'], strategy=record['strategy'], evaluation_alpha=1., **result))
    if len(records) != 24 or len({(r['route'], r['strategy']) for r in records}) != 24:
        raise ValueError('exactly four routes by six strategies required')
    acceptance = (channel_interpretation(records, scores, traces, paths, plan['contexts']) if factorial else
                  scheduling_acceptance(records, scores, traces, paths, plan['protocol']['acceptance'], plan['contexts']))
    out = root / 'analysis'; out.mkdir(exist_ok=False)
    render(out, records, traces, series, paths, keys, cfg, plan, scheduling=True)
    dump_csv(out / 'records.csv', records); dump_csv(out / 'scores.csv', scores)
    dump_csv(out / 'paired_differences.csv', acceptance['per_route_differences'])
    report = dict(records=records, scores=scores, acceptance=acceptance, physics_or_training_executed=0,
                  source_model_identity=manifest['model_identity'], h75_identity=plan['h75'],
                  source_run=str(source), data_role='Exploratory familiar DEV; single shared physical ancestor',
                  analysis_total_s=time.perf_counter()-started)
    (out / 'review.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# Nominal H75 original-user ' + ('channel factorial' if factorial else 'scheduling screen'), '',
             'Original XY trajectories first; all24 traces retain their native original-goal strict labels and true endpoints.', '',
             'Cached fixed0/.5/1 and zero traces are SHA/manifest matched. All reward curves and objective costs use original user alpha=1 and the unchanged original speed/path. Runtime rewards are separately reconstructed at their scoring alpha; hybrid runtime scoring alpha1 is not an executed scalar policy alpha. Own-alpha costs are not the comparison objective.', '',
             'Exploratory familiar DEV, one shared physical ancestor; no disturbance, replanning, tolerance predictor, independent TEST or stability claim. Missing or early-stopped outcomes remain censored/failed; no assumed full windows.', '']
    for route in paths:
        lines += [f'## {route}', '', f'![Original XY]({route}_xy.png)', '', f'![Overview]({route}_overview.png)', '',
                  f'[Diagnostics]({route}_diagnostics.png) · [Reward components]({route}_components.png) · [Rule/alpha traces]({route}_rules.png)', '']
    lines += ['## Sealed acceptance', '', ('Descriptive channel substitution; no automatic adoption.' if factorial else 'Nominal screening acceptance: ' + str(acceptance['accepted'])), '',
              '```json', json.dumps(acceptance, indent=2), '```', '',
              '[All24 native outcomes](records.csv) · [Original user1 costs](scores.csv) · [Full review](review.json)', '',
              ('Channel queries are separate speed_alpha/steer_alpha pairs, never a single executed alpha. Strict channel comparison retains original task_recovered: original goal-section crossing, full H10 and last1s path<=0.1m/yaw<=0.05rad/speed<=0.1m/s with no native failure; original goal1m is separately reported. No new training. ' if factorial else '') + 'PNG, per-trace original XY NPZ/CSV, per-step/cumulative signed reward and components, errors, actual/wheel/request speeds, executed alpha and rule logs accompany this report. Unequal observed windows and terminal replacement remain explicit; low truncated cost is not task success.']
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report


def analyze(root):
    started = time.perf_counter()
    plan = json.loads((root / 'plan.json').read_text())
    if plan.get('mode') in ('native_scheduling_v1', 'native_channel_factorial_v1'):
        return analyze_scheduling(root)
    status = json.loads((root / 'evaluation/status.json').read_text())
    assert status['phase'] == 'completed', 'No interim outcome analysis'
    assert digest(root / 'plan.json') == json.loads((root / 'seal.json').read_text())['plan_sha256']
    manifest = json.loads((root / 'evaluation/manifest.json').read_text())
    rows = json.loads((root / 'evaluation/results.json').read_text())['cases']
    assert len(rows) == 16 and status['completed_cases'] == 16, 'complete sixteen-case panel required'
    assert manifest['cases'] == plan['cases'], 'manifest inventory differs from sealed plan'
    def inventory(items):
        return sorted((r['route'], r['policy'], float(r['alpha']), float(r['speed'])) for r in items)
    assert inventory(rows) == inventory(plan['cases']), 'result inventory differs from sealed plan'
    cfg, keys = manifest['config'], manifest['reward_components']
    source = plan['initial_source']
    # This is the normalized controller identity verified by load_environment
    # against the native frozen spec; the spec has no top-level controller.
    spec_file = Path(source['upper_run']) / 'alpha0/frozen_config_stage52.json'
    assert digest(spec_file) == source['upper_config_file_sha256']['0']
    controller = source['controller']
    dt = cfg['timing']['lower_dt_s']
    horizon = cfg['training_future_phase_C']['episode_s']
    routes = list(dict.fromkeys(r['route'] for r in rows))
    assert len(routes) == 4
    geometry = {r['route']: r for r in json.loads((root / 'evaluation/geometry_receipt.json').read_text())}
    paths = {}
    for route in routes:
        with np.load(root / 'evaluation' / (route + '_path.npz')) as z:
            paths[route] = dict(z)
        paths[route]['second_start_m'] = geometry[route]['second_start_m']
        selected = [r for r in rows if r['route'] == route]
        assert len(selected) == 4
        assert sorted(float(r['alpha']) for r in selected if r['policy'].lower() != 'zero') == [0., .5, 1.]
        assert sum(r['policy'].lower() == 'zero' for r in selected) == 1
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    records, scores, traces, series = [], [], {}, {}
    for index, row in enumerate(rows):
        case = row.get('case_id', row.get('name', str(index)))
        file = Path(row['raw_file'])
        if not file.is_absolute():
            file = root / file
        assert digest(file) == row['raw_sha256'], str(file)
        with np.load(file) as z:
            raw = dict(z)
        n = len(raw['time'])
        assert n and np.asarray(raw['active_tick']).all()
        np.testing.assert_allclose(raw['time'], np.arange(n) * dt, rtol=0, atol=3e-6)
        for field in ('policy_fault', 'lower_fault', 'nonfinite', 'applied_force_impulse'):
            assert not np.any(raw[field]), field
        record = dict(row, analysis_id=case, **labels(raw, paths[row['route']], dt, horizon))
        for alpha in ((0., .5, 1.) if row['policy'].lower() == 'zero' else (float(row['alpha']),)):
            result, data = score(raw, paths[row['route']], cfg, controller, keys, alpha)
            scores.append(dict(analysis_id=case, evaluation_alpha=alpha, **result))
            series[case, alpha] = data
            if alpha == float(row['alpha']):
                logged = -float(np.asarray(raw['scored_tick_reward'], float).sum())
                np.testing.assert_allclose(result['scored_cost'], logged, rtol=1e-5, atol=1e-3, err_msg=case)
                record['own_reward_reconstruction_difference'] = result['scored_cost'] - logged
        traces[case] = raw
        records.append(record)
    loading_s = time.perf_counter() - started
    render(out, records, traces, series, paths, keys, cfg, plan)
    dump_csv(out / 'records.csv', records)
    dump_csv(out / 'scores.csv', scores)
    report = dict(records=records, scores=scores, source_model_identity=manifest.get('model_identity'),
                  h75_identity=plan.get('h75', plan.get('h75_identity')), analysis_loading_scoring_s=loading_s,
                  analysis_total_s=time.perf_counter()-started, physics_or_training_executed=0,
                  scope='16 nominal cases; shared initial source, not independent training seeds; no safety or stability claim',
                  thresholds=dict(final_window_s=1, cross_track_m=.1, heading_rad=.05, speed_m_s=.1))
    (out / 'review.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    lines = ['# Native H75 capability', '', 'Original XY trajectories are shown first. Native strict labels remain in records.csv; final-window thresholds are a separate diagnostic.', '',
             'Contact-free native domain/physical truncations are censored. Goal distances exclude the fatal transition. No independent-seed, safety, or stability claim.', '',
             'Costs use each executed H75 user alpha with the same-alpha rescored zero trace. Different-alpha weighted costs are not physical rankings. Terminal replacement is preserved; component curves show running contributions plus a separate terminal adjustment.', '']
    for route in routes:
        lines += [f'## {route}', '', f'![Original XY]({route}_xy.png)', '', f'![Overview]({route}_overview.png)', '', f'[Detailed diagnostics]({route}_diagnostics.png) · [Reward components]({route}_components.png)', '']
    lines += ['[All 16 outcomes](records.csv) · [Own-alpha and paired baseline costs](scores.csv) · [Machine-readable review](review.json)', '', f'Offline analysis elapsed: {report["analysis_total_s"]:.3f} s. Numerical NPZ and CSV are provided for every plotted trace/evaluation alpha.']
    (out / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return report


def render(out, records, traces, series, paths, keys, cfg, plan, scheduling=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dt = cfg['timing']['lower_dt_s']
    for route, path in paths.items():
        selected = [r for r in records if r['route'] == route]
        fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
        xyfig, xy = plt.subplots(figsize=(9, 8), constrained_layout=True)
        detail, dax = plt.subplots(5, 1, figsize=(12, 13), sharex=True, constrained_layout=True)
        components, cax = plt.subplots(len(keys)+1, 2, figsize=(13, 2.2*(len(keys)+1)), sharex=True, constrained_layout=True, squeeze=False)
        max_progress = max(float(np.max(traces[r['analysis_id']]['path_progress'])) for r in selected)
        display = path['s'] <= max(float(path['goal']) + 2., max_progress + 2.)
        for ax in (xy, axes[0, 0]):
            ax.plot(path['xy'][display, 0], path['xy'][display, 1], 'k--', label='Original reference', linewidth=1)
            for i, s in enumerate(bend_landmarks(path)):
                point = [np.interp(s, path['s'], path['xy'][:, j]) for j in (0, 1)]
                ax.plot(*point, 'ko'); ax.annotate(f'Bend {i+1}', point)
            goal = [np.interp(float(path['goal']), path['s'], path['xy'][:, j]) for j in (0, 1)]
            ax.plot(*goal, 'k*', markersize=12, label='Original final goal')
            ax.set(xlabel='World X (m)', ylabel='World Y (m)'); ax.set_aspect('equal', adjustable='datalim')
        for idx, row in enumerate(selected):
            case = row['analysis_id']; raw = traces[case]; alpha = float(row['alpha'])
            zero = row['policy'].lower() == 'zero'
            role = row.get('strategy', strategy(row))
            color = {'zero':'#555555', 'fixed0':'#0072B2', 'fixed0.5':'#D55E00', 'fixed1':'#009E73',
                     'alpha_schedule':'#CC79A7', 'curvature_speed':'#E69F00',
                     'speed0.5_steer1':'#CC79A7', 'speed1_steer0.5':'#E69F00'}[role]
            label = ('zero (frozen lower + ECBC/ESO)' if zero else f'H75 alpha={alpha:g}') if not scheduling else role
            t = (np.arange(len(raw['time']))+1)*dt
            marker = 'x' if row['actual_contact'] else 'D' if row['native_domain_stop'] else 'v' if row['native_physical_stop'] else 's'
            for ax in (xy, axes[0, 0]):
                ax.plot(*raw['actual_xy'].T, color=color, label=label)
                ax.plot(*raw['actual_xy'][-1], marker=marker, color=color)
            axes[1, 1].plot(t, raw['actual_forward_speed']-raw['v_user'], color=color, label=label)
            for ax, field in zip(dax[:3], ('path_cross_track', 'path_heading_error', 'phi')):
                ax.plot(t, raw[field], color=color, label=label)
            for field, style in [('actual_forward_speed','-'), ('wheel_speed_proxy',':'), ('v_user','--')]:
                dax[3].plot(t, raw[field], color=color, linestyle=style, label=f'{label} {field}')
            dax[4].plot(t, raw['actual_forward_speed']-raw['wheel_speed_proxy'], color=color, label=label)
            for a in ((1.,) if scheduling else ((0., .5, 1.) if zero else (alpha,))):
                data = series[case, a]; reward = -data['scored']; style = '--' if zero else '-'
                scorecolor = color if scheduling else {0.:'#0072B2', .5:'#D55E00', 1.:'#009E73'}[a]
                title = f'{label} evaluated alpha={a:g}'
                reward_x = np.arange(1, len(t)+1) if scheduling else t
                axes[0, 1].plot(reward_x, reward, color=scorecolor, linestyle=style, label=title)
                axes[1, 0].plot(reward_x, np.cumsum(reward), color=scorecolor, linestyle=style)
                contrib = -data['effective']*cfg['path_reward']['scale']*dt
                terminal = reward-contrib.sum(axis=1)
                numeric = dict(time_s=t, actual_xy=raw['actual_xy'], original_goal_distance_m=np.linalg.norm(raw['actual_xy']-goal,axis=1), reward=reward, cumulative_reward=np.cumsum(reward), terminal_adjustment=terminal)
                for field in ('phi','path_cross_track','path_heading_error','path_progress','actual_forward_speed','wheel_speed_proxy','v_user','physical_failure','domain_exit','contact_failure'):
                    numeric[field] = raw[field]
                if scheduling:
                    if plan.get('mode') != 'native_channel_factorial_v1':
                        numeric['executed_alpha'] = raw.get('executed_alpha', np.full(len(t), alpha))
                    for field in ('queried_latent_alpha_half', 'queried_latent_alpha_one', 'composed_latent', 'channel_alpha_pair'):
                        if field in raw:
                            numeric[field] = raw[field]
                    numeric['original_user_alpha'] = np.ones(len(t))
                    numeric['actual_lower_step'] = np.arange(1, len(t)+1)
                    numeric['runtime_scoring_alpha'] = np.full(len(t), 1. if 'channel_alpha_pair' in raw else alpha)
                    numeric['runtime_own_alpha_reward'] = raw['scored_tick_reward']
                    numeric['runtime_own_alpha_cumulative_reward'] = np.cumsum(raw['scored_tick_reward'])
                    numeric['cumulative_terminal_adjustment'] = np.cumsum(terminal)
                    for field in raw:
                        if field.startswith('rule_') or field in ('reward_chi', 'own_cost_weights'):
                            numeric[field] = raw[field]
                    for field in ('governed', 'latent_z', 'offsets', 'filtered_offset', 'target_offset', 'lower_action'):
                        if field in raw:
                            numeric[field] = raw[field]
                for k, key in enumerate(keys):
                    numeric['reward_'+key] = contrib[:, k]
                    numeric['cumulative_reward_'+key] = np.cumsum(contrib[:, k])
                for k, values in enumerate(list(contrib.T)+[terminal]):
                    cax[k, 0].plot(reward_x, values, color=scorecolor, linestyle=style, label=title)
                    cax[k, 1].plot(reward_x, np.cumsum(values), color=scorecolor, linestyle=style)
                stem = f'{route}_{idx}_scorealpha{a:g}'
                np.savez_compressed(out / (stem+'.npz'), **numeric)
                columns = {k:v for k,v in numeric.items() if np.asarray(v).ndim==1}
                columns.update(actual_x_m=raw['actual_xy'][:,0], actual_y_m=raw['actual_xy'][:,1])
                if scheduling:
                    for field, values in numeric.items():
                        array = np.asarray(values)
                        if array.ndim == 2:
                            columns.update({field+'_'+str(i): array[:,i] for i in range(array.shape[1])})
                dump_csv(out / (stem+'.csv'), [dict(zip(columns, vals)) for vals in zip(*columns.values())])
        if scheduling:
            rules, rax = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
            for row in selected:
                raw = traces[row['analysis_id']]
                if 'channel_alpha_pair' in raw:
                    t = (np.arange(len(raw['time']))+1)*dt
                    for i, channel in enumerate(('speed', 'steer')):
                        rax[0].plot(t, raw['channel_alpha_pair'][:,i], label=row['policy']+' '+channel+' query alpha')
                        rax[1].plot(t, raw['composed_latent'][:,i], label=row['policy']+' composed '+channel)
                    rax[2].plot(t, raw['governed'][:,0], label=row['policy']+' governed speed')
                if row['policy'] not in ('alpha_schedule', 'curvature_speed'):
                    continue
                t = (np.arange(len(raw['time']))+1)*dt
                rax[0].plot(t, raw['executed_alpha'], label=row['policy']+' executed alpha')
                for field in raw:
                    if field.startswith('rule_') and np.asarray(raw[field]).shape == (len(t),):
                        rax[1].plot(t, raw[field], label=row['policy']+' '+field)
                rax[2].plot(t, raw['governed'][:,0], label=row['policy']+' governed speed')
                rax[2].plot(t, raw['v_user'], '--', label=row['policy']+' original request')
            for ax, ylabel in zip(rax, ('Query alpha / scheduled alpha (unitless)', 'Declared rule logs (see labels)', 'Speed (m/s)')):
                ax.set_ylabel(ylabel); ax.grid(alpha=.2); ax.legend(fontsize=6)
            rax[0].axhline(1., color='black', linestyle=':', label='original user1')
            rax[-1].set_xlabel('Elapsed s')
            rules.suptitle(route+'; nominal query-channel/rule logs; original user1; no disturbance')
            rules.savefig(out/f'{route}_rules.png',dpi=135); plt.close(rules)
        for ax in (xy, axes[0, 0]): ax.legend(fontsize=7)
        reward_xlabel = 'Actual lower step' if scheduling else 'Elapsed s'
        axes[0,1].set(ylabel='Signed reward / lower step', xlabel=reward_xlabel); axes[0,1].legend(fontsize=6)
        axes[1,0].set(ylabel='Cumulative signed reward',xlabel=reward_xlabel)
        axes[1,1].set(ylabel='Actual - requested speed (m/s)',xlabel='Elapsed s')
        for ax, name in zip(dax, ['Cross-track (m)','Heading error (rad)','Roll phi (rad)','Speed (m/s): actual / dotted wheel / dashed request','Actual - wheel proxy (m/s)']): ax.set_ylabel(name); ax.grid(alpha=.2)
        dax[0].legend(fontsize=7); dax[-1].set_xlabel('Elapsed s')
        for k, key in enumerate(list(keys)+['terminal adjustment']):
            cax[k,0].set_ylabel(key); cax[k,1].set_ylabel('Cumulative '+key)
        cax[0,0].legend(fontsize=6)
        for ax in cax[-1]: ax.set_xlabel(reward_xlabel)
        identity = f'{route}; {selected[0].get("speed", "declared")} m/s; nominal; seed/shared initial source as sealed\nH75 checkpoint {plan['h75']['checkpoint_sha256'][:16]}; initial bank index {plan['initial_source']['bank_index']}; x=contact D=domain v=physical s=horizon; no external disturbance'
        for f, name in [(xyfig,'xy'),(fig,'overview'),(detail,'diagnostics'),(components,'components')]:
            f.suptitle(identity, fontsize=9); f.savefig(out/f'{route}_{name}.png',dpi=135); plt.close(f)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root.resolve())
