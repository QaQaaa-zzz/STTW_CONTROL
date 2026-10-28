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


def analyze(root):
    started = time.perf_counter()
    plan = json.loads((root / 'plan.json').read_text())
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


def render(out, records, traces, series, paths, keys, cfg, plan):
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
            color = '#555555' if zero else {0.:'#0072B2', .5:'#D55E00', 1.:'#009E73'}[alpha]
            label = 'zero (frozen lower + ECBC/ESO)' if zero else f'H75 alpha={alpha:g}'
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
            for a in ((0., .5, 1.) if zero else (alpha,)):
                data = series[case, a]; reward = -data['scored']; style = '--' if zero else '-'
                scorecolor = {0.:'#0072B2', .5:'#D55E00', 1.:'#009E73'}[a]
                title = f'{label} evaluated alpha={a:g}'
                axes[0, 1].plot(t, reward, color=scorecolor, linestyle=style, label=title)
                axes[1, 0].plot(t, np.cumsum(reward), color=scorecolor, linestyle=style)
                contrib = -data['effective']*cfg['path_reward']['scale']*dt
                terminal = reward-contrib.sum(axis=1)
                numeric = dict(time_s=t, actual_xy=raw['actual_xy'], original_goal_distance_m=np.linalg.norm(raw['actual_xy']-goal,axis=1), reward=reward, cumulative_reward=np.cumsum(reward), terminal_adjustment=terminal)
                for field in ('phi','path_cross_track','path_heading_error','path_progress','actual_forward_speed','wheel_speed_proxy','v_user','physical_failure','domain_exit','contact_failure'):
                    numeric[field] = raw[field]
                for k, key in enumerate(keys):
                    numeric['reward_'+key] = contrib[:, k]
                    numeric['cumulative_reward_'+key] = np.cumsum(contrib[:, k])
                for k, values in enumerate(list(contrib.T)+[terminal]):
                    cax[k, 0].plot(t, values, color=scorecolor, linestyle=style, label=title)
                    cax[k, 1].plot(t, np.cumsum(values), color=scorecolor, linestyle=style)
                stem = f'{route}_{idx}_scorealpha{a:g}'
                np.savez_compressed(out / (stem+'.npz'), **numeric)
                columns = {k:v for k,v in numeric.items() if np.asarray(v).ndim==1}
                columns.update(actual_x_m=raw['actual_xy'][:,0], actual_y_m=raw['actual_xy'][:,1])
                dump_csv(out / (stem+'.csv'), [dict(zip(columns, vals)) for vals in zip(*columns.values())])
        for ax in (xy, axes[0, 0]): ax.legend(fontsize=7)
        axes[0,1].set(ylabel='Signed reward / lower step', xlabel='Elapsed s'); axes[0,1].legend(fontsize=6)
        axes[1,0].set(ylabel='Cumulative signed reward',xlabel='Elapsed s')
        axes[1,1].set(ylabel='Actual - requested speed (m/s)',xlabel='Elapsed s')
        for ax, name in zip(dax, ['Cross-track (m)','Heading error (rad)','Roll phi (rad)','Speed (m/s): actual / dotted wheel / dashed request','Actual - wheel proxy (m/s)']): ax.set_ylabel(name); ax.grid(alpha=.2)
        dax[0].legend(fontsize=7); dax[-1].set_xlabel('Elapsed s')
        for k, key in enumerate(list(keys)+['terminal adjustment']):
            cax[k,0].set_ylabel(key); cax[k,1].set_ylabel('Cumulative '+key)
        cax[0,0].legend(fontsize=6)
        identity = f'{route}; {selected[0].get("speed", "declared")} m/s; nominal; seed/shared initial source as sealed\nH75 checkpoint {plan['h75']['checkpoint_sha256'][:16]}; initial bank index {plan['initial_source']['bank_index']}; x=contact D=domain v=physical s=horizon; no external disturbance'
        for f, name in [(xyfig,'xy'),(fig,'overview'),(detail,'diagnostics'),(components,'components')]:
            f.suptitle(identity, fontsize=9); f.savefig(out/f'{route}_{name}.png',dpi=135); plt.close(f)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root.resolve())
