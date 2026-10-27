"""Analyze all52 frozen native cases only after the sealed execution completes.

No simulator, policy or optimizer is imported. Crossscores change evaluation
weights on an already executed trace, never its trajectory or original goal.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def ratio(speed, controller):
    c = controller
    v = np.maximum(c['minimum_speed'], speed)
    m2 = -(c['mass'] * v * v * c['cg_height'] - c['mass'] * c['cg_forward'] * c['trail'] * c['gravity']) * np.cos(c['caster']) / c['wheelbase']
    m2 = np.where(np.abs(m2) < .1, np.where(m2 >= 0, .1, -.1), m2)
    return -c['mass'] * c['gravity'] * c['cg_height'] / m2


def score(raw, path, cfg, controller, keys, alpha):
    c = cfg['path_reward']
    assert c['reward_version'] == 'path_persistent_preference_v3'
    curvature = np.interp(raw['path_progress'], path['s'], path['curvature'])
    ff = np.arctan(controller['wheelbase'] * curvature / np.cos(controller['caster']))
    chi = np.clip((np.abs(ff / ratio(raw['v_user'], controller)) - .18) / .12, 0, 1)
    ev = np.asarray(raw['actual_forward_speed'], float) - raw['v_user']
    ey = np.asarray(raw['path_cross_track'], float)
    eh = np.asarray(raw['path_heading_error'], float)
    huber = lambda x: np.where(np.abs(x) <= 1, x * x, 2 * np.abs(x) - 1)
    cap = c['component_caps']
    eff = np.asarray(raw['effective_costs'], float).copy()
    speed0 = (10 - 9 * chi) * huber(ev / c['speed_scale_m_s'])
    speed1 = 10 * huber(ev / c['speed_scale_m_s'])
    path_cost = 10 * (huber(ey / c['cross_track_scale_m']) + c['path_heading_factor'] * huber(eh / c['path_heading_scale_rad']))
    primary0 = c['primary_weight'] * chi * huber(np.maximum(np.abs(ey) - c['primary_path_band_m'], 0) / c['primary_path_scale_m'])
    primary1 = c['primary_weight'] * chi * huber(np.maximum(np.abs(ev) - c['primary_speed_band_m_s'], 0) / c['primary_speed_scale_m_s'])
    eff[:, keys.index('speed')] = (1 - alpha) * np.minimum(speed0, cap['speed']) + alpha * np.minimum(speed1, cap['speed'])
    eff[:, keys.index('path')] = (1 - (1 - c['path_secondary_ratio']) * alpha) * np.minimum(path_cost, cap['path'])
    eff[:, keys.index('primary')] = (1 - alpha) * np.minimum(primary0, cap['primary']) + alpha * np.minimum(primary1, cap['primary'])
    dt = cfg['timing']['lower_dt_s']
    running = eff.sum(axis=1) * c['scale'] * dt
    scored = running.copy()
    native_stop = raw['physical_failure'] | raw['domain_exit']
    if native_stop.any():
        end = int(np.flatnonzero(native_stop)[0])
        assert end == len(scored) - 1
        remaining = round(cfg['training_future_phase_C']['episode_s'] / cfg['timing']['upper_dt_s']) - end // 4
        gamma = cfg['training_future_phase_C']['gamma']
        terminal = c['failure_extra'] + c['scale'] * cfg['timing']['upper_dt_s'] * c['failure_cost_rate_bound'] * (1 - gamma ** remaining) / (1 - gamma)
        scored[(end // 4) * 4:] = 0.
        scored[end] = terminal
    totals = eff.sum(axis=0) * c['scale'] * dt
    return dict(scored_cost=float(scored.sum()), running_cost=float(running.sum()),
                terminal_replacement_adjustment=float(scored.sum() - running.sum()),
                task_channels=float(sum(totals[keys.index(k)] for k in ('speed', 'path', 'primary'))),
                components=dict(zip(keys, totals.tolist()))), dict(scored=scored, running=running, effective=eff)


def render(out, cases, traces, scored_traces, paths, goals, keys, cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    roles = ['H75_50hz', 'H75_5hz', 'reference_grid_5hz', 'equilibrium_50hz', 'zero_50hz']
    colors = ['#0072B2', '#56B4E9', '#D55E00', '#009E73', '#555555']
    styles = ['-', '--', '-.', ':', '-']
    routes = list(paths)
    (out / 'diagnostics').mkdir()
    for alpha in (0., .5, 1.):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
        for ax, route in zip(axes.flat, routes):
            path = paths[route]
            largest_progress = max(float(traces[r['case_id']]['path_progress'].max()) for r in cases if r['route'] == route)
            support = path['s'] <= max(float(path['goal']) + 2., largest_progress + 2.)
            ax.plot(path['xy'][support, 0], path['xy'][support, 1], 'k--', linewidth=.8, label='Original reference')
            goal = goals[route]
            ax.add_patch(plt.Circle(goal, 1., fill=False, color='black', linewidth=.7))
            ax.plot(*goal, 'k*', markersize=7)
            for role, color, style in zip(roles, colors, styles):
                row = next(r for r in cases if r['route'] == route and r['role'] == role and (role == 'zero_50hz' or r['state_alpha'] == alpha))
                raw = traces[row['case_id']]
                ax.plot(raw['actual_xy'][:, 0], raw['actual_xy'][:, 1], color=color, linestyle=style, label=role)
                marker = 'x' if row['actual_contact'] else 'D' if row['native_domain_stop'] else 'v' if row['native_physical_stop'] else 's'
                ax.plot(*raw['actual_xy'][-1], marker=marker, color=color, markersize=4)
            ax.set(title=route, xlabel='World X (m)', ylabel='World Y (m)')
            ax.set_aspect('equal', adjustable='datalim')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncols=3)
        fig.suptitle(f'Frozen native validation, user alpha={alpha:g}, requested 2.6 m/s\nActual to native stop/10 s; square=end, diamond=domain, x=contact, triangle=other physical stop\nOriginal reference through largest observed progress + 2 m; zero reused')
        fig.savefig(out / f'xy_alpha{alpha:g}.png', dpi=160)
        plt.close(fig)
        for route in routes:
            selected = [next(r for r in cases if r['route'] == route and r['role'] == role and (role == 'zero_50hz' or r['state_alpha'] == alpha)) for role in roles]
            fig, axes = plt.subplots(8, 1, figsize=(11, 15), sharex=True, constrained_layout=True)
            components, grid = plt.subplots(len(keys), 2, figsize=(12, 19), sharex=True, constrained_layout=True)
            for row, color, style in zip(selected, colors, styles):
                raw = traces[row['case_id']]
                scores = scored_traces[row['case_id'], alpha]
                t = np.arange(len(raw['time'])) * .005 + .005
                common = dict(color=color, linestyle=style, linewidth=.9)
                axes[0].plot(t, scores['scored'], label=row['role'], **common)
                axes[1].plot(t, np.cumsum(scores['scored']), **common)
                axes[2].plot(t, raw['actual_forward_speed'], **common)
                axes[2].plot(t, raw['wheel_speed_proxy'], color=color, linestyle=':', linewidth=.6, alpha=.7)
                axes[2].plot(t, raw['governed'][:, 0], color=color, linestyle='--', linewidth=.6, alpha=.7)
                axes[3].plot(t, raw['path_cross_track'], **common)
                axes[4].plot(t, raw['path_heading_error'], **common)
                axes[5].plot(t, raw['phi'], **common)
                axes[6].plot(t, raw['actual_delta'], **common)
                axes[6].plot(t, raw['governed'][:, 1], color=color, linestyle='--', linewidth=.6, alpha=.7)
                axes[7].plot(t, raw['latent_z'][:, 0], **common)
                axes[7].plot(t, raw['latent_z'][:, 1], color=color, linestyle=':', linewidth=.6)
                for k in range(len(keys)):
                    contribution = scores['effective'][:, k] * cfg['path_reward']['scale'] * .005
                    grid[k, 0].plot(t, contribution, label=row['role'], **common)
                    grid[k, 1].plot(t, np.cumsum(contribution), **common)
            axes[2].axhline(2.6, color='black', linestyle='-.', linewidth=.7)
            for ax, label in zip(axes, ['Scored cost / lower tick', 'Cumulative scored cost', 'Speed (m/s)', 'Original path error (m)', 'Heading error (rad)', 'Roll (rad)', 'Steer (rad)', 'Latent target (unitless)']):
                ax.set_ylabel(label)
                ax.grid(alpha=.15)
            axes[0].legend(ncols=3, fontsize=8)
            axes[-1].set_xlabel('Posttick task elapsed (s)')
            fig.suptitle(f'{route}, user alpha={alpha:g}: all matching roles, zero reused\nSpeed: actual / dotted wheel proxy / dashed governed / black request; steer: actual / dashed governed\nLatent: speed channel / dotted steer channel; scored cost includes original terminal replacement')
            fig.savefig(out / 'diagnostics' / f'{route}_a{alpha:g}_series.png', dpi=130)
            plt.close(fig)
            for k, key in enumerate(keys):
                grid[k, 0].set_ylabel(key)
                grid[k, 1].set_ylabel('Cumulative ' + key)
            grid[-1, 0].set_xlabel('Posttick task elapsed (s)')
            grid[-1, 1].set_xlabel('Posttick task elapsed (s)')
            grid[0, 0].set_title('Per lower tick: scale x 0.005 x rate')
            grid[0, 1].set_title('Cumulative running component cost')
            handles, labels = grid[0, 0].get_legend_handles_labels()
            components.legend(handles, labels, loc='outside lower center', ncols=3, fontsize=8)
            components.suptitle(f'{route}, user alpha={alpha:g}: all original running components\nTerminal replacement is separate from component integrals; role colors/styles as in matching XY/series')
            components.savefig(out / 'diagnostics' / f'{route}_a{alpha:g}_components.png', dpi=120)
            plt.close(components)


def main(root):
    status = json.loads((root / 'status.json').read_text())
    assert status['phase'] == 'completed' and status['completed_cases'] == 52, 'No interim outcome analysis'
    plan = json.loads((root / 'plan.json').read_text())
    assert digest(root / 'plan.json') == json.loads((root / 'seal.json').read_text())['plan_sha256']
    for file, sha in {**plan['input_sha256'], **plan['native_source_sha256']}.items():
        assert digest(file) == sha, file
    manifest = json.loads((root / 'evaluation/manifest.json').read_text())
    cfg, keys = plan['config'], manifest['reward_components']
    paths, goals = {}, {}
    for context in plan['contexts']:
        route = context['route']
        with np.load(root / 'evaluation' / (route + '_path.npz')) as z:
            paths[route] = {k: z[k] for k in z.files}
        p = paths[route]
        goals[route] = np.array([np.interp(float(p['goal']), p['s'], p['xy'][:, i]) for i in (0, 1)])
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    cases, scores, traces, scored_traces = [], [], {}, {}
    for item in json.loads((root / 'evaluation/results.json').read_text()):
        assert digest(item['raw_file']) == item['raw_sha256']
        with np.load(item['raw_file']) as z:
            raw = {k: z[k] for k in z.files}
        n = len(raw['time'])
        assert 0 < n <= 2000 and raw['active_tick'].all()
        np.testing.assert_allclose(raw['time'], np.arange(n) * .005, rtol=0, atol=3e-6)
        assert not np.any(raw['policy_fault'] | raw['lower_fault'] | raw['nonfinite'])
        assert not np.any(raw['applied_force_impulse'])
        fatal = raw['physical_failure'] | raw['domain_exit']
        stop = int(np.flatnonzero(fatal)[0]) if fatal.any() else n
        assert (fatal.any() and stop == n - 1) or n == 2000
        dist = np.linalg.norm(np.asarray(raw['actual_xy'][:stop], float) - goals[item['route']], axis=1)
        hits = np.flatnonzero(dist <= 1.)
        contact = bool(raw['contact_failure'].any())
        row = dict(item, actual_lower_ticks=n, native_physical_stop=bool(raw['physical_failure'].any()),
                   actual_contact=contact, native_domain_stop=bool(raw['domain_exit'].any()),
                   complete_native_horizon=n == 2000 and not fatal.any(),
                   contact_observation='contact' if contact else 'complete_no_contact' if n == 2000 and not fatal.any() else 'truncated_no_contact',
                   goal1m_before_native_stop=bool(len(hits)),
                   first_arrival_posttick_s=(int(hits[0]) + 1) * .005 if len(hits) else None,
                   minimum_goal_distance_m=float(dist.min()) if len(dist) else None,
                   full_path_rmse_m=float(np.sqrt(np.mean(raw['path_cross_track'] ** 2))),
                   full_speed_rmse_m_s=float(np.sqrt(np.mean((raw['actual_forward_speed'] - raw['v_user']) ** 2))))
        own_error = None
        for alpha in plan['user_alphas']:
            result, timeseries = score(raw, paths[item['route']], cfg, plan['initial_source']['controller'], keys, alpha)
            scores.append(dict(case_id=item['case_id'], role=item['role'], route=item['route'],
                               execution_alpha=item['execution_alpha'], evaluation_alpha=alpha, **result))
            scored_traces[item['case_id'], alpha] = timeseries
            if alpha == item['state_alpha']:
                logged = float(-np.asarray(raw['scored_tick_reward'], float).sum())
                own_error = result['scored_cost'] - logged
                assert np.isclose(result['scored_cost'], logged, rtol=1e-5, atol=1e-3), 'Original reward mismatch'
        row['own_scored_cost_reconstruction_difference'] = own_error
        cases.append(row)
        traces[item['case_id']] = raw
    assert len(cases) == 52 and len(scores) == 156
    assert sum(r['actual_lower_ticks'] for r in cases) == status['actual_lower_ticks']
    lookup = {(r['role'], r['route'], r['execution_alpha'], r['evaluation_alpha']): r for r in scores}
    comparisons = []
    for route in paths:
        for alpha in plan['user_alphas']:
            for control, execution, candidate in [('zero_50hz', None, 'H75_50hz'),
                    ('equilibrium_50hz', alpha, 'H75_50hz'), ('reference_grid_5hz', alpha, 'H75_5hz'),
                    ('H75_5hz', alpha, 'H75_50hz')] + [('H75_50hz', a, 'H75_50hz') for a in plan['user_alphas']]:
                a = lookup[candidate, route, alpha, alpha]
                b = lookup[control, route, execution, alpha]
                comparisons.append(dict(candidate=candidate, control=control, route=route, user_alpha=alpha,
                    control_execution_alpha=execution, cost_delta=a['scored_cost'] - b['scored_cost'],
                    task_channels_delta=a['task_channels'] - b['task_channels']))
    predictions = list((root / 'evaluation/forecasts').glob('block*.npz'))
    predicted_ticks = sum(int(np.load(p)['observed_lower_ticks'].sum()) for p in predictions)
    assert len(predictions) == 150 and predicted_ticks == status['prediction_lower_ticks']
    review = dict(records=cases, scores=scores, comparisons=comparisons,
                  actual_lower_ticks=status['actual_lower_ticks'], allocated_actual_slots=status['allocated_actual_slots'],
                  prediction_lower_ticks=predicted_ticks, allocated_prediction_slots=status['allocated_prediction_slots'],
                  new_training_steps=0, raw_physical_trajectories=52, crossscore_rows=156,
                  own_reward_max_abs_reconstruction_difference=max(abs(r['own_scored_cost_reconstruction_difference']) for r in cases),
                  scope=plan['independence'], analysis_source_sha256=digest(__file__))
    (out / 'review.json').write_text(json.dumps(review, indent=2, allow_nan=False) + '\n')
    write_csv(out / 'records.csv', cases)
    write_csv(out / 'scores.csv', scores)
    write_csv(out / 'comparisons.csv', comparisons)
    render(out, cases, traces, scored_traces, paths, goals, keys, cfg)
    print(json.dumps(dict(actual_lower_ticks=review['actual_lower_ticks'], prediction_lower_ticks=predicted_ticks,
                          cases=52, crossscore_rows=156, output=str(out)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
