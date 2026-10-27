"""Read actual cached midpoint rollouts; never interpolate endpoint dynamics."""
import argparse
import json
from pathlib import Path

import jax.numpy as j
import numpy as np

from sttw_control.controller import ControllerConfig
from sttw_control.direct_command_policy import q_ratio
from sttw_control.geometric_path import build_path
from sttw_control.path_command_reward import configured_cost_keys
from sttw_control.preference_best import digest
from sttw_control.preference_trace_rescore import rescore_trace_alpha


def main(root):
    plan = json.loads((root / 'analysis_plan.json').read_text())
    source = Path(plan['source'])
    for name in ('manifest', 'results'):
        assert digest(source / (name + '.json')) == plan['source_' + name + '_sha256']
    manifest = json.loads((source / 'manifest.json').read_text())
    declaration = manifest['declaration']
    cfg = manifest['config']
    registry = json.loads(Path('learning/configs/frozen_middle_registry.json').read_text())
    model = registry['records']['STTW_MIDDLE_FULLTAIL_PPO75_U77']
    for key in ('actor', 'checkpoint'):
        assert digest(model[key]) == declaration['policies']['ppo75'][key + '_sha256'] == model[key + '_sha256']
    selected = [r for r in json.loads((source / 'results.json').read_text())['cases']
                if r['policy'] == 'ppo75' and r['force_N'] == 0 and r['onset_s'] == 1
                and r['alpha'] in plan['execution_alphas']]
    assert len(selected) == 12
    cc = ControllerConfig(**manifest['source']['controller'])
    keys = list(configured_cost_keys(cfg))
    assert keys == manifest['reward_components']
    task_indices = [keys.index(k) for k in ('speed', 'path', 'primary')]
    builders = {r['name']: r['builder'] for r in declaration['routes']}
    records, traces, geometry, identities = [], {}, {}, {}
    out = root / 'analysis'
    out.mkdir(exist_ok=False)
    for row in selected:
        file = source / (row['name'] + '.npz')
        identities[str(file)] = digest(file)
        with np.load(file) as z:
            raw = {k: z[k] for k in z.files}
        active = raw['active_tick'].astype(bool)
        assert active.all() and 0 < len(active) <= 2000
        np.testing.assert_allclose(raw['time'], np.arange(len(active)) * .005, atol=2e-6, rtol=0)
        assert not np.any(raw['applied_force_impulse'])
        assert not np.any(raw['policy_fault'] | raw['lower_fault'])
        np.testing.assert_allclose(raw['v_user'], row['speed'], atol=1e-6, rtol=0)
        path = build_path(builders[row['route']], np.zeros(3), cfg)
        # Original world placement comes from the saved path, never the
        # translation-invariant zero-origin curve used to reconstruct chi.
        geometry_file = source / (row['route'] + '_path.npz')
        identities[str(geometry_file)] = digest(geometry_file)
        with np.load(geometry_file) as z:
            geometry[row['route']] = {k: z[k] for k in z.files}
        curvature = np.interp(raw['path_progress'], np.asarray(path.s), np.asarray(path.curvature)).astype(np.float32)
        ff = np.arctan(np.float32(cc.wheelbase) * curvature / np.cos(np.float32(cc.caster)))
        chi = np.clip((abs(ff / np.asarray(q_ratio(j.float32(row['speed']), cc))) - .18) / .12, 0, 1)
        terminal = raw['physical_failure'].astype(bool) | raw['domain_exit'].astype(bool)
        data = dict(raw, chi=chi, requested_speed=raw['v_user'], contact_failure=terminal)
        own = rescore_trace_alpha(data, alpha=row['alpha'], config=cfg)
        np.testing.assert_allclose(own['scored_reward'], raw['scored_tick_reward'], rtol=3e-5, atol=3e-6)
        stop = int(np.flatnonzero(terminal)[0]) if terminal.any() else len(active)
        world_path = geometry[row['route']]
        goal = np.array([np.interp(float(world_path['goal']), world_path['s'], world_path['xy'][:, k]) for k in (0, 1)])
        hits = np.flatnonzero(np.linalg.norm(raw['actual_xy'][:stop] - goal, axis=1) <= 1.)
        scores = {}
        for user in plan['user_alphas']:
            score = rescore_trace_alpha(data, alpha=user, config=cfg)
            components = cfg['path_reward']['scale'] * .005 * score['effective_costs'].sum(axis=0)
            scores[str(user)] = dict(scored=score['scored_cost'], running=score['running_cost'],
                                    task_channels=float(components[task_indices].sum()),
                                    components={k: float(v) for k, v in zip(keys, components)})
        records.append(dict(name=row['name'], route=row['route'], speed=row['speed'], execution_alpha=row['alpha'],
                            raw_file=str(file), raw_sha256=digest(file), scores=scores,
                            complete=len(active) == 2000, physical_failure=bool(raw['physical_failure'].any()),
                            domain_exit=bool(raw['domain_exit'].any()), goal1m_before_stop=bool(len(hits)),
                            first_arrival_s=float((hits[0] + 1) * .005) if len(hits) else None,
                            own_reward_max_error=float(np.max(abs(own['scored_reward'] - raw['scored_tick_reward'])))))
        traces[row['name']] = raw
    contexts = sorted({(r['route'], r['speed']) for r in records})
    comparisons = []
    for route, speed in contexts:
        group = {r['execution_alpha']: r for r in records if (r['route'], r['speed']) == (route, speed)}
        assert set(group) == {0., .5, 1.}
        for user in plan['user_alphas']:
            a, b = group[user], group[.5]
            comparisons.append(dict(route=route, speed=speed, user_alpha=user,
                matched_minus_midpoint={k: a['scores'][str(user)][k] - b['scores'][str(user)][k]
                                        for k in ('scored', 'running', 'task_channels')}))
    result = dict(records=records, comparisons=comparisons, source_sha256=identities,
                  plan_sha256=digest(root / 'analysis_plan.json'), cli_sha256=digest(__file__),
                  new_physics_ticks=0, new_training_steps=0, historical_unique_traces=12,
                  scope=plan['inference_limit'])
    (out / 'review.json').write_text(json.dumps(result, indent=2) + '\n')
    render(out, records, traces, contexts, geometry)
    print(json.dumps(comparisons, indent=2))


def render(out, records, traces, contexts, geometry):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Same source context, matched zero controls; draw both alpha executions.
    baseline = json.loads(Path('runs/native_equilibrium_rule_v1/analysis/review.json').read_text())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout='constrained')
    for ax, (route, speed) in zip(axes.flat, contexts):
        path = geometry[route]
        # Full real trajectories remain visible; unused path support is cropped
        # to their observed envelope rather than compressing the task to 80m.
        ax.plot(path['xy'][:, 0], path['xy'][:, 1], ':', color='.4', label='Original geometric path')
        envelope = []
        for alpha, color, style in [(0., '#0072B2', '-'), (.5, '#009E73', '-.'), (1., '#D55E00', '--')]:
            row = next(r for r in records if (r['route'], r['speed'], r['execution_alpha']) == (route, speed, alpha))
            raw = traces[row['name']]
            envelope.append(raw['actual_xy'])
            ax.plot(*raw['actual_xy'].T, color=color, ls=style, label=f'H75 execution alpha={alpha:g}')
            ax.scatter(*raw['actual_xy'][-1], color=color, s=20, marker='x' if row['physical_failure'] else 'D' if row['domain_exit'] else 's')
        for r in baseline['records']:
            if (r['family'], r['route'], r['speed']) != ('zero', route, speed):
                continue
            assert digest(r['raw_file']) == r['raw_sha256']
            with np.load(r['raw_file']) as z:
                envelope.append(z['actual_xy'])
                ax.plot(*z['actual_xy'].T, color='.6', lw=.8, ls='-' if r['alpha'] == 0 else '--', label=f'Zero middle alpha={r["alpha"]:g}')
                ax.scatter(*z['actual_xy'][-1], color='.5', s=15, marker='D' if r['native_domain_failure'] else 's')
        goal = np.array([np.interp(float(path['goal']), path['s'], path['xy'][:, k]) for k in (0, 1)])
        ax.scatter(*goal, marker='*', color='black', label='Original world goal')
        envelope.append(goal[None, :])
        points = np.concatenate(envelope)
        lo, hi = points.min(axis=0) - .4, points.max(axis=0) + .4
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax.set(title=f'{route}, requested {speed:g} m/s', xlabel='World X (m)', ylabel='World Y (m)')
        ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=8)
    fig.suptitle('Historical native10s: H75 u77 / bank3 / zero force\nFixed midpoint vs endpoints, real stops; square=10s, diamond=domain, x=physical failure')
    fig.savefig(out / 'all_xy.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root.resolve())
