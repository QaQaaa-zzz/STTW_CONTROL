"""Project existing world impulses onto recorded yaw axes, without simulation."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_table(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(root):
    plan = json.loads((root / 'plan.json').read_text())
    for path, sha in plan['sources_sha256'].items():
        assert digest(path) == sha, path
    paths = list(plan['sources_sha256'])
    source = table(paths[-1])
    endpoints = {r['key']: r for r in table(paths[2])}
    records = []
    for row in source:
        assert digest(row['raw_file']) == row['raw_sha256']
        with np.load(row['raw_file']) as raw:
            active = raw['active_tick'].astype(bool)
            impulse = np.asarray(raw['applied_force_impulse'], dtype=float)
            yaw = np.asarray(raw['actual_yaw'], dtype=float)
            speed = np.asarray(raw['actual_forward_speed'], dtype=float)
            progress = np.asarray(raw['path_progress'], dtype=float)
            assert np.all(np.isfinite(impulse[active]))
            mask = active & (np.linalg.norm(impulse, axis=1) > 0)
            assert not np.any((~active) & (np.linalg.norm(impulse, axis=1) > 0))
            indices = np.flatnonzero(mask)
            out = dict(key=row['key'], role=row['role'], route=row['route'],
                       alpha=None if row['role'] == 'zero_middle' else float(row['alpha']),
                       force_N=float(row['force_N']), onset=int(row['onset_tick']),
                       survival=endpoints[row['key']]['survival'] == 'True',
                       goal=endpoints[row['key']]['goal'] == 'True',
                       nonzero_ticks=len(indices), raw_sha256=row['raw_sha256'])
            total = impulse[active].sum(axis=0)
            out.update(world_Jx_Ns=float(total[0]), world_Jy_Ns=float(total[1]),
                       world_Jz_Ns=float(total[2]), world_J_norm_Ns=float(np.linalg.norm(total)))
            if len(indices):
                assert indices[0] > 0
                j = impulse[mask]
                y = yaw[mask]
                pre_yaw = yaw[indices - 1]
                lateral = -np.sin(y) * j[:, 0] + np.cos(y) * j[:, 1]
                longitudinal = np.cos(y) * j[:, 0] + np.sin(y) * j[:, 1]
                old_lateral = -np.sin(pre_yaw) * j[:, 0] + np.cos(pre_yaw) * j[:, 1]
                out.update(exposure='observed', first_force_index=int(indices[0]),
                           last_force_index=int(indices[-1]),
                           pre_yaw_rad=float(yaw[indices[0] - 1]),
                           yaw_span_rad=float(np.ptp(np.unwrap(y))),
                           body_plane_Jlat_Ns=float(lateral.sum()),
                           body_plane_Jlong_Ns=float(longitudinal.sum()),
                           previous_yaw_Jlat_Ns=float(old_lateral.sum()),
                           previous_yaw_projection_difference_Ns=float(abs(lateral.sum() - old_lateral.sum())),
                           pre_speed_m_s=float(speed[indices[0] - 1]),
                           pulse_speed_mean_m_s=float(speed[mask].mean()),
                           pre_progress_m=float(progress[indices[0] - 1]))
            else:
                out.update(exposure='nominal' if out['force_N'] == 0 else 'not_exposed',
                           **{k: None for k in ('first_force_index', 'last_force_index', 'pre_yaw_rad',
                               'yaw_span_rad', 'body_plane_Jlat_Ns', 'body_plane_Jlong_Ns',
                               'previous_yaw_Jlat_Ns', 'previous_yaw_projection_difference_Ns',
                               'pre_speed_m_s', 'pulse_speed_mean_m_s', 'pre_progress_m')})
            records.append(out)
    roles = json.loads(Path(paths[0]).read_text())['learned_roles']
    pairs = []
    for role, route in itertools.product(roles, sorted({r['route'] for r in records})):
        group = [r for r in records if r['role'] == role and r['route'] == route and r['force_N']]
        for onset, force in sorted({(r['onset'], r['force_N']) for r in group}):
            subset = sorted([r for r in group if r['onset'] == onset and r['force_N'] == force], key=lambda r: r['alpha'])
            assert len(subset) == 5
            for a, b in itertools.combinations(subset, 2):
                out = dict(role=role, route=route, onset=onset, force_N=force,
                           alpha_a=a['alpha'], alpha_b=b['alpha'], key_a=a['key'], key_b=b['key'])
                observed = a['exposure'] == b['exposure'] == 'observed'
                out['both_exposed'] = observed
                for field in ('world_Jx_Ns', 'world_Jy_Ns', 'body_plane_Jlat_Ns',
                              'body_plane_Jlong_Ns', 'pre_yaw_rad', 'pre_speed_m_s', 'pre_progress_m'):
                    out[field + '_difference'] = b[field] - a[field] if observed else None
                pairs.append(out)
    assert len(records) == 434 and len(pairs) == 720
    exposed = [r for r in records if r['exposure'] == 'observed']
    summary = dict(n_records=len(records), distinct_raw_sha256=len({r['raw_sha256'] for r in records}),
                   nominal=sum(r['exposure'] == 'nominal' for r in records),
                   exposed=len(exposed), not_exposed=sum(r['exposure'] == 'not_exposed' for r in records),
                   n_related_alpha_pairs=len(pairs),
                   max_post_previous_yaw_projection_difference_Ns=max(r['previous_yaw_projection_difference_Ns'] for r in exposed),
                   max_within_role_world_J_component_difference_Ns=max(abs(p[k + '_difference']) for p in pairs if p['both_exposed'] for k in ('world_Jx_Ns', 'world_Jy_Ns')),
                   reversal_example=[r for r in records if r['role'] == 'task_only' and r['route'] == 'right' and r['onset'] == 456 and r['force_N'] and r['alpha'] in (0., .5, .75)],
                   new_physics_ticks=0, new_training_steps=0,
                   limits=plan['limits'])
    write_table(root / 'records.csv', records)
    write_table(root / 'alpha_pairs.csv', pairs)
    (root / 'review.json').write_text(json.dumps(summary, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00', '#666666']
    styles = ['-', '--', '-.', ':', '-', '--']
    for route in sorted({r['route'] for r in records}):
        events = sorted({(r['onset'], r['force_N']) for r in records if r['route'] == route and r['force_N']})
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for ax, (onset, force) in zip(axes.flat, events):
            for role, color, style in zip(roles, colors, styles):
                group = sorted([r for r in records if r['route'] == route and r['role'] == role and r['onset'] == onset and r['force_N'] == force], key=lambda r: r['alpha'])
                ax.plot([r['alpha'] for r in group], [r['body_plane_Jlat_Ns'] for r in group], color=color, linestyle=style, marker='.', label=role)
            z = next(r for r in records if r['role'] == 'zero_middle' and r['route'] == route and r['onset'] == onset and r['force_N'] == force)
            ax.axhline(z['body_plane_Jlat_Ns'], color='black', linestyle=':', label='zero (one trace)')
            ax.set(title=f'onset tick {onset}, {force:+g} N', xlabel='executed alpha', ylabel='recorded-yaw lateral impulse (N s)', xticks=[0, .25, .5, .75, 1],
                   ylim=(-3.3, 0) if force < 0 else (0, 3.3))
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncols=4)
        fig.suptitle(f'{route}: all disturbed contexts, 0.1 s world-force pulse\nYaw-plane projection of saved per-tick impulses; not exact substep body-frame load or new trials')
        fig.savefig(root / f'{route}_lateral_exposure.png', dpi=160)
        plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != 'reversal_example'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    main(parser.parse_args().root)
