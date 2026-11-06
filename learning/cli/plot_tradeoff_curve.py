"""Connect saved preference outcomes in alpha order, without fitting a frontier."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.evaluation / 'tradeoff_common_window.json').read_text())
    groups = {}
    for row in rows:
        groups.setdefault((row['terrain'], row['user_speed_m_s']), []).append(row)
    labels = ['alpha 0', 'alpha .25', 'alpha .5', 'alpha .75', 'alpha 1']
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#8A6500']
    markers = ['o', 's', '^', 'D', 'v']
    diagnostics = []
    for detail in (False, True):
        fig, axes = plt.subplots(2, 5, figsize=(18, 7), layout='constrained')
        for ax, ((terrain, speed), group) in zip(axes.flat, groups.items()):
            lookup = {r['label']: r for r in group}
            ordered = [lookup[label] for label in labels]
            x = [r['path_rmse_m'] for r in ordered]
            y = [r['speed_rmse_m_s'] for r in ordered]
            ax.plot(x, y, color='.45', linewidth=1.4, zorder=1)
            for xx, yy, label, color, marker in zip(x, y, labels, colors, markers):
                ax.scatter(xx, yy, color=color, marker=marker, s=35, label=label, zorder=3)
            if not detail:
                zero = lookup['Zero']
                ax.scatter(zero['path_rmse_m'], zero['speed_rmse_m_s'], color='.3',
                           marker='X', s=45, label='Zero (separate controller)')
            ax.set(title=f'{terrain} | {speed} m/s\nwindow 0–{ordered[0]["common_window_end_s"]:.2f} s',
                   xlabel='Path RMSE (m)', ylabel='Speed RMSE (m/s)')
            ax.grid(alpha=.2)
            if not detail:
                diagnostics.append(dict(terrain=terrain, requested_speed=speed,
                    segments=[dict(alpha_from=i*.25, alpha_to=(i+1)*.25,
                        delta_path_m=x[i+1]-x[i], delta_speed_error_m_s=y[i+1]-y[i],
                        slope_per_s=(y[i+1]-y[i])/(x[i+1]-x[i])) for i in range(4)]))
        axes[0, 0].legend(fontsize=6.5)
        fig.suptitle('Original H75 + Local300 | dots connected in alpha order 0 → .25 → .5 → .75 → 1\n'
                     'Straight segments are visual guides, not fitted dynamics or a Pareto frontier. '
                     + ('H75 detail; Zero omitted only from this view.' if detail else 'Zero is not connected.')
                     + ' Same saved paired windows; one initial state, DEV.', fontsize=11)
        name = 'tradeoff_common_window_connected' + ('_detail' if detail else '')
        fig.savefig(args.evaluation / (name + '.png'), dpi=150)
        plt.close(fig)
    (args.evaluation / 'tradeoff_curve_segments.json').write_text(json.dumps(
        diagnostics, indent=2) + '\n')
    # Saved roll, speed and pose are post-step signals. Yaw rate is the backward
    # difference between consecutive post-step poses; no smoothing is applied.
    route_path = Path('runs/h75_selected_local300_flat/bundle/route.npz')
    route = np.load(route_path)
    dynamics = []
    for case in json.loads((args.evaluation / 'index.json').read_text()):
        data = np.load(Path(case['run']) / 'trajectory.npz')
        trace = data['lower_trace'].reshape(-1, 6, 11)
        reference_curvature = np.interp(data['progress'], route['s'], route['curvature'])
        mask = trace[:, :, 6].astype(bool).all(1) & (abs(reference_curvature[:, :5]) > .1).all(1)
        mask = mask[1:]
        speed = trace[1:, :, 2]
        yaw_rate = np.diff(np.unwrap(data['pose'][:, :, 2], axis=0), axis=0) / .005
        for lane, alpha in enumerate([0, .25, .5, .75, 1]):
            selected_speed = speed[mask, lane]
            dynamics.append(dict(terrain=case['terrain'], request=case['user_speed_m_s'],
                alpha=alpha, sample_count=int(mask.sum()),
                mean_speed=float(selected_speed.mean()),
                mean_yaw_curvature=float((yaw_rate[mask, lane] / selected_speed).mean()),
                mean_abs_roll=float(abs(data['measurement'][1:][mask, lane, 0]).mean()),
                mean_abs_v_yaw_rate=float(abs(selected_speed * yaw_rate[mask, lane]).mean()),
                mean_abs_g_tan_roll=float((9.8 * abs(np.tan(data['measurement'][1:][mask, lane, 0]))).mean())))
    (args.evaluation / 'tradeoff_bend_dynamics.json').write_text(json.dumps(dict(
        scope='Shared time mask: all five H75 lanes at abs(reference curvature)>0.1/m; '
              'all six lanes active. Post-step signals; yaw rate backward difference at5ms. '
              'Yaw-rate/speed is a low-slip curvature proxy, not geometric XY curvature. '
              'No constant-lean assumption, smoothing, new inference or physics.',
        route=str(route_path), rows=dynamics), indent=2) + '\n')
    fig, axes = plt.subplots(3, 5, figsize=(17, 9), sharex=True, sharey='row', layout='constrained')
    fields = [('mean_speed', 'Mean speed (m/s)'), ('mean_yaw_curvature', 'Mean yaw-rate / speed (1/m)'),
              ('mean_abs_roll', 'Mean absolute roll (rad)')]
    for col, terrain in enumerate(dict.fromkeys(r['terrain'] for r in dynamics)):
        for request, style in [(2.3, '-'), (2.6, '--')]:
            selected = [r for r in dynamics if r['terrain'] == terrain and r['request'] == request]
            for row, (field, ylabel) in enumerate(fields):
                ax = axes[row, col]
                ax.plot([r['alpha'] for r in selected], [r[field] for r in selected],
                        color='.45', linestyle=style, label=f'Request {request} m/s')
                for r, color, marker in zip(selected, colors, markers):
                    ax.scatter(r['alpha'], r[field], color=color, marker=marker, s=30, zorder=3)
                ax.grid(alpha=.2)
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == 0:
                    ax.set_title(terrain)
                if row == 2:
                    ax.set_xlabel('Alpha')
                ax.set_xticks([0, .25, .5, .75, 1])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Saved H75 bend dynamics | five alpha colors/markers as in tradeoff plot\n'
                 'Shared bend time window within each condition; conditions have different windows. '
                 'Descriptive means, one initial state, DEV; no steady-equilibrium fit.', fontsize=11)
    fig.savefig(args.evaluation / 'tradeoff_bend_dynamics.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
