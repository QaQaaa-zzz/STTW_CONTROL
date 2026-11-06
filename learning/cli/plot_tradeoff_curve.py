"""Connect saved preference outcomes in alpha order, without fitting a frontier."""
import argparse
import json
from pathlib import Path
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


if __name__ == '__main__':
    main()
