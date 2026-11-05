"""Make manuscript figures from the complete matched raw-trajectory inventory."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.analysis / 'case_metrics.json').read_text())
    if len(rows) not in (192, 288) or {r['method'] for r in rows} != {'H75', 'PP'}:
        parser.error('expected completed H75 + Pure Pursuit low/mid or full matrix')
    args.out.mkdir(parents=True, exist_ok=True)
    colors = {'PP': '#333333', 0.0: '#0072B2', 0.5: '#009E73', 1.0: '#D55E00'}
    styles = {'PP': '-', 0.0: '--', 0.5: '-.', 1.0: '-'}
    labels = {'PP': 'Pure pursuit', 0.0: r'H75 $\alpha=0$', 0.5: r'H75 $\alpha=0.5$', 1.0: r'H75 $\alpha=1$'}
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6))
    for i, bank in enumerate((2, 5)):
        matched = {('PP' if r['method'] == 'PP' else r['alpha']): r for r in rows
                   if r['bank_index'] == bank and r['route'] == 'low_s_left_R1p55' and
                   r['event'] == 't2.8_F+36'}
        ax_xy, ax_speed = axes[i]
        path = np.load(matched['PP']['original_path_file'])
        m = path['s'] <= float(path['goal'])
        ax_xy.plot(*path['xy'][m].T, color='#888888', ls=':', lw=1.5, label='Original path')
        for key in ('PP', 0.0, 0.5, 1.0):
            r = matched[key]
            raw = np.load(r['raw_file'])
            n = r['observed_ticks']
            ax_xy.plot(*raw['actual_xy'][:n].T, color=colors[key], ls=styles[key], lw=1.2, label=labels[key])
            ax_speed.plot(raw['time'][:n], raw['actual_forward_speed'][:n],
                          color=colors[key], ls=styles[key], lw=1.2, label=labels[key])
            ax_xy.plot(*raw['actual_xy'][n - 1], marker='o' if r['arrived'] else 'x', ms=3,
                       color=colors[key], ls='none')
        ax_xy.plot(*matched['PP']['original_goal_xy'], marker='*', ms=6, color='#555555')
        ax_xy.set(xlabel='World x (m)', ylabel='World y (m)',
                  title=f'({"a" if i == 0 else "c"}) Bank {bank}: trajectory')
        ax_xy.set_aspect('equal')
        ax_speed.axhline(1.9, color='#888888', ls=':', lw=1)
        ax_speed.axvspan(2.8, 2.95, color='#bbbbbb', alpha=.35)
        ax_speed.set(xlabel='Time (s)', ylabel='True forward speed (m/s)',
                     title=f'({"b" if i == 0 else "d"}) Bank {bank}: measured speed')
        for ax in (ax_xy, ax_speed):
            ax.grid(alpha=.2)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(left=.08, right=.99, bottom=.11, top=.84, wspace=.27, hspace=.52)
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .99), ncol=5, frameon=False)
    fig.savefig(args.out / 'low_s_pulse_matched.pdf')
    fig.savefig(args.out / 'low_s_pulse_matched.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25))
    for ax, route, title in zip(axes, ('mid_sharp90_left', 'mid_sharp90_right'),
                                ('(a) Exact left corner', '(b) Exact right corner')):
        matched = {('PP' if r['method'] == 'PP' else r['alpha']): r for r in rows
                   if r['bank_index'] == 2 and r['route'] == route and r['event'] == 'zero'}
        path = np.load(matched['PP']['original_path_file'])
        m = path['s'] <= float(path['goal'])
        if np.max(np.abs(path['curvature'])) != 0 or abs(float(path['goal'] - path['turn_end']) - 7.) > 1e-9:
            raise ValueError('corner path is not the declared exact polyline')
        ax.plot(*path['xy'][m].T, color='#222222', ls=':', lw=1.7, label='Exact original path')
        for key in ('PP', 0.0, 0.5, 1.0):
            r = matched[key]
            raw = np.load(r['raw_file'])
            n = r['observed_ticks']
            ax.plot(*raw['actual_xy'][:n].T, color=colors[key], ls=styles[key], lw=1.2, label=labels[key])
            ax.plot(*raw['actual_xy'][n - 1], marker='o' if r['arrived'] else 'x', ms=3,
                    color=colors[key], ls='none')
        ax.plot(*matched['PP']['original_goal_xy'], marker='*', ms=7, color='#222222')
        ax.set(xlabel='World x (m)', ylabel='World y (m)', title=title)
        ax.set_aspect('equal')
        ax.grid(alpha=.2)
    handles, names = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=.08, right=.99, bottom=.17, top=.82, wspace=.35)
    fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .99), ncol=5, frameon=False)
    fig.savefig(args.out / 'exact_sharp90_matched.pdf')
    fig.savefig(args.out / 'exact_sharp90_matched.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
