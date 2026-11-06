"""Plot saved same-state alpha queries; no policy inference or physics."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    alphas = [0, .25, .5, .75, 1]
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#9B7600']
    markers = ['o', 's', '^', 'D', 'v']
    fig, axes = plt.subplots(4, 5, figsize=(16, 11), sharex=True,
                             sharey='row', layout='constrained')
    evidence = []
    for phase_index, phase in enumerate(('entry_projection', 'exit_projection')):
        for col, source_alpha in enumerate(alphas):
            source = args.analysis / phase / f'alpha{source_alpha:g}' / 'same_state_alpha/queries.csv'
            with source.open() as stream:
                rows = list(csv.DictReader(stream))
            query = [float(r['alpha']) for r in rows]
            if query != alphas:
                raise ValueError(f'Expected all five ordered alpha queries: {source}')
            evidence.append(dict(source=str(source.resolve()), phase=phase,
                source_alpha=source_alpha, time_s=float(rows[0]['time_s'])))
            for channel, units, offset in [('speed', 'm/s', 0), ('steer', 'rad', 1)]:
                ax = axes[phase_index * 2 + offset, col]
                target = [float(r[f'target_{channel}_correction']) for r in rows]
                previous = float(rows[0][f'previous_{channel}_correction'])
                ax.axhline(previous, color='.45', linestyle='--', linewidth=1,
                           label='Previous executed correction')
                for alpha, value, color, marker in zip(query, target, colors, markers):
                    ax.scatter(alpha, value, color=color, marker=marker, s=45, zorder=3)
                ax.set_title(f'{phase.split("_")[0]} | source alpha={source_alpha:g}\n'
                             f't={float(rows[0]["time_s"]):.2f} s', fontsize=10)
                ax.set_xticks(alphas, ['0', '.25', '.5', '.75', '1'])
                ax.grid(alpha=.2)
                if col == 0:
                    ax.set_ylabel(f'Mapped {channel} correction ({units})')
                if phase_index == 1 and offset == 1:
                    ax.set_xlabel('Queried alpha')
    axes[0, 0].legend(fontsize=7, loc='best')
    fig.suptitle('Original H75 u77 | 10 saved states, 5 queries per state | DEV\n'
                 'Within each panel only alpha changes; states differ between columns. '
                 'Dots are targets, not vehicle motion; dashed line is prior executed correction.', fontsize=12)
    output = args.analysis / 'same_state_alpha_overview.png'
    fig.savefig(output, dpi=160)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(dict(
        physics_steps=0, optimizer_updates=0, smoothing=False,
        scope='Frozen-state forward queries; no new closed-loop evidence',
        sources=evidence), indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
