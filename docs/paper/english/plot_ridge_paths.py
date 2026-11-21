"""Draw actual 200 Hz motion against the original fixed reference in two layouts."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('layout_run', type=Path)
args = parser.parse_args()
fig, axes = plt.subplots(1, 2, figsize=(7.1, 4.4), sharey=True)
colors = ['#0072B2', '#CC79A7', '#009E73', '#D55E00', '#78549A', '#555555']
for layout, (center, ax) in enumerate(zip([3.9, 4.5], axes)):
    run = args.layout_run / f'ridge100_v2.6_layout{layout}_seed401'
    contract = json.loads((run / 'contract.json').read_text())
    with np.load(run / 'trajectory.npz') as data:
        trace = data['lower_trace'].reshape(-1, 12, 11)
        failed = data['failure'].any(axis=0)
        progress = data['progress']
    with np.load(Path(contract['bundle']) / 'route.npz') as path:
        limit = max(progress[:, i][trace[:, i, 6].astype(bool)].max() for i in range(6))
        xy = path['xy'][path['s'] <= limit + 0.2]
        ax.plot(*xy.T, 'k:', lw=1.4, label='Original path')
        ax.plot(*[np.interp(center, path['s'], values) for values in path['xy'].T],
                marker='s', color='black', markersize=4, label='Ridge center')
    for i, label in enumerate(['0', '0.25', '0.5', '0.75', '1', 'Zero']):
        observed = trace[trace[:, i, 6].astype(bool), i, :2]
        ax.plot(*observed.T, color=colors[i], lw=1.2,
                ls='--' if i == 5 else '-', label=label)
        if failed[i]:
            ax.plot(*observed[-1], marker='x', color=colors[i])
    ax.set(title=f'Ridge center {center:.1f} m', xlabel='World x (m)', aspect='equal')
    ax.grid(alpha=0.18)
axes[0].set_ylabel('World y (m)')
fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc='lower center', frameon=False)
fig.tight_layout(rect=[0, .13, 1, 1])
fig.savefig(Path(__file__).parent / 'figures/ridge_paths.pdf', bbox_inches='tight')
