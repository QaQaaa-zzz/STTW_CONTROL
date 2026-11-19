"""Export the complete two-width prefix screen and measured handback geometry."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('screens', nargs=2, type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
table = []
fig, axes = plt.subplots(2, 1, figsize=(11, 5.7), layout='constrained')
for folder, ax in zip(args.screens, axes, strict=True):
    cfg = json.loads((folder/'screen.json').read_text())
    width = cfg['ditch_terrain']['width_m']
    rows = list(csv.DictReader((folder/'prefixes.csv').open()))
    audits = json.loads((folder/'screen/jump_validation.json').read_text())['cases']
    seen = set()
    for row, audit in zip(rows, audits, strict=True):
        entry = {k: v for k, v in row.items() if k != 'source'}
        entry.update(width_m=width, flight=any(e['solid_bank_rear_launch'] or e['bank_to_bank_flight']
                     for e in audit['events']), handoff_y_m=None, handoff_yaw_deg=None)
        table.append(entry)
        if row['eligible'] != 'True':
            continue
        with np.load(row['source']) as z:
            tr = z['trace']
            k = np.flatnonzero(tr[:, -1] == 2)[0]
            entry.update(handoff_y_m=tr[k,2], handoff_yaw_deg=np.rad2deg(tr[k,6]))
            identity = (row['root'], z['qpos'][:k+1].tobytes())
            if identity in seen:
                continue
            seen.add(identity)
            ax.plot(*z['reference_xy'].T, ':', color='0.5', lw=.8)
            ax.plot(tr[:,1], tr[:,2], label=f"root {row['root']}, lane {row['lane']}")
            ax.scatter(*tr[k,[1,2]], s=22)
            ax.plot(*tr[-1,[1,2]], 'x', color='black')
    ax.set(xlim=(1,18), ylim=(-1.4,1.4), xlabel='World x (m)', ylabel='World y (m)',
           aspect='equal', title=f'{width} m gap')
    ax.legend(fontsize=8)
fig.suptitle('Frozen Source | actual and reference (dotted) | handback (dot), endpoint (cross)')
fig.savefig(args.output/'prefix_xy.png', dpi=145)
with (args.output/'prefixes.csv').open('w') as f:
    writer = csv.DictWriter(f, table[0], lineterminator='\n')
    writer.writeheader()
    writer.writerows(table)
