"""Summarize the paired Source entry-history diagnostic at its actual endpoints."""
import argparse
import csv
import json
from itertools import groupby
from pathlib import Path
from shutil import copyfile

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from run_playground_task_reversal import write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with (args.run/'paired.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == json.loads((args.run/'terminal.json').read_text())['scored']
    groups = [(key, list(cases)) for key, cases in groupby(rows, key=lambda r:(r['group'], r['root']))]
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 2*len(groups)),
                             layout='constrained', squeeze=False)
    summary = []
    for ax, ((group, root), cases) in zip(axes[:, 0], groups):
        original = {'discovery':'task_reversal_bend_prefixes',
                    'new_roots':'task_reversal_new_domain/group_0'}[group]
        for stage in ('flight', 'loaded', 'qualified'):
            before = np.array([r['original_'+stage]=='True' for r in cases])
            after = np.array([r['ground_actions_'+stage]=='True' for r in cases])
            summary.append(dict(group=group, root=root, stage=stage, cases=len(cases),
                                original=sum(before), ground_actions=sum(after),
                                retained=sum(before & after), gained=sum(~before & after),
                                lost=sum(before & ~after), neither=sum(~before & ~after)))
        references = set()
        for case in cases:
            for folder, color in ((args.run.parent/original/'screen', '#0072b2'),
                                  (args.run/group, '#d55e00')):
                with np.load(folder/case['case']/'trajectory.npz') as z:
                    xy, ref = z['qpos'][:, :2], z['reference_xy']
                route = ref.tobytes()
                if route not in references:
                    ax.plot(*ref.T, color='0.4', ls=':', lw=.7)
                    references.add(route)
                ax.plot(*xy[::5].T, color=color, lw=.7, alpha=.65)
                ax.plot(*xy[-1], marker='x', color=color, ms=4)
        ax.set(aspect='equal', xlim=(1,18), ylim=(-1.6,1.6), xlabel='World x (m)',
               ylabel='World y (m)', title=f'Start {root} | {len(cases)} pairs')
    fig.legend([Line2D([], [], color=c, ls=style) for c, style in
                [('#0072b2','-'),('#d55e00','-'),('0.4',':')]],
               ['Original history','Incoming ground actions','Reference'], loc='outside upper center', ncol=3)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'incoming_actions_xy.png', dpi=140)
    write_rows(args.output/'incoming_actions_counts.csv', summary)
    copyfile(args.run/'paired.csv', args.output/'incoming_actions_pairs.csv')


if __name__ == '__main__':
    main()
