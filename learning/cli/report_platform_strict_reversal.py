"""Summarize paired, finite-library reversal and task-conditioned constant policies."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def report(raw, out):
    trials = pd.read_csv(raw/'physical_scores.csv')
    paired = pd.read_csv(raw/'paired_reversal.csv')
    sections = ['near','mid','far']
    keys = ['layout','skill','section','root']
    solutions = trials.groupby(keys+['approach'])[['left_success','right_success']].any()
    left, right = solutions.left_success, solutions.right_success
    by_root = pd.DataFrame(dict(only_left=left & ~right,only_right=right & ~left,
                                universal=right & left)).groupby(keys).any()
    by_root = paired.set_index(keys).join(by_root)
    by_root['eligible'] = by_root.paired_prefixes & (by_root.eligible_approaches>=2)
    by_root['exclusive'] = by_root.only_left & by_root.only_right & by_root.eligible
    by_root['strict'] = by_root.exclusive & ~by_root.universal
    by_root.to_csv(out/'p1_roots.csv')
    result = []
    figure, axes = plt.subplots(2,2,figsize=(9,6),layout='constrained')
    for (layout,skill,section), roots in by_root.groupby(keys[:3]):
        rows = trials[(trials.layout==layout)&(trials.skill==skill)&(trials.section==section)]
        totals = rows.groupby('root')[['left_success','right_success']].any().sum()
        fixed = rows[(rows.approach==1)&(rows.departure=='left')].left_success.sum()+rows[(rows.approach==0)&(rows.departure=='right')].right_success.sum()
        result.append(dict(layout=layout,skill=skill,section=section,total=len(roots),eligible=roots.eligible.sum(),
                           exclusive=roots.exclusive.sum(),strict=roots.strict.sum(),universal=roots.universal.sum(),
                           fixed=fixed,oracle_left=totals.left_success,oracle_right=totals.right_success,
                           oracle=totals.sum(),gap=totals.sum()-fixed))
        if section!='near':
            continue
        pick=roots[roots.strict]
        root=(pick if len(pick) else roots).index[0][-1]
        ax=axes[int(layout=='alternate'),int(skill=='rb_run1')]
        for row in rows[rows.root==root].itertuples():
            with np.load(row.trajectory) as z:
                tr=z['trace']; xy=z['published_reference_xy']
            c=['#4d5156','#078274','#aa4f22'][row.approach]
            ax.plot(*xy.T,':',lw=.6,color=c,alpha=.65)
            ax.plot(tr[:,1],tr[:,2],lw=1,color=c)
            ax.scatter(*tr[-1,1:3],s=10,color=c,marker='o' if row.goal else 'x')
        ax.set(xlim=(2,18.6),xlabel='world x (m)',ylabel='world y (m)',title=f'{layout} / {skill} / root {root}',aspect='equal')
    figure.suptitle('Near section witnesses, actual and issued reference XY')
    figure.savefig(out/'p1_xy.png',dpi=140)
    pd.DataFrame(result).to_csv(out/'p1_summary.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args();report(args.raw,args.out)
