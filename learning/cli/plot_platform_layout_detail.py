"""Plot paired physical exits on three wall layouts from archived trajectories."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


def plot(raw, output, arms=('fixed', 'C-R'), selected=False):
    plan = json.loads((raw/'test_freeze_manifest.json').read_text())
    records = pd.read_csv(raw/'online_cases.csv')
    if selected:
        pairs = pd.read_csv(raw/'matched_cases.csv')
        representatives = pairs.sort_values(['rescued','lost','changed'], ascending=False,
                                            kind='stable').drop_duplicates('layout')
        records = records[records.trajectory.isin(pd.concat(
            [representatives.c_trajectory, representatives.cr_trajectory]))]
    else:
        records = records[(records.root == plan['roots'][0]) & (records.force == 0)
                          & (records.section == 'near') & (records.side == 1)]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.8), sharex=True)
    fig.subplots_adjust(left=.12, right=.97, bottom=.08, top=.85, hspace=.43)
    colors = ('#a45220', '#0b5592')
    labels = {'alpha0':'Fixed preference', 'alpha_enabled':'Selectable preference'}
    for ax, (layout, title) in zip(axes, [('config_l','Left offset'), ('config_r','Right offset'),
                                         ('config_double','Staggered walls')]):
        rows = records[(records.layout == layout) & (records.skill == 'rb_run1')].set_index('planner')
        case = rows.loc[arms[0]]
        gate = plan['section_x'][['near','mid','far'].index(case.section)]
        ax.plot([gate,gate], [.06*case.side,3.9*case.side], color='#387558', lw=1)
        for x0,x1,y0,y1 in plan['maps'][layout]['walls']:
            ax.add_patch(Rectangle((x0,y0), x1-x0, y1-y0, color='#505050',zorder=3))
        for arm,color,style in zip(arms,colors,('-','--')):
            record = rows.loc[arm]
            with np.load(record.trajectory) as z:
                xy = z['trace'][:,1:3]
                ref = z['published_reference_xy']
            ax.plot(*ref.T, ':', color=color, lw=.9)
            ax.plot(*xy.T, style, color=color, lw=1.5)
            if not record.success:
                ax.scatter(*xy[-1], marker='x', s=30, color=color, zorder=4)
        ax.set(xlim=(1,20) if selected else (11.5,15.5), ylim=(-1.8,1.8) if selected else (-.95,.75),
               aspect='equal', ylabel='World y (m)')
        ax.set_title(f'{title}  |  start {case.root}, {case.force:+g} N, {case.section} gate',fontsize=10)
    axes[-1].set_xlabel('World x (m)')
    handles = [Line2D([],[],ls=ls,color=c,label=labels.get(arm,arm))
               for arm,c,ls in zip(arms,colors,('-','--'))]
    handles.extend([Line2D([],[],ls=':',color='.35',label='Published exit'),
                    Line2D([],[],color='#387558',label='Assigned gate'),
                    Patch(color='#505050',label='Wall'),
                    Line2D([],[],ls='',marker='x',color='.3',label='Failed endpoint')])
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=9)
    fig.savefig(output,dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('figure',type=Path)
    p.add_argument('--arms',nargs=2,default=['fixed','C-R'])
    p.add_argument('--selected', action='store_true')
    a=p.parse_args()
    plot(a.archive, a.figure, a.arms, a.selected)
