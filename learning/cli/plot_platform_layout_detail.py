"""Compare the two physical paths on each wall layout at the first assigned start."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


def plot(raw, output, arms=('fixed','C-R')):
    plan=json.loads((raw/'test_freeze_manifest.json').read_text())
    rows=pd.read_csv(raw/'online_cases.csv')
    rows=rows[(rows.root==plan['roots'][0]) & (rows.force==0) & (rows.section=='near') &
              (rows.side==1) & (rows.skill=='rb_run1')]
    fig,axes=plt.subplots(3,1,figsize=(6.1,3.6),sharex=True,sharey=True,layout='constrained')
    for ax,(layout,title) in zip(axes,[('config_l','Left offset'),('config_r','Right offset'),('config_double','Staggered')]):
        for name,color,style in zip(arms,('#a45220','#0b5592'),('-','--')):
            case=rows[(rows.layout==layout)&(rows.planner==name)].iloc[0]
            with np.load(case.trajectory) as z:
                xy=z['qpos'][:,:2]
                ref=z['published_reference_xy']
            ax.plot(*ref.T,':',color=color,lw=.9)
            ax.plot(*xy.T,style,color=color,lw=1.7,label={'alpha0':'Fixed preference','alpha_enabled':'Selectable preference'}.get(name,name))
            if case.collision: ax.scatter(*xy[-1],marker='x',s=40,color=color,zorder=5)
        for x0,x1,y0,y1 in plan['maps'][layout]['walls']:
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='#484848'))
        ax.axvline(plan['section_x'][0],color='.65',lw=.8)
        ax.set(xlim=(11.5,15.5),ylim=(-.55,.65),aspect='equal',ylabel='y (m)')
        ax.text(.02,.92,title,va='top',transform=ax.transAxes,fontsize=9)
    axes[0].legend(loc='upper right',ncol=2,fontsize=8)
    axes[-1].set_xlabel('world x (m)')
    fig.savefig(output,dpi=220,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('figure',type=Path)
    a=p.parse_args();plot(a.archive,a.figure)
