"""Per-layout rates and paired differences from fixed with a bootstrap over initial states."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from run_platform_task_oracle import read


def report(folder):
    plan = read(folder/'test_freeze_manifest.json')
    rows = pd.read_csv(folder/'online_cases.csv')
    keys = ['layout','skill','root','force','section','side']
    fixed = rows[rows.planner=='fixed'][keys+['success','collision']]
    cases = rows.merge(fixed,on=keys,suffixes=('','_fixed'),validate='many_to_one')
    roots = plan['roots']
    draw = np.random.default_rng(plan['root_bootstrap_seed']).integers(0,len(roots),(10000,len(roots)))
    records = []
    for label,g in cases.groupby(['layout','skill','planner']):
        ones = np.ones(len(g))
        vote = g.query_feasible & (g.query_batch_ms>0)
        veto = ~g.query_feasible & (g.query_batch_ms>0)
        metrics = {'success':(g.success,ones),'wall':(g.collision,ones),
                   'success_minus_fixed':(g.success.astype(int)-g.success_fixed.astype(int),ones),
                   'wall_minus_fixed':(g.collision.astype(int)-g.collision_fixed.astype(int),ones),
                   'admitted_success':(vote&g.success,vote),'admitted_wall':(vote&g.collision,vote),
                   'veto_oracle_unsolvable':(veto&~g.oracle_feasible,veto)}
        for name,(num,den) in metrics.items():
            totals=pd.DataFrame({'root':g.root,'n':num,'d':den}).groupby('root').sum().loc[roots].to_numpy(dtype=float)
            n,d=totals.sum(axis=0)
            samples=totals[draw].sum(axis=1)
            valid=samples[samples[:,1]>0]
            lo,hi=np.quantile(valid[:,0]/valid[:,1],[.025,.975]) if len(valid) else (np.nan,np.nan)
            records.append((*label,name,int(n),int(d),n/d if d else np.nan,lo,hi))
    pd.DataFrame(records,columns=['layout','skill','planner','metric','numerator','denominator','rate','low','high']).to_csv(folder/'layout_metrics.csv',index=False)

    # A preassigned physical comparison per layout and skill; both gate sides and all four methods.
    fig,axes=plt.subplots(len(plan['maps']),4,figsize=(17,3.4*len(plan['maps'])),layout='constrained')
    colours={'fixed':'#676767','nominal':'#a84b2a','C':'#2376a2','C-R':'#158059'}
    for i,(layout,m) in enumerate(plan['maps'].items()):
        for j,(skill,side) in enumerate((('source',1),('source',-1),('rb_run1',1),('rb_run1',-1))):
            ax=axes[i,j]
            group=rows[(rows.layout==layout)&(rows.skill==skill)&(rows.side==side)&(rows.root==roots[0])&(rows.force==0)&(rows.section=='near')]
            for r in group.itertuples():
                with np.load(r.trajectory) as z:
                    xy=z['qpos'][:,:2];ref=z['published_reference_xy']
                c=colours[r.planner]
                ax.plot(*ref.T,':',lw=.6,c=c);ax.plot(*xy.T,lw=1,c=c,label=r.planner)
                ax.scatter(*xy[-1],s=9,c=c)
            for x0,x1,y0,y1 in m['walls']: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='black'))
            ax.set(xlim=(1,19),ylim=(-1.4,1.4),aspect='equal',xlabel='world x (m)',ylabel='y (m)',title=f'{layout} / {skill} / '+('left' if side==1 else 'right'))
            ax.legend(fontsize=7,ncol=2)
    fig.suptitle(f'Root {roots[0]}, no push, near gate. Dotted: published route. Dot: actual endpoint.')
    fig.savefig(folder/'layout_xy.png',dpi=150);plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    report(p.parse_args().folder)
