"""Publish contract rates, paired decision causes and physical trajectories for the 15 ms ablation."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def report(baseline, replay, out):
    out.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(baseline/'blind_metrics.csv').query("force == 'all' and planner.notna()")
    histories = pd.read_csv(baseline/'history_counts.csv').set_index(['skill','planner'])
    pairs = pd.read_csv(replay/'paired_cases.csv')
    methods = ['fixed','nominal','C','C-R']
    table = []
    for skill in ('source','rb_run1'):
        for planner in methods:
            rows = metrics[(metrics.planner == planner) & (metrics.skill == skill)].set_index('metric')
            item = dict(skill=skill,planner=planner,histories=histories.at[(skill,planner),'histories'])
            for m in ['success','collision','admitted_success','admitted_wall','veto_oracle_unsolvable']:
                r = rows.loc[m]
                item[m] = (f'{int(r.numerator)}/{int(r.denominator)} [{100*r.low:.1f}, {100*r.high:.1f}]'
                           if r.denominator else '—')
            table.append(item)
    pd.DataFrame(table).to_csv(out/'main_table.csv',index=False)
    cases = pairs.query("skill == 'rb_run1' and success_c != success_cr").sort_values(['success_c','layout','root','section'])
    cols = ['layout','root','force','section','side','success_c','success_cr','termination_c','termination_cr',
            'collision_c','collision_cr','margin_m','clearance_m','clear','in_domain','gate','goal_hold']
    cases[cols].to_csv(out/'changed_outcomes.csv',index=False)
    fig, axes = plt.subplots(10,4,figsize=(14,20),layout='constrained')
    for ax,r in zip(axes.flat,cases.itertuples()):
        for col,colour in [('c','#156298'),('cr','#bc5424')]:
            with np.load(getattr(r,'trajectory_'+col)) as z:
                xy=z['qpos'][:,:2]; ref=z['published_reference_xy']
            ax.plot(*ref.T,':',c=colour,lw=.7)
            ax.plot(*xy.T,c=colour,lw=1); ax.scatter(*xy[-1],s=9,c=colour)
        ax.set(aspect='equal',xlim=(9,19),ylim=(-1.3,1.3),
               title=f'{r.layout[-1].upper()} {r.root} {r.force:+}N {r.section} {r.side:+} | {int(r.success_c)}→{int(r.success_cr)}')
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(cases):]: ax.set_visible(False)
    fig.suptitle('Physical XY (m) after platform approach | C blue, C-R orange | dotted: published routes | dot: endpoint')
    fig.savefig(out/'paired_xy.png',dpi=128); plt.close(fig)
    paired = []
    for skill,group in pairs.groupby('skill'):
        for method in ('c','cr','safe'):
            g=group; succ=g['success_'+method]; wall=g['collision_'+method]; vote=g['query_feasible_'+method]
            delta=succ.astype(int)-g.success_c.astype(int)
            roots=pd.DataFrame({'root':g.root,'delta':delta,'wall_delta':wall.astype(int)-g.collision_c.astype(int),
                                'n':np.ones(len(g))}).groupby('root').sum().to_numpy()
            draws=np.random.default_rng(20261120).integers(0,len(roots),(10000,len(roots)))
            sums=roots[draws].sum(axis=1); ci=np.quantile(sums[:,:2]/sums[:,2:], [.025,.975],axis=0)*100
            paired.append(dict(skill=skill,method=method,contracts=len(g),success=int(succ.sum()),walls=int(wall.sum()),
                               admitted=int(vote.sum()),admitted_walls=int((vote&wall).sum()),rescued=int((delta>0).sum()),
                               lost=int((delta<0).sum()),changed_vs_cr=int((g['chosen_dep_'+method]!=g.chosen_dep_cr).sum()),
                               goal_delta_low_pp=ci[0,0],goal_delta_high_pp=ci[1,0],wall_delta_low_pp=ci[0,1],wall_delta_high_pp=ci[1,1]))
    pd.DataFrame(paired).to_csv(out/'paired_summary.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('baseline','replay','output'): p.add_argument(arg,type=Path)
    a=p.parse_args(); report(a.baseline,a.replay,a.output)
