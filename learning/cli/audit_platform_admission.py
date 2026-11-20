"""Audit executed and shadow votes, including outcomes when the oracle has no solution."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def audit(raw, dest):
    df=pd.read_csv(raw/'online/online_cases.csv')
    queried=df.query_batch_ms>0
    df['vote']=np.select([~queried,df.query_feasible],['no_query','admitted'],default='fallback')
    df['unsolvable']=~df.oracle_feasible
    df['unsolvable_wall']=df.unsolvable & df.collision
    df['feasible_admitted']=df.oracle_feasible & df.query_feasible
    fields=dict(contracts=('root','size'),complete=('success','sum'),wall=('collision','sum'),
                oracle_feasible=('oracle_feasible','sum'),unsolvable_wall=('unsolvable_wall','sum'),
                feasible_admitted=('feasible_admitted','sum'))
    df.groupby(['planner','vote']).agg(**fields).to_csv(dest/'admission.csv')
    df.groupby(['planner','skill','side','vote']).agg(**fields).to_csv(dest/'admission_by_task.csv')
    keys=['layout','skill','root','section','side','force']
    fixed=df[df.planner=='fixed'].set_index(keys)
    cr=df[df.planner=='C-R'].set_index(keys).reindex(fixed.index)
    avoided=fixed[fixed.collision & ~cr.collision].copy()
    avoided['queried']=cr.loc[avoided.index,'query_batch_ms']>0
    avoided.groupby('skill').agg(avoided=('collision','sum'),oracle_unsolvable=('unsolvable','sum'),
                                 queried=('queried','sum')).to_csv(dest/'avoided_wall.csv')
    scores=pd.read_csv(raw/'physical_scores.csv')
    match=scores[(scores.layout=='alternate')&(scores.skill=='rb_run1')&(scores.approach==1)&
                 (scores.departure=='left')&(scores.section=='near')&scores.root.isin([96,110])]
    fig,(xy,angle)=plt.subplots(2,1,figsize=(8,5),layout='constrained')
    states=[]
    for r in match.itertuples():
        with np.load(r.trajectory) as z:
            q=z['qpos'];t=z['trace'][:,0];h=round(r.handoff_s/.005)+1
            line,=xy.plot(*q[h:,:2].T,label=f'{r.root}  {r.termination}')
            ref=z['published_reference_xy'];xy.plot(*ref.T,':',color=line.get_color(),alpha=.55)
            xy.scatter(*q[h,:2],marker='o',color=line.get_color());xy.scatter(*q[-1,:2],marker='x',color=line.get_color())
            angle.plot(t[h:]-t[h],np.rad2deg(q[h:,8]),color=line.get_color())
            states.append(dict(root=r.root,x=q[h,0],y=q[h,1],steer_deg=np.rad2deg(q[h,8]),result=r.termination))
    xy.set(aspect='equal',xlim=(9,18.5),xlabel='World x (m)',ylabel='World y (m)')
    xy.legend(fontsize=8);angle.set(xlabel='Time after handback (s)',ylabel='Steering angle (deg)')
    fig.savefig(dest/'paired_state_xy.png',dpi=150);plt.close(fig)
    pd.DataFrame(states).to_csv(dest/'paired_state.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw',type=Path);p.add_argument('dest',type=Path)
    args=p.parse_args();audit(args.raw,args.dest)
