"""Calibrate on primary roots 8..77 and score matched held-out physical routes."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def stats(rows, margin):
    clear=(rows.clear>margin)&np.isfinite(rows.clear)
    passed=clear&rows.reachable
    wall=rows.termination=='forbidden_obstacle_contact'
    timeout=rows.termination=='timeout'
    goal=rows.goal
    side=np.sign(rows.near)
    near_truth=rows.left_success|rows.right_success
    hit=passed&(rows.near.abs()>=.06)&(rows.near.abs()<=3.9)&(((side>0)&rows.left_success)|((side<0)&rows.right_success))
    return dict(physical=len(rows),goals=int(goal.sum()),accepted=int(passed.sum()),
                goal_hits=int((passed&goal).sum()),goal_reject=int((~passed&goal).sum()),
                near_task_goals=int(near_truth.sum()),near_task_hits=int(hit.sum()),
                collisions=int(wall.sum()),collision_clear=int((clear&wall).sum()),
                collision_accepted=int((passed&wall).sum()),timeouts=int(timeout.sum()),
                timeout_accepted=int((passed&timeout).sum()))


def score(raw,out):
    scores=pd.concat(pd.read_csv(p) for p in sorted(raw.glob('forecasts_*.csv')))
    records=[]; near_pairs=[]
    for skill,rows in scores.groupby('skill'):
        kin=rows[(rows.layout=='primary')&(rows.root<78)&(rows.method=='kinematic')]
        cx=kin.loc[kin.termination=='forbidden_obstacle_contact','clear'].sort_values(ascending=False)
        margin=max(0.,cx.iloc[int(len(cx)*.02)]) if len(cx) else 0.
        base=stats(kin,margin)
        target=base['goal_hits']/base['goals']
        for method,part in rows.groupby('method'):
            cal=part[(part.layout=='primary')&(part.root<78)]
            feasible=cal[cal.goal&cal.reachable].clear.sort_values(ascending=False).to_numpy()
            needed=round(target*cal.goal.sum())
            matched=max(0.,np.nextafter(feasible[min(needed,len(feasible))-1],-np.inf)) if len(feasible) and needed else np.inf
            for label,eps in [('bare',0.),('shared',margin),('cal_recall',matched)]:
                for (layout,holdout),test in part.groupby(['layout',part.root>=78]):
                    records.append(dict(skill=skill,method=method,margin=label,epsilon_m=eps,layout=layout,
                                        split='heldout44' if holdout else 'calibration70',**stats(test,eps)))
        test=rows[(rows.method=='kinematic')&(rows.root>=78)]
        for (layout,approach,dep),cell in test.groupby(['layout','approach','departure']):
            pos=cell[cell.goal];neg=cell[~cell.goal]
            if pos.empty or neg.empty:continue
            def xyz(df):
                yaw=np.arctan2(2*(df.qw*df.qz+df.qx*df.qy),1-2*(df.qy**2+df.qz**2))
                return np.c_[df.x/.1,df.y/.1,yaw/np.deg2rad(5),df.speed/.2]
            dist=np.linalg.norm(xyz(pos)[:,None]-xyz(neg)[None],axis=-1)
            i,j=np.unravel_index(dist.argmin(),dist.shape);a,b=pos.iloc[i],neg.iloc[j]
            def roll(v):
                return np.arctan2(2*(v.qw*v.qx+v.qy*v.qz),1-2*(v.qx**2+v.qy**2))
            def yaw(v):
                return np.arctan2(2*(v.qw*v.qz+v.qx*v.qy),1-2*(v.qy**2+v.qz**2))
            near_pairs.append(dict(layout=layout,skill=skill,approach=approach,departure=dep,
                                   goal_root=a.root,failed_root=b.root,termination=b.termination,
                                   dx_m=a.x-b.x,dy_m=a.y-b.y,dvx_m_s=a.speed-b.speed,
                                   dyaw_deg=np.rad2deg(np.angle(np.exp(1j*(yaw(a)-yaw(b))))),
                                   droll_deg=np.rad2deg(roll(a)-roll(b)),
                                   dsteer_deg=np.rad2deg(a.steer-b.steer),normalized_distance=dist[i,j]))
    pd.DataFrame(records).to_csv(out/'p2_summary.csv',index=False)
    pd.DataFrame(near_pairs).to_csv(out/'p2_pairs.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('raw',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();score(args.raw,args.out)
