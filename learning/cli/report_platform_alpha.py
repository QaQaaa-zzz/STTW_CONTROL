"""Paired physical speed/alpha contrasts and leave-one-root-out action selection."""
import argparse
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pandas as pd
from report_platform_controls import flight
from run_platform_task_oracle import read
from run_playground_task_reversal import crossings_by_section


def report(out):
    plan=read(out/'plan.json')
    base=Path(plan['base']); rows=[];prefixes={}
    for folder in [base]+sorted(out.glob('part_*')):
        for skill in plan['skills']:
            case=folder/skill
            model=mujoco.MjModel.from_binary_path(str(case/'terrain.mjb'))
            terrain=read(case/'resolved_config.json')['ditch_terrain'];edge=terrain['front_x_m']
            for r in read(case/'status.json')['results']:
                v=r['variant']; root=v['bank_row']
                if root not in plan['roots']:continue
                a=v.get('approach_alpha',0.);speed=v['approach_speed_m_s']
                path=case/r['case_dir']/'trajectory.npz'
                with np.load(path) as z:
                    q=z['qpos'];vel=z['qvel'];tr=z['trace'];hand=r['handoff_s'];entry=r['rb_entry_s']
                    vx=forward=hy=hv=heading=np.nan
                    airborne=loaded=verified=False
                    if entry is not None:
                        k=round(entry/.005);yaw=tr[k,6]
                        vx=vel[k,0];forward=vx*np.cos(yaw)+vel[k,1]*np.sin(yaw)
                        end=len(q) if hand is None else round(hand/.005)+2
                        loads=z['contact_substep'][:end];flags=z['wheel_contact_flags'][:end,2]
                        key=hashlib.sha256(q[:end].tobytes()+loads.tobytes()+flags.tobytes()).digest()
                        if key not in prefixes:
                            support=flags&(loads[:,-1,:2]>1e-5).all(1)
                            take,_,land=flight(model,q[:end],loads,support,tr[:end,0],edge,edge,
                                               terrain['far_bank_height_m'],entry)
                            prefixes[key]=take,land
                        take,land=prefixes[key]
                        airborne=take is not None;loaded=land is not None
                        verified=loaded and hand is not None and land<=hand+1e-5
                    if hand is not None:
                        k=round(hand/.005)+1;hy=q[k,1];hv=vel[k,0];heading=tr[k,6]
                    result=dict(root=root,skill=skill,approach=r['route'],dep=v['departure'],alpha=a,speed=speed,
                        entry_vx=vx,entry_forward=forward,hand_y=hy,hand_vx=hv,hand_heading=heading,
                        airborne=airborne,loaded=loaded,verified=verified,goal=r['goal'],
                        termination=r['termination'],trajectory=str(path))
                    for label,_,_,y in crossings_by_section(tr,np.inf if hand is None else hand,[12.7,13.6,15.]):
                        for side in (-1,1):
                            result[f'{label}_{side}']=bool(verified and r['goal'] and not r['forbidden_contact_steps'] and y is not None and .06<=side*y<=3.9)
                    rows.append(result)
    df=pd.DataFrame(rows);df.to_csv(out/'physical.csv',index=False)
    tests=[f'{sec}_{side}' for sec in ('near','mid','far') for side in (-1,1)]
    prefix=df[df.dep=='straight']
    prefix.groupby(['skill','speed','alpha','approach']).agg(roots=('root','size'),entered=('entry_vx','count'),
        airborne=('airborne','sum'),loaded=('loaded','sum'),entry_vx_p10=('entry_vx',lambda a:a.quantile(.1)),
        entry_vx_p90=('entry_vx',lambda a:a.quantile(.9)),entry_vx_mean=('entry_vx','mean'),entry_forward_mean=('entry_forward','mean'),verified=('verified','sum')).to_csv(out/'prefix_coverage.csv')
    keys=['root','skill','approach','dep','speed']
    matched=df[df.alpha!=0].merge(df[df.alpha==0],on=keys,suffixes=('','_0'),validate='many_to_one')
    matched['match_vx']=(matched.entry_vx-matched.entry_vx_0).abs()<=.05
    matched['match_forward']=(matched.entry_forward-matched.entry_forward_0).abs()<=.05
    matched['delta_hand_y']=matched.hand_y-matched.hand_y_0
    matched['delta_yaw_deg']=np.rad2deg(np.arctan2(np.sin(matched.hand_heading-matched.hand_heading_0),np.cos(matched.hand_heading-matched.hand_heading_0)))
    # A contract is new only if every alpha-zero action at every speed failed for that same root/task.
    fixed=df[df.alpha==0].groupby(['skill','root'])[tests].any()
    support=matched[matched.match_vx].groupby(['skill','root'])[tests].any().reindex_like(fixed).fillna(False)
    pd.DataFrame([dict(skill=skill,task=task,contracts=len(group),base_feasible=int(group[task].sum()),
                  new_matched=int((~group[task]&support.loc[skill,task].to_numpy()).sum()))
                  for skill,group in fixed.groupby(level='skill') for task in tests]).to_csv(out/'matched_contracts.csv',index=False)
    matched['gain']=matched.verified&~matched.verified_0
    matched['loss']=~matched.verified&matched.verified_0
    stats=[]
    for (skill,a,speed,approach),g in matched.groupby(['skill','alpha','speed','approach']):
        one=g[g.dep=='straight'];pair=one[one.match_vx];routes=g[g.match_vx]
        for task in tests:
            stats.append(dict(skill=skill,alpha=a,speed=speed,approach=approach,task=task,all_roots=len(one),
              both_enter=int((one.entry_vx.notna()&one.entry_vx_0.notna()).sum()),matched=len(pair),
              matched_forward=int(one.match_forward.sum()),added_handbacks=int(pair.gain.sum()),
              lost_handbacks=int(pair.loss.sum()),both_handed=int(pair.delta_hand_y.count()),
              median_delta_yaw_deg=pair.delta_yaw_deg.median() if pair.delta_yaw_deg.notna().any() else np.nan,
              median_delta_y_m=pair.delta_hand_y.median() if pair.delta_hand_y.notna().any() else np.nan,
              matched_routes=len(routes),added_completions=int((routes[task]&~routes[task+'_0']).sum()),
              lost_completions=int((~routes[task]&routes[task+'_0']).sum())))
    pd.DataFrame(stats).to_csv(out/'matched_speed.csv',index=False)
    selections=[]
    for skill,g in df.groupby('skill'):
        for mode,subset in [('fixed0',g[(g.alpha==0)&(g.speed==2)]),('fixed0.5',g[(g.alpha==.5)&(g.speed==2)]),
                            ('fixed1',g[(g.alpha==1)&(g.speed==2)]),('speed_only',g[g.alpha==0]),('joint',g)]:
            for task in tests:
                matrix=subset.pivot(index='root',columns=['speed','alpha','approach','dep'],values=task)
                # Stable tie break: prefer the reference speed, then smaller alpha and route/dep index.
                matrix=matrix[sorted(matrix.columns,key=lambda v:(abs(v[0]-2.),v[1],v[2],v[3]))]
                values=matrix.to_numpy(dtype=int);actions=(values.sum(0)-values).argmax(1)
                for i,root in enumerate(matrix.index):
                    sp,a,r,dep=matrix.columns[actions[i]]
                    selections.append(dict(skill=skill,mode=mode,task=task,root=root,speed=sp,alpha=a,approach=r,dep=dep,
                                           feasible=values[i].max(),chosen=values[i,actions[i]]))
    pd.DataFrame(selections).to_csv(out/'selection.csv',index=False)
    fig,axes=plt.subplots(2,3,figsize=(12,5),layout='constrained')
    for (skill,r),g in df[(df.root==plan['roots'][0])&(df.speed==2)&(df.dep=='straight')].groupby(['skill','approach']):
        ax=axes[int(skill=='rb_run1'),r]
        for row in g.sort_values('alpha').itertuples():
            with np.load(row.trajectory) as z:
                tr=z['trace'];xy=z['published_reference_xy']
            line,=ax.plot(tr[:,1],tr[:,2],label=f'alpha {row.alpha:g}')
            ax.plot(*xy.T,':',color=line.get_color(),alpha=.45);ax.scatter(*tr[-1,1:3],s=8,color=line.get_color())
        ax.set(title=f'{skill}, approach {r}',xlabel='world x (m)',ylabel='world y (m)',aspect='equal')
    axes[0,0].legend(fontsize=7);fig.savefig(out/'xy.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    report(p.parse_args().out)
