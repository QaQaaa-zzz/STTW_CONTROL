"""Evaluate each planner on the same saved, exhaustive nine-branch physical episode bank."""
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pandas as pd
import torch

from report_platform_controls import flight
from run_platform_online_review import planned_actions
from run_platform_task_oracle import read
from run_playground_task_reversal import crossings_by_section
from sttw_control.platform_online import PlatformDepartureQuery, velocity_state
from sttw_control.playground_history_model import load_history_model


@torch.inference_mode()
def replay(out):
    plan=read(out/'kinematic/plan.json')
    raw=Path(plan['p2']);wide=raw.parent;old=raw/'online'
    reports=[]
    for layout in ('primary','alternate'):
        base=wide/layout/'roots_0008'
        cfg=read(base/'rb_run1/resolved_config.json');hand=cfg['ground_handoff'];sections=dict(zip(('near','mid','far'),plan['section_x']))
        physics=mujoco.MjModel.from_binary_path(str(base/'rb_run1/terrain.mjb'))
        bank=dict(np.load(wide/layout/'departure_bank.npz'));initial=dict(np.load(cfg['approach_initial_bank']))
        ref_index={xy.tobytes():i for i,xy in enumerate(bank['xy'])}
        device=lambda x:torch.as_tensor(np.asarray(x),device='cuda',dtype=torch.float32)
        edge=read(base/'rb_run1/resolved_config.json')['ditch_terrain']['front_x_m']
        archive={};state={};digests={}
        for path in sorted((old/layout).glob('part_*/rb_run1/status.json')):
            for row in read(path)['results']:
                opt=row['variant']
                if opt['planner']!='oracle':continue
                key=(opt['bank_row'],opt['force_y_N'],row['route']);d=row['selected_departure']
                with np.load(path.parent/row['case_dir']/'trajectory.npz') as z:
                    tr=z['trace'];handback=row['handoff_s'];passed=False;ref=0
                    if handback is not None:
                        t=round(handback/.005)+1;q=z['qpos'];v=z['qvel'];loads=z['contact_substep'][:t+1]
                        flags=z['wheel_contact_flags'][:t+1,2];end=t+1
                        digest=hashlib.sha256(q[:end].tobytes()+v[:end].tobytes()).digest()
                        supported=flags&(loads[:,-1,:2]>1e-5).all(1)
                        _,_,land=flight(physics,q[:end],loads,supported,tr[:end,0],edge,edge,.16,row['rb_entry_s'])
                        passed=land is not None and land<=handback+1.e-5
                        if key not in state:
                            state[key]=(q[t],v[t],q[t-4],v[t-4],t)
                            digests[key]=digest
                        if digest!=digests[key]:raise ValueError(f'different physical prefix: {layout} {key}')
                        ref=ref_index[z['published_reference_xy'].tobytes()]
                    crosses=crossings_by_section(tr,np.inf if handback is None else handback,plan['section_x'])
                    tasks={(name,side):bool(passed and row['goal'] and not row['forbidden_contact_steps'] and y is not None and .06<=side*y<=3.9)
                           for name,_,_,y in crosses for side in (-1,1)}
                    archive[key,d]=(ref,tasks,bool(row['forbidden_contact_steps']))
        # Validate the empirical oracle for each root/force/task against the original report.
        model=load_history_model(Path(plan['ground_model']))
        tables=pd.read_csv(raw/f'forecasts_{layout}_rb_run1.csv')
        baseline=read(old/'plan.json')['margin_m']
        sel=read(out/'phase2_relative/plan.json')
        relative=dict(raster=str(raw/(layout+'_terrain.npz')),initializer=str(out/'phase2_relative/initializer.npz'),height=.16)
        for label,method,eps,rel in [('original_0211','kinematic',baseline,None),
                                      ('kinematic','kinematic',plan['margin_m'],None),
                                      ('phase2_relative','phase2_relative',sel['margin_m'],relative)]:
            engine=PlatformDepartureQuery(physics,model,hand['goal'],eps,hand['road_half_width_m'],rel)
            requests=[];pending={}
            tab=tables[(tables.method==method)&(tables.root<78)]
            for root in plan['roots']:
                plans=planned_actions(tab,initial,root,sections,eps)
                for force in plan['forces_N']:
                    for (section,side),actions in plans.items():
                        a,d=actions['C'];key=root,force,a
                        feasible=any(archive[(root,force,i),j][1][section,side] for i in range(3) for j in range(3))
                        rec=dict(layout=layout,method=label,root=root,force=force,section=section,side=side,approach=a,
                                 planned=d,chosen=d,c_vote=False,cr_vote=False,queried=key in state,oracle=feasible)
                        if key in state:pending.setdefault(state[key][4],[]).append(len(requests))
                        requests.append(rec)
            for t,ids in pending.items():
                block=[requests[i] for i in ids];keys=[(r['root'],r['force'],r['approach']) for r in block]
                q,v,pq,pv=[device([state[k][j] for k in keys]) for j in range(4)]
                u=velocity_state(q,v);prev=velocity_state(pq,pv)
                refs=np.array([archive[k,d][0] for k in keys for d in range(3)])
                route=SimpleNamespace(s=device(bank['s']),**{k:device(bank[k][refs]) for k in ('xy','heading','curvature')})
                groups=torch.arange(len(refs),device='cuda').reshape(-1,3)
                chosen,cv,old_vote=engine.choose(q,u[:,0],torch.full_like(u[:,0],2.),route,groups,
                       device([sections[r['section']] for r in block]),device([r['side'] for r in block]),
                       device([r['planned'] for r in block]).long(),10.-t*.005,u,prev)
                for r,(d,cr,c) in zip(block,torch.stack((chosen,cv,old_vote),1).cpu().tolist(),strict=True):
                    r.update(chosen=d,c_vote=bool(c),cr_vote=bool(cr))
            for r in requests:
                key=r['root'],r['force'],r['approach']
                for name,d,vote in [('C',r['planned'],r['c_vote']),('C-R',r['chosen'],r['cr_vote'])]:
                    _,tasks,wall=archive[key,d]
                    reports.append(dict(r,planner=name,executed=d,admitted=vote,success=tasks[r['section'],r['side']],wall=wall))
            pd.DataFrame(reports).to_csv(out/'common_bank_cases.csv',index=False)
    allrows=pd.DataFrame(reports);keys=['layout','root','force','section','side','planner']
    oldcases=pd.read_csv(plan['oracle_cases'])
    pairs=allrows[allrows.method=='original_0211'].merge(oldcases[oldcases.skill=='rb_run1'],on=keys,suffixes=('','_live'),validate='one_to_one')
    parity=dict(n=len(pairs),physical_outcome_diff=int((pairs.success!=pairs.success_live).sum()),
                chosen_diff=int((pairs.executed!=pairs.chosen_dep).sum()),oracle_diff=int((pairs.oracle!=pairs.oracle_feasible).sum()))
    pd.DataFrame([parity]).to_csv(out/'common_bank_parity.csv',index=False)
    allrows['admitted_wall']=allrows.admitted&allrows.wall;allrows['admitted_oracle']=allrows.admitted&allrows.oracle
    allrows.groupby(['method','planner','side']).agg(n=('root','size'),success=('success','sum'),wall=('wall','sum'),
        admitted=('admitted','sum'),admitted_wall=('admitted_wall','sum'),admitted_oracle=('admitted_oracle','sum'),
        oracle=('oracle','sum')).to_csv(out/'common_bank_summary.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    replay(p.parse_args().out)
