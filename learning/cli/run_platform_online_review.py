"""Paired fixed, nominal, C and C-R plans with a matched nine-action offline oracle."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from run_platform_task_oracle import read
from run_playground_wide_gap_oracle import run, save


def planned_actions(table, bank, root, sections, margin):
    # At the common physical start, use initial attitude and velocities, no test outcomes.
    ids=np.arange(8,78)
    obs=np.column_stack((bank['qpos'][:,1:2],bank['qpos'][:,3:7],bank['qvel'][:,:6]))
    std=obs[ids].std(0).clip(.005)
    distance=((obs[ids]-obs[root])/std)**2
    near=ids[np.argsort(distance.sum(1))[:5]]
    nominal=[]
    for a in range(3):
        states=table[table.approach==a].drop_duplicates('root')
        if states.empty:
            nominal.append(None)
            continue
        x=states[['x','y','speed','qw','qx','qy','qz']].to_numpy()
        nominal.append(int(states.iloc[np.argmin((((x-x.mean(0))/x.std(0).clip(.01))**2).sum(1))].root))
    actions={}
    for label in sections:
        for side in (-1,1):
            gate=(table[label]*side>=.06)&(table[label].abs()<=3.9)
            safe=table.clear>margin
            values=(16*(safe&gate&table.reachable).astype(float)+4*safe+2*gate+table.reachable.astype(float)+.01*table.clear.clip(-1,1))
            grid=table.assign(value=values).pivot(index='root',columns=['approach','departure'],values='value')
            grid=grid.reindex(index=ids,columns=pd.MultiIndex.from_product([range(3),['left','straight','right']])).fillna(-4)
            nom=np.stack([grid.loc[nominal[a]].to_numpy().reshape(3,3)[a] if nominal[a] is not None else np.full(3,-4.) for a in range(3)])
            n=divmod(int(nom.argmax()),3)
            c=divmod(int(grid.loc[near].mean().argmax()),3)
            fixed=(1,0) if side==1 else (0,2)
            actions[label,side]={'fixed':fixed,'nominal':n,'C':c,'C-R':c}
    return actions


def execute(protocol):
    out=protocol.parent;plan=read(protocol);raw=Path(plan['p2']);wide=raw.parent
    method=plan.get('method','kinematic')
    calibrated='method' in plan
    relative={}
    if method=='phase2_relative':
        import torch
        from evaluate_platform_review import fit_hidden
        from sttw_control.playground_history_model import load_history_model
        capture=wide.parent/'replication/primary/rb_run1_capture'
        with torch.inference_mode():
            model=load_history_model(Path(plan['ground_model']))
            mu,scale,weight=fit_hidden(capture,model,read(capture/'model_state_schema.json'))
        np.savez(out/'initializer.npz',mu=mu,scale=scale,weight=weight)
        del model
        torch.cuda.empty_cache()
    for layout in ('primary','alternate'):
        base=wide/layout/'roots_0008'
        for chunk in range(1 if calibrated else 2):
            dest=out/layout/f'part_{chunk}';dest.mkdir(exist_ok=True,parents=True)
            for skill in plan.get('skills',('source','rb_run1')):
                if (dest/skill/'status.json').exists():continue
                cfg=read(base/(skill+'.json'));h=cfg['ground_handoff']
                bank=dict(np.load(cfg['approach_initial_bank']))
                tab=pd.read_csv(raw/f'forecasts_{layout}_{skill}.csv')
                # Same absolute margin and predictor family for both skills.
                tab=tab[(tab.method==method)&(tab.root<78)]
                eps=plan['margin_m'];roots=plan['roots'] if calibrated else plan['roots'][15*chunk:15*chunk+15]
                variants=[]; lanes=[]; directions=[]
                sections=dict(zip(('near','mid','far'),plan['section_x']))
                template=h['handoff_variants'][0]
                for root in roots:
                    plans=planned_actions(tab,bank,root,sections,eps)
                    for force in plan['forces_N']:
                        options=[(name,label,side,a,d) for (label,side),methods in plans.items()
                                 for name,(a,d) in methods.items() if not calibrated or name in ('C','C-R')]
                        if not calibrated:options.extend(('oracle','near',1,a,d) for a in range(3) for d in range(3))
                        for name,label,side,a,d in options:
                            lanes.append([a,'h75',len(variants)])
                            directions.append(d)
                            variants.append(dict(template,bank_row=root,root_variant=root,planner=name,
                                bin=label,side=side,section_x=sections[label],force_y_N=force,planned_departure=d,
                                departure=['left','straight','right'][d]))
                h['handoff_variants']=variants
                cfg['paired_departures']['direction_ids']=directions
                if method=='phase2_relative':
                    relative=dict(relative=dict(raster=str(raw/(layout+'_terrain.npz')),initializer=str(out/'initializer.npz'),
                                                height=read(Path(cfg['world_map']))['terrain']['far_bank_height_m']))
                cfg.update(evaluation_lanes=lanes,ccd_capacity=8*len(lanes),role='paired online controller comparison on reserved rows, DEV',
                           platform_online=dict(margin_m=eps,ground_model=plan['ground_model'],committed_vote=calibrated,**relative))
                run(cfg,dest/skill,skill,10,out)
    save(out/'complete.json',dict(worlds_per_batch=1080,batches=2,rollouts=2160) if calibrated else dict(worlds_per_batch=1485,batches=8,rollouts=11880))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('protocol',type=Path)
    execute(p.parse_args().protocol)
