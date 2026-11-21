"""Physical-branch audit and matched C/C-R results from the online planner rollout."""
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
    records=[]
    for case in sorted(out.glob('*/part_*/*/status.json')):
        folder=case.parent;layout=folder.parent.parent.name;skill=folder.name
        model=mujoco.MjModel.from_binary_path(str(folder/'terrain.mjb'))
        terrain=read(folder/'resolved_config.json')['ditch_terrain']
        edge=terrain['front_x_m']
        flight_cache={}
        for row in read(case)['results']:
            variant=row['variant'];hand=row['handoff_s']
            path=folder/row['case_dir']/'trajectory.npz'
            with np.load(path) as z:
                trace=z['trace'];q=z['qpos'];v=z['qvel']
                end=len(q) if hand is None else round(hand/.005)+2
                digest=hashlib.sha256(q[:end].tobytes()+v[:end].tobytes()).hexdigest()
                verified=False
                if hand is not None:
                    # Include physical contact evidence in the replay key before reusing a verdict.
                    loads=z['contact_substep'][:end];flags=z['wheel_contact_flags'][:end,2]
                    key=(digest,hashlib.sha256(loads.tobytes()+flags.tobytes()).digest())
                    if key not in flight_cache:
                        supported=flags&(loads[:,-1,:2]>1e-5).all(1)
                        _,_,land=flight(model,q[:end],loads,supported,trace[:end,0],edge,edge,
                                        terrain['far_bank_height_m'],row['rb_entry_s'])
                        flight_cache[key]=land is not None and land<=hand+1.e-5
                    verified=flight_cache[key]
                for section,_,_,y in crossings_by_section(trace,np.inf if hand is None else hand,plan['section_x']):
                    for side in (-1,1):
                        if variant['planner']!='oracle' and (section!=variant['bin'] or side!=variant['side']):continue
                        success=bool(verified and row['goal'] and not row['forbidden_contact_steps']
                                     and y is not None and .06<=side*y<=3.9)
                        records.append(dict(layout=layout,skill=skill,root=variant['bank_row'],
                            planner=variant.get('arm',variant['planner']),approach_alpha=variant.get('approach_alpha',0.),section=section,side=side,force=variant['force_y_N'],
                            approach=row['route'],planned_dep=variant['planned_departure'],chosen_dep=row['selected_departure'],
                            departure_applied_tick=row.get('departure_applied_tick'),verified=verified,success=success,collision=bool(row['forbidden_contact_steps']),
                            termination=row['termination'],force_steps=row['applied_force_ticks'],force_onset=row['air_event_tick'],
                            handoff_s=hand,query_batch_ms=row['query_batch_ms'],query_feasible=row['handoff_query_feasible'],
                            prefix=digest,trajectory=str(path)))
    data=pd.DataFrame(records)
    keys=['layout','skill','root','force','section','side']
    candidates=data[data.planner=='oracle']
    if plan.get('blind'):
        candidates.to_csv(out/'oracle_cases.csv',index=False)
    oracle=candidates.groupby(keys).success.any().rename('oracle_feasible')
    assigned=data[data.planner!='oracle'].join(oracle,on=keys)
    # A separately packed physical run can diverge before RB entry.
    # Only the exhaustive candidates executed in this batch can supply its oracle.

    assigned.to_csv(out/'online_cases.csv',index=False)
    summary=assigned.groupby(['layout','skill','force','section','planner']).agg(tasks=('root','size'),
               locked_true_jump=('verified','sum'),oracle_feasible=('oracle_feasible',lambda x:x.sum(min_count=1)),
               chosen_completed=('success','sum'),collision=('collision','sum'),issued_full_pulse=('force_steps',lambda x:(x==20).sum()))
    summary.to_csv(out/'online_funnel.csv')
    first,second=plan.get('pair',['C','C-R'])
    c=assigned[assigned.planner==first].set_index(keys)
    cr=assigned[assigned.planner==second].set_index(keys)
    pairs=pd.DataFrame(dict(same_prefix=(c.prefix==cr.prefix),rescued=~c.success&cr.success,
                            lost=c.success&~cr.success,changed=c.chosen_dep!=cr.chosen_dep,
                            c_wall=c.collision,cr_wall=cr.collision,clock_ms=cr.query_batch_ms,
                            c_trajectory=c.trajectory,cr_trajectory=cr.trajectory))
    pairs.to_csv(out/'matched_cases.csv')
    pairs['net']=pairs.rescued.astype(int)-pairs.lost.astype(int)
    pairs[pairs.same_prefix].groupby(['layout','skill','force','section']).agg(
        tasks=('net','size'),rescued=('rescued','sum'),lost=('lost','sum'),net=('net','sum'),
        changed=('changed','sum'),c_wall=('c_wall','sum'),cr_wall=('cr_wall','sum')).to_csv(out/'matched_summary.csv')
    per_task=pairs[pairs.same_prefix].groupby(['layout','root','section','side']).net.sum()
    pd.Series(dict(distinct_task_units=len(per_task),net_positive_units=(per_task>0).sum(),
                   net_negative_units=(per_task<0).sum(),unchanged_units=(per_task==0).sum(),
                   mismatched_pairs=(~pairs.same_prefix).sum())).to_json(out/'decision_units.json')
    layouts=sorted(assigned.layout.unique(),reverse=True)
    skills=sorted(assigned.skill.unique(),reverse=True)
    fig,axes=plt.subplots(len(layouts),len(skills),figsize=(5*len(skills),2.5*len(layouts)),squeeze=False,layout='constrained')
    for (layout,skill),group in pairs.reset_index().groupby(['layout','skill']):
        group=group.sort_values(['rescued','lost','changed'],ascending=False)
        r=group.iloc[0];ax=axes[layouts.index(layout),skills.index(skill)]
        for name,color in [('c','#155789'),('cr','#a14324')]:
            with np.load(r[name+'_trajectory']) as z:
                tr=z['trace'];ref=z['published_reference_xy']
            ax.plot(*ref.T,':',color=color,lw=.8);ax.plot(tr[:,1],tr[:,2],color=color,label=first if name=='c' else second)
            ax.scatter(*tr[-1,1:3],s=10,color=color)
        ax.set_title(f"{layout}/{skill}, root {r.root}, {r.force:+} N, {r.section}");ax.set(xlim=(1,19),xlabel='world x (m)',ylabel='world y (m)',aspect='equal');ax.legend()
    fig.savefig(out/'online_xy.png',dpi=140)
    if 'method' in plan:
        write_calibrated_comparison(out,plan)


def write_calibrated_comparison(out,plan):
    parent=out.parent
    archives=[('original_0211',Path(plan['oracle_cases']))]
    archives.extend((m,parent/m/'online_cases.csv') for m in ('kinematic','phase2_relative'))
    tables=[]
    for method,path in archives:
        if not path.exists():continue
        cases=pd.read_csv(path)
        cases=cases[cases.skill=='rb_run1'].copy()
        cases['method']=method
        cases['queried']=cases.query_batch_ms>0
        cases['admitted_goal']=cases.success & cases.query_feasible
        cases['admitted_wall']=cases.collision & cases.query_feasible
        cases['admitted_oracle']=cases.oracle_feasible & cases.query_feasible
        if method!='original_0211':
            cases[['admitted_oracle','oracle_feasible']]=np.nan
        tables.append(cases)
    all_cases=pd.concat(tables)
    keys=['layout','root','force','section','side','planner']
    reference=all_cases[all_cases.method=='original_0211']
    matched=all_cases[all_cases.method!='original_0211'].merge(reference,on=keys,suffixes=('','_old'),validate='many_to_one')
    same=matched[matched.approach==matched.approach_old].copy()
    same['same_prefix']=same.prefix==same.prefix_old
    same.groupby(['method','layout','planner']).agg(same_approach=('root','size'),
        same_physical_prefix=('same_prefix','sum')).to_csv(parent/'cross_batch_prefix.csv')
    metrics=dict(n=('root','size'),complete=('success','sum'),walls=('collision','sum'),queried=('queried','sum'),
        admitted=('query_feasible','sum'),admitted_goal=('admitted_goal','sum'),admitted_wall=('admitted_wall','sum'),
        oracle_feasible=('oracle_feasible',lambda x:x.sum(min_count=1)),admitted_oracle=('admitted_oracle',lambda x:x.sum(min_count=1)))
    g=all_cases.groupby(['method','planner']).agg(**metrics)
    g.to_csv(parent/'pareto.csv')
    all_cases.groupby(['method','planner','side','force','layout']).agg(**metrics).to_csv(parent/'calibrated_sides.csv')
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    for (method,planner),r in g.iterrows():
        ax.scatter(100*r.walls/r.n,100*r.complete/r.n,label=f'{method} {planner}')
    ax.set(xlabel='Physical wall contact (%)',ylabel='Completed assigned contracts (%)')
    ax.legend(fontsize=7,ncol=2);fig.savefig(parent/'safety_completion.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    report(p.parse_args().out)
