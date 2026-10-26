"""Reproduce paired control decisions from completed, raw-audited results."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def review(root,output):
    root=Path(root).resolve();output=Path(output)
    for file in ['status.json','analysis/status.json']:
        if json.loads((root/file).read_text())['phase']!='completed':
            raise ValueError('completed execution and raw audit required')
    plan=json.loads((root/'plan.json').read_text())
    rows=json.loads((root/'analysis/results.json').read_text())
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('duplicate trajectory')
    roles=plan['evaluation_roles'];radii=[.1,.2,.3,.5]
    disturbed=[r for r in rows if r['force_N']]
    def arrived(row,radius):
        distance=row['minimum_upright_goal_distance_m']
        if distance is not None and distance<=radius:return True
        return None if row['arrival_outcome']=='unknown' else False
    for row in rows:
        if arrived(row,plan['goal_radius_m'])!=row['goal_arrived']:
            raise ValueError('main arrival disagrees with audited minimum distance')
    def counts(group):
        return {role:dict(total=len(rr),survived=sum(r['physical_outcome']=='survived' for r in rr),
            unknown=sum(r['physical_outcome']=='unknown' for r in rr),
            arrivals={str(radius):sum(arrived(r,radius) is True for r in rr) for radius in radii},
            arrival_unknown={str(radius):sum(arrived(r,radius) is None for r in rr) for radius in radii},
            mean_original_alpha_cost=float(np.mean([r['scored_cost'] for r in rr])))
            for role in roles if (rr:=[r for r in group if r['role']==role])}
    groups={}
    for row in disturbed:
        key=(row['context'],row['alpha'],row['event_id'],row['force_N'],row['pulse_onset_task_tick'],
             row['pulse_duration_s'],tuple(row['world_direction']),row['requested_speed_m_s'])
        cell=groups.setdefault(key,{})
        if row['role'] in cell:raise ValueError('duplicate paired role')
        cell[row['role']]=row
    if any(set(cell)!=set(roles) for cell in groups.values()):raise ValueError('complete matched controls required')
    pairs={}
    for target in ['feedback_nominal','feedback_010','feedback_shared','feedback_005','parent_feedback_010']:
        for baseline in ['free_GAE','parent_causal','feedback_010','feedback_nominal']:
            if target==baseline or target not in roles or baseline not in roles:continue
            result=dict(physical_gains=[],physical_losses=[],physical_unknown=0,radii={})
            cost=[]
            for cell in groups.values():
                a,b=cell[target],cell[baseline];identity=a['case_id'].rsplit('/',1)[0]
                if 'unknown' in [a['physical_outcome'],b['physical_outcome']]:result['physical_unknown']+=1
                elif a['physical_outcome']!=b['physical_outcome']:
                    result['physical_gains' if a['physical_outcome']=='survived' else 'physical_losses'].append(identity)
                if a['physical_outcome']==b['physical_outcome']=='survived':cost.append(a['scored_cost']-b['scored_cost'])
            for radius in radii:
                gains=[];losses=[];unknown=0
                for cell in groups.values():
                    a,b=cell[target],cell[baseline];x,y=arrived(a,radius),arrived(b,radius)
                    if x is None or y is None:unknown+=1
                    elif x!=y:(gains if x else losses).append(a['case_id'].rsplit('/',1)[0])
                result['radii'][str(radius)]=dict(gains=gains,losses=losses,unknown=unknown)
            result.update(both_survived=len(cost),mean_both_survived_cost_delta=float(np.mean(cost)) if cost else None,
                both_survived_cost_lower=sum(x<0 for x in cost))
            pairs[target+'_vs_'+baseline]=result
    stratified={}
    for field in ['bank_index','nominal_profile_context','alpha','pulse_duration_s']:
        values=sorted({r[field] for r in disturbed if field in r},key=str)
        stratified[field]={str(value):counts([r for r in disturbed if r.get(field)==value]) for value in values}
    digest=lambda file:hashlib.sha256(Path(file).read_bytes()).hexdigest()
    result=dict(nominal=counts([r for r in rows if not r['force_N']]),disturbed=counts(disturbed),
        paired=pairs,stratified=stratified,source=dict(plan_sha256=digest(root/'plan.json'),
        results_sha256=digest(root/'analysis/results.json'),script_sha256=digest(__file__)),
        physics_steps=0,optimizer_updates=0,
        scope='Derived recomputation of raw-authenticated DEV results; no new raw replay, fitting, independent seeds or significance claim. Counts at different radii and starts remain correlated.')
    if output.exists():raise ValueError('keep prior derived audit immutable')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=review(args.run,args.output)
    print(json.dumps(dict(nominal=result['nominal'],disturbed=result['disturbed']),indent=2))
