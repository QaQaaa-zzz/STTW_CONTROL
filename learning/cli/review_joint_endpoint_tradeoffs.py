"""Read-only joint goal/survival audit; keep historical panels separate."""
import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worktree', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); root=a.worktree.resolve(); out=a.output.resolve()
    out.mkdir(exist_ok=False)
    inputs={}
    def read(path):
        inputs[str(path.resolve())]=digest(path)
        return json.loads(path.read_text())
    oldroot=root/'runs/online_survival_endpoint_evaluation_v2'
    old=read(oldroot/'goal_arrival_relaxed_v1/results.json')
    status=read(oldroot/'status.json')
    assert status['phase']=='completed' and status['endpoint_authenticated']
    # Original authenticated four-radius tables contain the actual pre-fatal minimum.
    byid={}
    for x in old:
        byid.setdefault(x['case_id'],[]).append(x)
    assert len(byid)==488
    cases=[]
    for cid, group in byid.items():
        assert len(group)==4 and {x['goal_radius_m'] for x in group}=={.1,.2,.3,.5}
        x=group[0]; distance=x['minimum_upright_goal_distance_m']
        assert len({(z['raw_sha256'],z['minimum_upright_goal_distance_m'],z['physical_outcome']) for z in group})==1
        for z in group:
            if z['goal_arrived'] is not None:
                assert z['goal_arrived']==(distance<=z['goal_radius_m'])
        cases.append(dict(panel=x['panel'],case_id=cid,role=x['role'].replace('parent_speed_floor','floor'),
            alpha=x['alpha'],force_N=x['force_N'],context=str(x['context_index']),event=str(x['force_N']),
            replicate=x['replicate'],distance=distance,physical_outcome=x['physical_outcome'],
            arrival_known=x['arrival_outcome']!='unknown',raw_sha256=x['raw_sha256'],source='old488_derived'))
    legacy=root/'runs/legacy_goal_arrival_review_20261026'
    rec=read(legacy/'receipt.json')
    source=next(x for x in rec['sources'] if x['dataset']=='hidden_time')
    table, planfile=Path(source['table']),Path(source['plan'])
    assert digest(table)==source['table_sha256'] and digest(planfile)==source['plan_sha256']
    plan=read(planfile);inputs[str(table)]=digest(table)
    for name,sha in plan['original_inputs_sha256'].items():
        assert digest(name)==sha
    assert plan['horizon_s']==10 and plan['lower_dt_s']==.005
    with (legacy/'cases.csv').open() as f:
        previous=list(csv.DictReader(f))
    previous={(x['case_id'],float(x['goal_radius_m'])):x for x in previous if x['dataset']=='hidden_time'}
    paths={}
    for name,x in plan['contexts'].items():
        assert digest(x['path'])==x['path_sha256'];inputs[x['path']]=x['path_sha256']
        with np.load(x['path'],allow_pickle=False) as z:
            paths[name]=np.array([np.interp(float(z['goal']),z['s'],z['xy'][:,i]) for i in (0,1)])
    hidden=list(csv.DictReader(table.open()));assert len(hidden)==110
    for x in hidden:
        path=Path(x['raw_file']);assert digest(path)==x['raw_sha256'];inputs[str(path)]=x['raw_sha256']
        with np.load(path,allow_pickle=False) as z:
            active=z['active_tick'].astype(bool);clock=z['physical_tick'][active]
            assert np.array_equal(clock,np.arange(len(clock))) and len(clock)<=2000
            xy=z['actual_xy'][active].astype(float)
            contact=z['contact_failure'][active].astype(bool)
            fault=(z['lower_fault'][active]|z['nonfinite'][active]|z['policy_fault'][active]).astype(bool)
            fault|=~np.isfinite(xy).all(axis=1)
            bad=np.flatnonzero(contact|fault);stop=int(bad[0]) if len(bad) else len(clock)
            distances=np.linalg.norm(xy[:stop]-paths[x['context']],axis=1)
            minimum=float(distances.min()) if len(distances) else None
            observed='contact' if contact.any() else 'survived' if len(clock)==2000 and not fault.any() else 'unknown'
            assert observed==x['physical_outcome']
            known=observed!='unknown'
            for radius in (.1,.2,.3,.5):
                prev=previous[x['case_id'],radius]
                assert np.isclose(minimum,float(prev['minimum_upright_goal_distance_m']),atol=1e-10)
                assert (minimum<=radius)==(prev['goal_arrived']=='True')
        cases.append(dict(panel='hidden_time',case_id=x['case_id'],role=x['role'],
            alpha=float(x['alpha']) if x['alpha'] else None, force_N=float(x['force_N']),context=x['context'],
            event=x['event_id'],replicate=0,distance=minimum,physical_outcome=observed,
            arrival_known=known,raw_sha256=x['raw_sha256'],source='hidden110_raw'))
    assert len(cases)==598 and len({(x['panel'],x['case_id']) for x in cases})==598
    summaries=[]; pairs=[]
    for radius in (.5,.75,1.):
        for x in cases:
            x['arrived']=None if x['distance'] is None or (not x['arrival_known'] and x['distance']>radius) else x['distance']<=radius
        groups=defaultdict(list)
        for x in cases:groups[x['panel'],x['role'],bool(x['force_N'])].append(x)
        for (panel,role,disturbed),xs in sorted(groups.items()):
            summaries.append(dict(panel=panel,role=role,disturbed=disturbed,radius_m=radius,n=len(xs),
                arrived=sum(x['arrived'] is True for x in xs),survived=sum(x['physical_outcome']=='survived' for x in xs),
                arrived_and_survived=sum(x['arrived'] is True and x['physical_outcome']=='survived' for x in xs),
                arrived_and_contact=sum(x['arrived'] is True and x['physical_outcome']=='contact' for x in xs),
                nonarrived_survived=sum(x['arrived'] is False and x['physical_outcome']=='survived' for x in xs),
                nonarrived_contact=sum(x['arrived'] is False and x['physical_outcome']=='contact' for x in xs),
                unknown=sum(x['arrived'] is None or x['physical_outcome']=='unknown' for x in xs)))
        cells=defaultdict(dict)
        for x in cases:
            if x['role']=='zero_middle':continue  # Unique zero episodes are not alpha/noise replicas.
            key=(x['panel'],x['context'],x['alpha'],x['event'],x['replicate'])
            assert x['role'] not in cells[key];cells[key][x['role']]=x
        for key,cell in cells.items():
            target=cell['physical_credit']
            for baseline in ('task_only','floor','parent','causal_eso_allalpha'):
                if baseline not in cell:continue
                base=cell[baseline];assert base['force_N']==target['force_N']
                if target['arrived'] is None or base['arrived'] is None or 'unknown' in (target['physical_outcome'],base['physical_outcome']):
                    da=ds=None;relation='unknown'
                else:
                    da=int(target['arrived'])-int(base['arrived'])
                    ds=int(target['physical_outcome']=='survived')-int(base['physical_outcome']=='survived')
                    relation='equal' if da==ds==0 else 'target_dominates' if da>=0 and ds>=0 else 'baseline_dominates' if da<=0 and ds<=0 else 'tradeoff'
                pairs.append(dict(panel=key[0],context=key[1],alpha=key[2],event=key[3],replicate=key[4],
                    force_N=target['force_N'],radius_m=radius,target='physical_credit',baseline=baseline,
                    arrival_delta=da,survival_delta=ds,relation=relation,target_case_id=target['case_id'],baseline_case_id=base['case_id']))
    paired_summaries=[]
    groups=defaultdict(list)
    for x in pairs:groups[x['panel'],x['baseline'],bool(x['force_N']),x['radius_m']].append(x)
    for (panel,baseline,disturbed,radius),xs in sorted(groups.items()):
        paired_summaries.append(dict(panel=panel,baseline=baseline,disturbed=disturbed,radius_m=radius,n=len(xs),
            **{r:sum(x['relation']==r for x in xs) for r in ('target_dominates','baseline_dominates','tradeoff','equal','unknown')},
            arrival_gain=sum(x['arrival_delta']==1 for x in xs),arrival_loss=sum(x['arrival_delta']==-1 for x in xs),
            survival_gain=sum(x['survival_delta']==1 for x in xs),survival_loss=sum(x['survival_delta']==-1 for x in xs)))
    for x in cases:del x['arrived']
    result=dict(records=cases,summaries=summaries,pairs=pairs,paired_summaries=paired_summaries,
        input_sha256=inputs,analysis_script_sha256=digest(__file__),new_physics_steps=0,optimizer_updates=0,
        limits='Retrospective DEV. Original488 and hidden110 are separate panels, not independent repetitions. Old488 minimum distances reused; hidden110 raw recomputed. No Jalpha reconstruction, no training, no new scalar objective or oracle policy.')
    (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([s for s in summaries if s['radius_m']==1.],indent=2))


if __name__=='__main__':main()
