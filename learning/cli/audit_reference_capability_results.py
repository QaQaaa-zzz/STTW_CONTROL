"""Independent endpoint arithmetic on saved XY/contact clocks, never simulation."""
import argparse
import json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def audit(root,*,confirmation=False,support_permission=False):
    if confirmation and support_permission:raise ValueError('one explicit analysis protocol only')
    root=root.resolve();source=root/('analysis' if support_permission else 'preservation_timecheck_v1/analysis' if confirmation else 'production/analysis')
    if json.loads((source/'status.json').read_text())['phase']!='completed':raise ValueError('complete analysis required')
    file=source/'results.json';rows=json.loads(file.read_text())
    if len(rows)!=(216 if support_permission else 432 if confirmation else 360):raise ValueError('complete declared logical inventory required')
    plan=json.loads((root/'plan.json').read_text());goals={}
    for context,p in plan['contexts'].items():
        if digest(p['path'])!=p['path_sha256']:raise ValueError('original path changed')
        with np.load(p['path']) as path:
            goals[context]=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,j]) for j in range(2)])
    cache={};checked=[];counts={}
    for row in rows:
        key=(row['raw_file'],row['raw_sha256'],row['context'])
        if key not in cache:
            if digest(row['raw_file'])!=row['raw_sha256']:raise ValueError('changed raw')
            with np.load(row['raw_file']) as raw:
                active=np.asarray(raw['active_tick'],bool);index=np.flatnonzero(active)
                if not np.array_equal(index,np.arange(len(index))):raise ValueError('observed samples must be an actual prefix')
                ticks=np.asarray(raw['physical_tick'])[index]
                if not np.array_equal(ticks,np.arange(len(index))) or len(index)>2000:raise ValueError('original clock prefix required')
                xy=np.asarray(raw['actual_xy'])[index]
                contact=np.asarray(raw['contact_failure'],bool)[index]
                fault=(np.asarray(raw['nonfinite'],bool)|raw['lower_fault']|raw['policy_fault'])[index]
                if fault.any() or not np.isfinite(xy).all():raise ValueError('unknown/fault must not become negative evidence')
                first_contact=np.flatnonzero(contact)
                before=xy[:int(first_contact[0])] if len(first_contact) else xy
                distance=np.sqrt(((before-goals[row['context']])**2).sum(axis=1))
                physical='contact' if len(first_contact) else 'survived' if len(index)==2000 else 'unknown'
                if physical=='unknown':raise ValueError('incomplete physical observation remains unknown')
                cache[key]=dict(distance=distance,physical=physical,physical_ticks=len(index),minimum=float(distance.min()) if len(distance) else None)
        q=cache[key]
        if row['physical_outcome']!=q['physical'] or row['physical_ticks']!=q['physical_ticks']:raise ValueError('physical label/cost differs')
        for radius,goal in row['goals'].items():
            hits=np.flatnonzero(q['distance']<=float(radius));arrived=bool(len(hits));first=(int(hits[0])+1)*.005 if arrived else None
            if goal['goal_arrived']!=arrived or goal['first_arrival_s']!=first:raise ValueError('arrival differs from independent XY arithmetic')
            if q['minimum'] is None:
                if goal['minimum_upright_goal_distance_m'] is not None:raise ValueError('empty prefix invented distance')
            elif abs(q['minimum']-goal['minimum_upright_goal_distance_m'])>1e-8:raise ValueError('continuous distance mismatch')
        group=counts.setdefault(row['role'],dict(logical_rows=0,arrived_1m=0,survived=0))
        group['logical_rows']+=1;group['arrived_1m']+=int(row['goals']['1.0']['goal_arrived']);group['survived']+=int(q['physical']=='survived')
        checked.append(dict(case_id=row['case_id'],raw_sha256=row['raw_sha256'],minimum_goal_distance_m=q['minimum']))
    result=dict(logical_rows=len(rows),unique_raw_contexts=len(cache),counts_including_nominal=counts,checked=checked,
        source_results_sha256=digest(file),script_sha256=digest(__file__),new_physics_steps=0,optimizer_updates=0,
        scope='Independent direct saved-XY/contact/clock endpoint arithmetic, no original scorer/goal helper. All declared radii. No reward reconstruction, new samples, precision/replay gate or independent training repeats.')
    atomic_json(source/'independent_endpoint_audit.json',result)
    print(json.dumps({k:result[k] for k in ['logical_rows','unique_raw_contexts','new_physics_steps']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--confirmation',action='store_true');p.add_argument('--support-permission',action='store_true');a=p.parse_args()
    audit(a.root,confirmation=a.confirmation,support_permission=a.support_permission)
