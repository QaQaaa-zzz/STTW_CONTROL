"""Recount archived reference processes against the current arrival contract."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def review(root,stage,output):
    root=Path(root).resolve();output=Path(output)
    if output.exists():raise FileExistsError(output)
    if json.loads((root/'status.json').read_text())['phase']!='completed':raise ValueError('complete archive required')
    plan=json.loads((root/'plan.json').read_text());path=Path(plan['path_file'])
    if digest(path)!=plan['path_sha256']:raise ValueError('original path changed')
    reuse=root/'reuse_cases.json'
    if digest(reuse)!=plan['sealed_files'][str(reuse)]:raise ValueError('cache receipt changed')
    with np.load(path,allow_pickle=False) as z:
        goal=np.array([np.interp(float(z['goal']),z['s'],z['xy'][:,i]) for i in range(2)])
    cases=json.loads((root/stage/'results.json').read_text())
    index=json.loads((root/stage/'case_index.json').read_text());byid={r['case_id']:r for r in cases}
    if len(byid)!=len(cases) or {r['case_id'] for r in index}!=set(byid):raise ValueError('case/index mismatch')
    sources=[(byid[i['case_id']],root/i['file'],i['sha256'],False) for i in index]
    sources += [(r['case'],Path(r['index']['file']),r['index']['sha256'],True) for r in json.loads(reuse.read_text())]
    if len({r[0]['case_id'] for r in sources})!=len(sources):raise ValueError('duplicate new/cache case')
    rows=[]
    for c,file,sha,cached in sources:
        if digest(file)!=sha:raise ValueError('raw changed: '+str(file))
        with np.load(file,allow_pickle=False) as z:
            active=z['active_tick'].astype(bool)
            if not np.array_equal(z['physical_tick'][active],np.arange(c['physical_ticks'])):raise ValueError('clock mismatch')
            failure=z['contact_failure']|z['nonfinite']|z['lower_fault']|z['policy_fault']
            upright=active&(np.cumsum(failure)==0)
            distance=np.linalg.norm(z['actual_xy']-goal,axis=1)
            if c['physical_outcome']=='survived' and (active.sum()!=2000 or np.any(failure[active])):
                raise ValueError('invalid complete survival label')
            minimum=float(np.min(distance[upright])) if np.any(upright) else None
            row=dict(case_id=c['case_id'],role=c['role'],alpha=c['alpha'],event_id=c['event_id'],
                force_N=c['force_N'],physical_outcome=c['physical_outcome'],cached=cached,
                minimum_upright_goal_distance_m=minimum,raw_file=str(file),raw_sha256=sha)
            for radius in [.1,.2,.3,.5]:row['arrived_'+str(radius)]=minimum is not None and minimum<=radius
            rows.append(row)
    summary=dict(source_run=str(root),stage=stage,plan_sha256=digest(root/'plan.json'),script_sha256=digest(__file__),
        authenticated_logical_cases=len(rows),new_raw_cases=sum(not r['cached'] for r in rows),
        cached_raw_cases=sum(r['cached'] for r in rows),physics_steps=0,optimizer_updates=0,groups=[])
    keys=sorted({(r['role'],str(r['alpha']),bool(r['force_N'])) for r in rows})
    for role,alpha,disturbed in keys:
        group=[r for r in rows if (r['role'],str(r['alpha']),bool(r['force_N']))==(role,alpha,disturbed)]
        summary['groups'].append(dict(role=role,alpha=alpha,disturbed=disturbed,total=len(group),
            survived=sum(r['physical_outcome']=='survived' for r in group),
            arrivals={str(v):sum(r['arrived_'+str(v)] for r in group) for v in [.1,.2,.3,.5]}))
    summary['limits']='Retrospective same-goal 10s DEV reclassification; strict labels unchanged. '
    summary['limits']+='Related alpha branches and reused traces are not independent starts or additional physics. '
    summary['limits']+='Do not pool this archive with other panels or use eventwise winners as a deployed method.'
    output.mkdir()
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    with (output/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=review(a.run,a.stage,a.output)
    print(json.dumps({k:result[k] for k in ['authenticated_logical_cases','new_raw_cases','cached_raw_cases','physics_steps']},indent=2))
