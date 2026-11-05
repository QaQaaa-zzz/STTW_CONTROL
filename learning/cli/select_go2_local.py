"""Rank completed nominal DEV candidates from saved trajectories, no simulation."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--evaluation',type=Path,required=True)
p.add_argument('--config',type=Path,required=True)
a=p.parse_args()
s=json.loads(a.config.read_text())
rows=[]
for folder in sorted(a.evaluation.glob('model_*')):
    identity=json.loads((folder/'identity.json').read_text())
    data=np.load(folder/'trajectory.npz')
    cases=[]
    for i,name in enumerate(s['cases']):
        valid=data['valid'][:,i]
        n=int(valid.sum())
        ev=data['speed'][:n,i]-data['reference'][:n,i,0]
        ed=data['measurement'][:n,i,2]-data['reference'][:n,i,1]
        failed=bool(data['failed'][:,i].any())
        duration=float(data['time'][n-1,i]+data['stop_tick'][n-1,i]*s['physics_dt'])
        survived=not failed and n==round(s['seconds']/s['control_dt'])
        tolerance=.01 if name.startswith('post_return_small') else .02
        hold=round(.5/s['control_dt'])
        tracked=survived and bool((np.abs(ev[-hold:])<=.05).all() and (np.abs(ed[-hold:])<=tolerance).all())
        cases.append(dict(case=name,survived=survived,failed=failed,duration_s=duration,
            final_tracking_hold=tracked,speed_rmse=float(np.sqrt(np.mean(ev**2))),
            steer_rmse=float(np.sqrt(np.mean(ed**2))),
            peak_roll=float(np.max(np.abs(data['measurement'][:n,i,0])))))
    score=dict(survived=sum(c['survived'] for c in cases),
               tracking_hold=sum(c['final_tracking_hold'] for c in cases),
               observed_seconds=sum(c['duration_s'] for c in cases),
               normalized_tracking_rmse=float(np.mean([c['speed_rmse']/.05+c['steer_rmse']/.02 for c in cases])))
    row=dict(**identity,run=str(folder.resolve()),cases=cases,score=score)
    rows.append(row)
rows.sort(key=lambda row:(-row['score']['survived'],-row['score']['tracking_hold'],
                         -row['score']['observed_seconds'],row['score']['normalized_tracking_rmse']))
result=dict(role='nominal standard DEV selection, not held-out generalization',
            criterion='lexicographic survival count, final .5s speed/steer tracking count, observation duration, normalized tracking RMSE',
            reward_comparison='not used; old and new training rewards differ',
            best=rows[0],candidates=rows)
(a.evaluation/'selection.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result['best'],indent=2))
