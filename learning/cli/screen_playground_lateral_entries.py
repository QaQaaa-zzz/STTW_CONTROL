"""Test pre-jump lateral approach offsets on the frozen development terrain."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from run_playground_wide_gap_oracle import run,save

p=argparse.ArgumentParser(description=__doc__)
for name in ('screen','output','gpu_lock'):
    p.add_argument(name,type=Path)
a=p.parse_args()
out=a.output.resolve()
out.mkdir()
(out/'gpu.lock').symlink_to(a.gpu_lock.resolve())
base=json.loads((a.screen/'screen.json').read_text())
h=base['ground_handoff']
bank=dict(np.load(base['oracle_branch_reference']))
roots=[v['root_variant'] for v in h['handoff_variants']]
magnitudes=(.10,.25,.40,.55,.70,.85)
triggers=(-2.01,-1.99,-1.97,-1.95,-1.93,-1.91,-1.89)
save(out/'plan.json',dict(role='Known development map only. Left gate had no completed trajectory; expand lateral entry coverage.',
    base=str(a.screen),physics=base['world_map'],roots=sorted(set(roots)),lateral_shifts_m=magnitudes,triggers=triggers,
    gate='No changes to the two contracts, forbidden geometry, method weights or flight and STTW handback criteria.',
    budget='Two 84-lane batches, one per proven bend profile. 168 new commands. Stop after both. Report all failed prefixes once.',
    profiles=[54,59],shape='world y offset ramps over x 2.5..8.3, flat to x 10.3, returns to the common reference by x 12.0. Quintic smoothstep and arc resampling.'))
records=[]
for profile in (54,59):
    refs={k:v.copy() for k,v in bank.items()}
    commands=[(dy,trigger) for dy in magnitudes for trigger in triggers]*2
    s=refs['s']
    for lane,(dy,_) in enumerate(commands):
        old=bank['xy'][profile];x=old[:,0]
        def step(lo,hi):
            u=np.clip((x-lo)/(hi-lo),0,1)
            return u**3*(10-15*u+6*u*u)
        line=old.copy()
        line[:,1]+=dy*(step(2.5,8.3)-step(10.3,12.0))
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(line,axis=0),axis=1))]
        xy=np.column_stack([np.interp(s,arc,line[:,k]) for k in (0,1)])
        xy[:,0]+=np.maximum(s-arc[-1],0.)
        tangent=np.gradient(xy,s,axis=0)
        heading=np.unwrap(np.arctan2(tangent[:,1],tangent[:,0]))
        refs['xy'][lane],refs['heading'][lane],refs['curvature'][lane]=xy,heading,np.gradient(heading,s)
    ref=str(out/f'profile{profile}.npz')
    np.savez_compressed(ref,**refs)
    variants=[dict(v,activation_offset_m=c[1]) for v,c in zip(h['handoff_variants'],commands,strict=True)]
    cfg=dict(base,oracle_branch_reference=ref,ground_handoff=dict(h,reference_path=ref,handoff_variants=variants),
             snapshot_loaded_far=True)
    case=out/f'profile{profile}'
    res=run(cfg,case,'source',10,out)
    audits=json.loads((case/'jump_validation.json').read_text())['cases']
    for lane,(r,j,(dy,trigger)) in enumerate(zip(res,audits,commands,strict=True)):
        assert r['case_dir']==j['case']
        land=[v for v in (j['solid_bank_landing_s'],j['both_far_after_flight_s']) if v is not None]
        hand=r['handoff_s']
        qualified=bool(land and hand is not None and min(land)<=hand+1e-5)
        y=yaw=None
        if qualified:
            with np.load(case/r['case_dir']/'trajectory.npz') as z:
                tr=z['trace'];k=np.flatnonzero(tr[:,-1]==2)[0];y,yaw=map(float,tr[k,[2,6]])
        records.append(dict(profile=profile,lane=lane,root=roots[lane],approach_shift_m=dy,trigger_offset_m=trigger,
            flight=any(e['solid_bank_rear_launch'] or e['bank_to_bank_flight'] for e in j['events']),
            loaded_s=min(land) if land else None,handoff_s=hand,qualified=qualified,y_h_m=y,yaw_h_rad=yaw,termination=r['termination']))
    with (out/'prefixes.csv').open('w') as f:
        writer=csv.DictWriter(f,records[0],lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
