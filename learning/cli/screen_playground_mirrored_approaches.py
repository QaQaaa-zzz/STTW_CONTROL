"""Test left/right approach commands under identical forward physics on the frozen map."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
from run_playground_wide_gap_oracle import run, save

p = argparse.ArgumentParser(description=__doc__)
for field in ('discovery','domain','output','gpu_lock'):
    p.add_argument(field,type=Path)
a=p.parse_args()
summary=json.loads((a.domain/'terminal.json').read_text())
if summary['same_root_pairs']>=2:
    raise SystemExit('Existing domain met its target; use those measured returns.')
out=a.output.resolve()
out.mkdir()
(out/'gpu.lock').symlink_to(a.gpu_lock.resolve())
cfg=json.loads((a.discovery/'screen.json').read_text())
hand=cfg['ground_handoff']
reference=dict(np.load(cfg['oracle_branch_reference']))
roots=[(0,-.05,-.03,-.015),(3,-.05,.03,.015),(21950,0.,0.,0.)]
choices=list(itertools.product(roots,(-1,1),(54,59),(-2.01,-1.99,-1.97,-1.95,-1.93,-1.91,-1.89)))
save(out/'plan.json',dict(role='Known-map DEV discrimination; three starts and physically executed left/right approaches.',
    world=cfg['world_map'],roots=roots,angle_units='rad',templates=[54,59],commands='Mirror the geometric y commands only. Same actual robot, controller, terrain and initial snapshot within each root. Same contract scoring sections and departures.',
    original_grid=str(a.discovery),budget='One neutral snapshot and one ten-second 84-lane batch. Stop after this batch.'))
refs={'s':reference['s']}
for key in ('xy','heading','curvature'):
    values=np.stack([reference[key][template] for _,__,template,___ in choices])
    signs=np.asarray([sign for _,sign,__,___ in choices])[:,None]
    if key=='xy':
        values[:,:,1]*=signs
    else:
        values*=signs
    refs[key]=values
route=str(out/'routes.npz')
np.savez_compressed(route,**refs)
variants=[dict(v,root_variant=root,start_y_m=y,start_yaw_rad=yaw,start_roll_rad=roll,
               activation_offset_m=trigger,data_split='DEV')
          for v,((root,y,yaw,roll),sign,template,trigger) in zip(hand['handoff_variants'],choices,strict=True)]
neutral=str(Path(cfg['initial_snapshot_path']).parent.parent/'neutral.npz')
start={k:v for k,v in cfg.items() if k not in ('initial_snapshot_path','oracle_branch_reference','oracle_initial_speed_m_s','oracle_initial_alpha')}
start.update(ground_handoff=dict(hand,reference_path=neutral,handoff_variants=variants),snapshot_loaded_far=False,verify_active_jump=False)
run(start,out/'neutral','source',.005,out)
full=dict(cfg,initial_snapshot_path=str(out/'neutral/source_initial_snapshot.pkl'),oracle_branch_reference=route,
          ground_handoff=dict(hand,handoff_variants=variants,reference_path=route))
rows=run(full,out/'screen','source',10,out)
audits=json.loads((out/'screen/jump_validation.json').read_text())['cases']
records=[]
for i,(r,j,(root,sign,template,trigger)) in enumerate(zip(rows,audits,choices,strict=True)):
    assert r['case_dir']==j['case']
    land=[t for t in (j['solid_bank_landing_s'],j['both_far_after_flight_s']) if t is not None]
    handback=r['handoff_s']
    qualified=bool(land and handback is not None and min(land)<=handback+1e-5)
    y=yaw=None
    if qualified:
        with np.load(out/'screen'/r['case_dir']/'trajectory.npz') as z:
            tr=z['trace'];y,yaw=map(float,tr[np.flatnonzero(tr[:,-1]==2)[0],[2,6]])
    records.append(dict(root=root[0],lane=i,approach='left' if sign==-1 else 'right',template=template,
                   trigger_offset_m=trigger,flight=any(e['solid_bank_rear_launch'] or e['bank_to_bank_flight'] for e in j['events']),
                   loaded_s=min(land) if land else None,handoff_s=handback,qualified=qualified,
                   y_h_m=y,yaw_h_rad=yaw,termination=r['termination']))
with (out/'prefixes.csv').open('w') as f:
    writer=csv.DictWriter(f,records[0],lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
save(out/'terminal.json',dict(scored=len(records),qualified=sum(r['qualified'] for r in records)))
