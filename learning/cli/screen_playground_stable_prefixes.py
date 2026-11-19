"""Offline on the declared wide-gap snapshot, screen distinct approaches for stable handback."""
import argparse,csv,itertools,json,subprocess
from pathlib import Path
import numpy as np
from run_playground_wide_gap_oracle import run,save
p=argparse.ArgumentParser(description=__doc__);p.add_argument('anchor_config',type=Path);p.add_argument('out',type=Path);p.add_argument('--grid',choices=['bend','route_speed'],default='bend');p.add_argument('--lock',type=Path,required=True)
a=p.parse_args();o=a.out.resolve();o.mkdir();(o/'gpu.lock').symlink_to(a.lock.resolve());base=json.loads(a.anchor_config.read_text())
grid=list(itertools.product([.96,.98,1.,1.02,1.04,1.06],[-2.01,-1.99,-1.97,-1.95,-1.93,-1.91,-1.89]))
if a.grid=='route_speed':
 grid=[(1.,x) for _ in range(6) for x in (-2.025,-1.985,-1.965,-1.935,-1.915,-1.875,-1.835)]
save(o/'plan.json',dict(source=str(a.anchor_config.resolve()),role='Known map / Phase 2 candidate discovery only',grid=a.grid,budget='One 84-lane batch, 42 commands per root, a 10 s cutoff and frozen Source. Lane speeds and routes are read from the declared anchor.',plans=grid,protocol='Retain lane-specific route and validated speed from anchor; change trigger, with lateral bend scaling only in bend mode.'))
try:
 h=base['ground_handoff'];n=len(h['handoff_variants']);assert n==2*len(grid)
 plans=grid*2
 ref=dict(np.load(base['oracle_branch_reference']));arc=ref['s']
 for i,(scale,offset) in enumerate(plans):
  if scale==1.:continue
  line=ref['xy'][i].copy();line[:,1]*=scale;distance=np.r_[0,np.cumsum(np.hypot(*np.diff(line,axis=0).T))]
  line=np.stack([np.interp(arc,distance,line[:,axis]) for axis in (0,1)],axis=1)
  line[:,0]+=np.maximum(arc-distance[-1],0.)
  dx,dy=np.gradient(line,arc,axis=0).T;heading=np.unwrap(np.arctan2(dy,dx))
  ref['xy'][i]=line;ref['heading'][i]=heading;ref['curvature'][i]=np.gradient(heading,arc)
 refpath=str(o/'approach_routes.npz');np.savez_compressed(refpath,**ref)
 cfg=dict(base,oracle_branch_reference=refpath,snapshot_loaded_far=True,ground_handoff=dict(h,reference_path=refpath,handoff_variants=[dict(v,activation_offset_m=q[1]) for v,q in zip(h['handoff_variants'],plans)]))
 case=o/'screen';rows=run(cfg,case,'source',10,o);j=json.loads((case/'jump_validation.json').read_text())['cases']
 records=[]
 for i,(r,audit) in enumerate(zip(rows,j,strict=True)):
  assert r['case_dir']==audit['case']
  times=[v for v in [audit['solid_bank_landing_s'],audit['both_far_after_flight_s']] if v is not None]
  hand=r['handoff_s'];loaded=min(times) if times else None
  records.append(dict(root=r['variant']['root_variant'],lane=i,speed_m_s=base['oracle_initial_speed_m_s'][i],bend_scale=plans[i][0],trigger_offset_m=plans[i][1],verified_landing_s=loaded,handoff_s=hand,eligible=bool(times and hand is not None and loaded<=hand+1e-5),termination=r['termination'],source=str(case/r['case_dir']/'trajectory.npz')))
 with (o/'prefixes.csv').open('w') as f:
  w=csv.DictWriter(f,records[0],lineterminator='\n');w.writeheader();w.writerows(records)
 msg='宽沟筛查 84 条，审计飞越后承重且交回 '+str(sum(r['eligible'] for r in records))+' 条。仍须同根去重并检验左右任务。'
except Exception as e:
 msg=f'合格前缀筛查出错：{e}';raise
finally:
 save(o/'terminal.json',dict(status=msg))
 subprocess.run(['/usr/bin/python3','learning/scripts/notify_chapter3_job.py','--title','第三章双稳定交回前缀筛查','--message',msg,'--receipt',str(o/'bark.json')])
