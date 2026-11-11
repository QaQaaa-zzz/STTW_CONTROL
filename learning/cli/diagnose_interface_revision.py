"""Audit prediction spread and nominal horizon filtering using saved data."""
import json,numpy as np
from pathlib import Path
from sttw_control.route_horizon import nominal_horizon
import argparse
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base;out=b/'chapter3_review_repairs'
p=b/'chapter3_interface_all_routes/upstream_distance';d=np.load(p/'dataset.npz');pred=np.load(p/'predictions.npz');v=d['dev']&d['hit'];collapse=[]
for name,j in [('x',0),('y',1),('vx',7),('vy',8),('wx',10),('time',38)]:
 y=d['target'][v,j];m=pred['predicted_handoff_mean'][v,j];collapse.append(dict(field=name,pred_std=float(m.std()),observed_std=float(y.std()),ratio=float(m.std()/y.std())))
horizon=[]
for rep in range(3):
 run=b/f'chapter3_interface_online/repeat{rep}';cfg=json.loads((run/'config.json').read_text());roads=np.load(cfg['task_reference_path'])['xy'];pars=np.array(cfg['moving_event_planner']['candidate_parameters']);summary=json.loads((run/'evaluation/summary.json').read_text())['cases']
 for r in json.loads((run/'evaluation/moving_planning.json').read_text())['records']:
  allow,dt=nominal_horizon(np.array(r['query_root'])[None],roads,pars,cfg['ground_handoff']['goal'],r['query_step']*.02,10.)
  k=r['selected'];horizon.append(dict(repeat=rep,lane=r['lane'],method=r['method'],selected=k,speed=float(pars[k,5]),ground=bool(pars[k,-1]<.5),success=summary[r['lane']]['success'],nominal_rejected=not bool(allow[0,k]),nominal_seconds=float(dt[0,k]),remaining_seconds=10-r['query_step']*.02))
result=dict(upstream_variation=collapse,horizon=horizon,successful_ground_rejected=sum(x['ground'] and x['success'] and x['nominal_rejected'] for x in horizon),scope='Retrospective diagnosis; does not predict rerun success; nominal command not speed bound')
(out/'diagnosis.json').write_text(json.dumps(result,indent=2));print('Successful ground episodes rejected by proposed rule:',result['successful_ground_rejected']);print(collapse)
