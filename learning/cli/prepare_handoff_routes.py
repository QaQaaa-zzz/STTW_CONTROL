"""Freeze a same-slot, full-prefix downstream-route intervention diagnostic."""
import argparse,copy,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
p.add_argument('--route-prefix',help='Select all candidates of this upstream route, preserving original root split')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
cfg=json.loads((a.source/'config.json').read_text());protocol=json.loads((a.source/'protocol.json').read_text())
if a.route_prefix:
 selected=[i for i,row in enumerate(protocol['cases']) if row['candidate'].startswith(a.route_prefix)]
 if not selected:raise ValueError('No upstream candidates match route prefix')
 cfg['ground_initial_conditions']=[cfg['ground_initial_conditions'][i] for i in selected]
 for k in ['reference_route_ids','ground_only_lanes','observer_initialization_lanes']:
  if k in cfg['ground_handoff']:cfg['ground_handoff'][k]=[cfg['ground_handoff'][k][i] for i in selected]
 for k in ['methods','forced_candidates','decision_x_m']:
  if isinstance(cfg['moving_event_planner'].get(k),list):cfg['moving_event_planner'][k]=[cfg['moving_event_planner'][k][i] for i in selected]
 protocol['cases']=[dict(protocol['cases'][i],source_lane=i,lane=j) for j,i in enumerate(selected)]
 cfg['num_envs']=len(selected)
for row in protocol['cases']:
 row.setdefault('source_root',row['root']);row['source_role']=row['role']
z=np.load(cfg['ground_handoff']['reference_path']);bank={k:z[k] for k in z.files};s=bank['s'];n=len(bank['xy'])
road=np.load(cfg['task_reference_path']);roads=road['xy'];variants=[('straight',0.),('left',.6),('right',-.6)]
xy,head,curve=[],[],[]
for name,amplitude in variants:
 x=np.linspace(-5.,26.,6201);phase=np.clip((x-12.)/4.,0.,1.);y=amplitude*np.sin(np.pi*phase)**2
 arc=np.r_[0,np.cumsum(np.hypot(np.diff(x),np.diff(y)))];xx=np.interp(s,arc,x);yy=np.interp(s,arc,y)
 heading=np.unwrap(np.arctan2(np.gradient(yy,s),np.gradient(xx,s)))
 xy.append(np.column_stack((xx,yy)));head.append(heading);curve.append(np.gradient(heading,s))
newxy=np.array(xy)
np.savez_compressed(a.output/'issued.npz',s=s,xy=np.concatenate((bank['xy'],newxy)),heading=np.concatenate((bank['heading'],head)),curvature=np.concatenate((bank['curvature'],curve)))
np.savez_compressed(a.output/'roads.npz',s=s,xy=np.concatenate((roads,newxy)))
for index,(name,amplitude) in enumerate(variants):
 out=a.output/name;out.mkdir();c=copy.deepcopy(cfg);c['ground_handoff']['reference_path']=str(a.output/'issued.npz');c['task_reference_path']=str(a.output/'roads.npz');c['ground_handoff']['handoff_route_ids']=[n+index]*c['num_envs'];c['report_directory']=str(out);c['data_role']='Original grouped TRAIN/DEV task-conditioned collection' if a.route_prefix else 'DEV same-slot downstream-route intervention;historical branches;not independent TEST'
 pr=copy.deepcopy(protocol)
 for row in pr['cases']:row['method']=name;row['role']=row['source_role'] if a.route_prefix else 'DEV_DOWNSTREAM_ROUTE'
 pr['route_names']+=['downstream_'+v[0] for v in variants];pr['scope']=c['data_role'];pr['downstream_route_id']=n+index;pr['amplitude_m']=amplitude
 (out/'config.json').write_text(json.dumps(c,indent=2));(out/'protocol.json').write_text(json.dumps(pr,indent=2))
contract=dict(question='Does changing only the commanded posthandoff route change continuation from a matched incoming state?',source=str(a.source),selection=f'All {len(protocol["cases"])} declared branches;upstream route filter={a.route_prefix};no outcome-based selection',variants=dict(variants),geometry='Straight until x12;sin-squared lateral excursion over12..16m returning to original goal;fixed0.6m corridor half-width,goal,speed1.5/alpha.5;same terrain',intervention='At actual handoff only,choose prebuilt route-bank index;no physical state reset;all prehandoff routes/commands unchanged',budget=f'3runs x{cfg["num_envs"]}lanes x10s;{3*cfg["num_envs"]*4000}physical lane steps max;each600s wall cap;0policy training',comparison='Verify all recorded prefixes and handoff roots/times before causal interpretation;report nohandoff and prior-road cases separately',stop='No tuning amplitudes or thresholds after outcomes;retain failures',source_role='Retrospective mechanism diagnostic,not population success or recoverability boundary')
(a.output/'declaration.json').write_text(json.dumps(contract,indent=2));print(a.output)
