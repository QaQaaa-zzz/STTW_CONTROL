"""Calibrate actual planar geometry-support errors on existing DEV trajectories.

This is a bounding-box support metric for fixed axis-aligned maps, not a surface
Hausdorff distance or a probability guarantee. No policy or model is refitted.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import mujoco
from sttw_control.playground_arx import states,robot_radius_bound,geometry_error_bound
from sttw_control.playground_prediction import NominalRollout,BodyEnvelope
from sttw_control.playground_planner import geometric_prediction
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--horizons',nargs='+',type=float,default=[1.,2.,3.,3.8]);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
steps_list=[round(h/.02) for h in a.horizons];maximum_steps=max(steps_list)
manifest=json.loads((a.collection/'manifest.json').read_text());entries=[e for e in manifest['entries'] if e['split'] in ['DEV_ROOT','DEV_MAP']];protocol=dict(role='Calibration DEV only, no TEST, refit, policy training or new physical episodes',entries=[e['name'] for e in entries],anchor_s=.02,horizons_s=a.horizons,metric='Maximum over time, every collision geom, lower/upper x/y support discrepancy. Inflating each predicted planar AABB by this observed error encloses actual AABBs. Applicable to current axis-aligned road/forbidden rectangles only, not general 3D geometry or probabilistic safety.',margin_rule='For each method/horizon take maximum across DEV group P95 support error, plus .02m declared map/discretization allowance. Keep prior global3D bound for comparison; do not select by E1 success.',budget=f'One saved-data evaluation with {maximum_steps} free prediction steps each, no sweep or retries',scope='Reset+20ms anchors only; no claim of rolling-state/history coverage. Partial/missing windows are reported, not padded.')
(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2));bank=np.load(a.collection/'routes.npz');tensor=lambda x:torch.as_tensor(x,device='cuda',dtype=torch.float32);s=tensor(bank['s']);records=[];coverage=[];started=time.perf_counter()
first=Path(entries[0]['output']);cfg=json.loads((first/'resolved_config.json').read_text());follower=json.loads((Path(cfg['ground_handoff']['bundle'])/'config.json').read_text())['follower'];follower.update(cfg['ground_handoff'].get('follower_overrides',{}));plant=mujoco.MjModel.from_binary_path(str(first/'terrain.mjb'));envelope=BodyEnvelope(plant);radius,jr=robot_radius_bound(plant);predictor=NominalRollout(dict(np.load(a.collection/'model/nominal_arx.npz')),follower);predictor.compile_step()
examples={}
for entry in entries:
 root=Path(entry['output']);rows=json.loads((root/'status.json').read_text())['results'];schema=json.loads((root/'model_state_schema.json').read_text());actual=[];xis=[];previous=[];selected=[];routes=[]
 for row in rows:
  d=np.load(root/row['case_dir']/'trajectory.npz');q,v=states(d['qpos'][::4],d['qvel'][::4],schema)
  if len(q)<maximum_steps+2:continue
  actual.append(q[1:maximum_steps+2]);xis.append(v[1]);previous.append(v[0]);selected.append(row);routes.append(row['route'])
 coverage.append(dict(group=entry['name'],declared=len(rows),full_declared_horizon=len(selected),missing=len(rows)-len(selected)))
 actual=np.array(actual);pose=tensor(actual[:,0]);xi=tensor(np.array(xis));prev=tensor(np.array(previous));xy=tensor(bank['xy'][routes]);h=tensor(bank['heading'][routes]);k=tensor(bank['curvature'][routes]);truth_low,truth_high=envelope(tensor(actual))
 for method in ['geometry','dynamic']:
  prediction,_=geometric_prediction(pose,xi,s,xy,h,k,maximum_steps,.27903) if method=='geometry' else predictor(pose,xi,prev,s,xy,h,k,maximum_steps)
  low,high=envelope(prediction);errors=torch.maximum((low-truth_low).abs(),(high-truth_high).abs()).amax((2,3));errors=errors.cpu().numpy();pred=prediction.cpu().numpy();old=geometry_error_bound(pred,actual,radius,jr)
  for steps in steps_list:
   for i,row in enumerate(selected):records.append(dict(method=method,group=entry['name'],split=entry['split'],root=row['variant']['root_id'],route=row['route'],horizon_s=round(steps*.02,8),max_planar_support_error_m=float(errors[i,:steps+1].max()),global_3d_bound_m=float(old[i,:steps+1].max()),max_xy_error_m=float(np.linalg.norm(pred[i,:steps+1,:2]-actual[i,:steps+1,:2],axis=1).max())))
  examples[entry['name']+'_'+method+'_predicted']=pred[:8];examples[entry['name']+'_actual']=actual[:8]
summary=[]
for method in ['geometry','dynamic']:
 for h in a.horizons:
  for entry in entries:
   z=[x for x in records if x['method']==method and x['horizon_s']==h and x['group']==entry['name']];error=[x['max_planar_support_error_m'] for x in z];summary.append(dict(method=method,horizon_s=h,group=entry['name'],trajectories=len(z),roots=len(set(x['root'] for x in z)),p95_support_error_m=float(np.quantile(error,.95)),max_support_error_m=max(error),p95_global3d_bound_m=float(np.quantile([x['global_3d_bound_m'] for x in z],.95))))
margin=[dict(method=m,horizon_s=h,margin_m=max(x['p95_support_error_m'] for x in summary if x['method']==m and x['horizon_s']==h)+.02) for m in ['geometry','dynamic'] for h in a.horizons]
for name,data in [('records',records),('summary',summary),('coverage',coverage),('margins',margin)]: (a.output/(name+'.json')).write_text(json.dumps(data,indent=2))
np.savez_compressed(a.output/'examples.npz',**examples);(a.output/'timing.json').write_text(json.dumps(dict(elapsed_s=time.perf_counter()-started),indent=2));print(json.dumps(margin,indent=2))
