"""Known-DEV G/I/C/Oracle finite-candidate decisions; no future data in G/I/C selection."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch,mujoco
from sttw_control.playground_prediction import BodyEnvelope
from sttw_control.skill_sequence_rollout import native_to_ground

p=argparse.ArgumentParser(description=__doc__)
for name in ['continuous','independent','oracle-matrix','routes','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
C=json.loads((a.continuous/'cases.json').read_text());I=json.loads((a.independent/'cases.json').read_text());contract=json.loads((a.continuous/'contract.json').read_text());base=Path(contract['oracle'])/'map48'
world=json.loads((base/'world.json').read_text());schema=json.loads((base/'base/model_state_schema.json').read_text());bank=np.load(a.routes);geometry=BodyEnvelope(mujoco.MjModel.from_binary_path(str(base/'base/terrain.mjb')),device='cpu',dimensions=3)
tensor=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float32)
rect=world['terrain']['forbidden_rectangles'];boxes=tensor([[r[0],r[2],0,r[1],r[3],world['terrain'].get('obstacle_height_m',.5)] for r in rect]);bounds=world['bounds_xy_m'];goal=world['goal'];G=[];predicted=[]
t0=time.perf_counter()
for row in C:
 z=np.load(row['prefix']);pose,_=native_to_ground(tensor(z['qpos'][:1]),tensor(z['qvel'][:1]),schema)
 route=row['route'];xy=bank['xy'][route];s=bank['s'];nearest=np.square(xy-z['qpos'][0,:2]).sum(-1).argmin();distance=s[nearest]+np.arange(501)*.02*1.8
 motion=np.repeat(pose.numpy(),501,axis=0);motion[:,:2]=np.stack([np.interp(distance,s,xy[:,j]) for j in range(2)],-1)
 heading=np.interp(distance,s,np.unwrap(bank['heading'][route]));motion[:,3:7]=np.stack([np.cos(heading/2),np.zeros(501),np.zeros(501),np.sin(heading/2)],-1)
 # G assumes exact reference tracking at fixed speed/height, no learned dynamics or aerial motion.
 valid=(motion[:,0]>=goal['x_min_m'])&(motion[:,0]<=goal['x_max_m'])&(abs(motion[:,1])<=goal['abs_y_max_m'])&(1.8*np.cos(heading)>=goal['min_vx_m_s'])
 holds=np.lib.stride_tricks.sliding_window_view(valid,26).all(-1);has=bool(holds.any());stop=int(holds.argmax()+25) if has else 500
 lo,hi=geometry(tensor(motion[:stop+1])[None]);gap=torch.maximum(boxes[None,None,None,:,:3]-hi[:,:,:,None,:],lo[:,:,:,None,:]-boxes[None,None,None,:,3:]).amax(-1).amin();road=torch.stack([lo[...,0]-bounds[0],bounds[1]-hi[...,0],lo[...,1]-bounds[2],bounds[3]-hi[...,1]],-1).amin();clear=float(torch.minimum(gap,road))
 G.append(dict(row,admitted=bool(has and clear>=0),goal_time_s=stop*.02,clearance_m=clear,has_goal=has));predicted.append(motion)
g_time=time.perf_counter()-t0
oracle=list(csv.DictReader(a.oracle_matrix.open()));lookup={(int(r['root']),int(r['upstream']),int(r['downstream'])):r for r in oracle}
O=[dict(r,admitted=r['actual_success'],goal_time_s=float(lookup[r['root'],r['upstream'],r['downstream']]['duration_s'])) for r in C]
methods={'G':G,'I':I,'C':C,'Oracle':O};decisions=[];counts={}
for name,rows in methods.items():
 counts[name]={key:sum(bool(r['admitted'])==ad and r['actual_success']==success for r in rows) for key,ad,success in [('true_admit',True,True),('false_admit',True,False),('false_reject',False,True),('true_reject',False,False)]}
 for root in sorted({r['root'] for r in rows}):
  candidates=[r for r in rows if r['root']==root]
  # The cached same predictor/gates feed both exhaustive joint and full backtracking.
  start=time.perf_counter();valid=[r for r in candidates if r['admitted']];chosen=min(valid,key=lambda r:(r['goal_time_s'],r['route'])) if valid else None;joint_ms=(time.perf_counter()-start)*1000
  start=time.perf_counter();best=None;queries=0
  for up in sorted({r['upstream'] for r in candidates}):
   for r in sorted((r for r in candidates if r['upstream']==up),key=lambda r:r['downstream']):
    queries+=1
    if r['admitted'] and (best is None or (r['goal_time_s'],r['route'])<(best['goal_time_s'],best['route'])):best=r
  backtracking_ms=(time.perf_counter()-start)*1000
  assert (None if chosen is None else chosen['route'])==(None if best is None else best['route'])
  actual=None if chosen is None else lookup[root,chosen['upstream'],chosen['downstream']]
  decisions.append(dict(method=name,root=root,route=None if chosen is None else chosen['route'],no_plan=chosen is None,actual_success=False if chosen is None else chosen['actual_success'],admitted_failure=False if chosen is None else not chosen['actual_success'],actual_duration_s=None if actual is None else float(actual['duration_s']),actual_clearance_m=None if actual is None else float(actual['min_obstacle_distance_m']),failure_stage=None if actual is None else actual['failure_stage'],candidate_queries=queries,joint_selection_ms=joint_ms,backtracking_selection_ms=backtracking_ms,backtracking_same=True))
(a.output/'cases.json').write_text(json.dumps(methods,indent=2)+'\n');(a.output/'decisions.json').write_text(json.dumps(decisions,indent=2)+'\n');(a.output/'summary.json').write_text(json.dumps(dict(confusion=counts,decisions=decisions,geometry_query_s=g_time,query_s={name:json.loads((path/'summary.json').read_text())['runtime_s'] for name,path in [('C',a.continuous),('I',a.independent)]}),indent=2)+'\n');np.savez_compressed(a.output/'geometry.npz',pose=np.array(predicted))
(a.output/'contract.json').write_text(json.dumps(dict(continuous=str(a.continuous),independent=str(a.independent),oracle_matrix=str(a.oracle_matrix),routes=str(a.routes),scope='Known DEV map48 18 candidate conditions, two task roots, no new physics. All G/I/C selection sees predicted gates/time only; true future solely scoring. Oracle uses actual outcomes as offline upper bound.',geometry='Exact reference at1.8m/s, yaw tangent, level posture, initial locked joints and constant height; original goal hold and obstacle/world envelope gates. No dynamic prediction.',backtracking='Full same-candidate/predictor/gate traversal, no greedy truncation; selection time excludes cached model queries.',calibration='Raw envelopes, no newly calibrated margins; not formal P4 acceptance'),indent=2)+'\n');print(json.dumps(counts))
