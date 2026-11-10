"""Compare recorded model futures with measured events, without refitting."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.ditch_event_model import EventPlanner

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--method',default='event_model');p.add_argument('--model',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
cfg=json.loads((a.run/'config.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text());summary=json.loads((a.run/'evaluation/summary.json').read_text());records=json.loads((a.run/'evaluation/moving_planning.json').read_text())['records'];data=dict(np.load(a.run/'evaluation/trajectories.npz'));planner=EventPlanner(a.model or cfg['moving_event_planner'].get('event_models',{}).get(a.method,cfg['moving_event_planner']['model']));params=np.asarray(planner.meta['candidate_parameters'],np.float32);rows=[];completion=[]
# Root rows are post-step; observation/history rows are causal pre-step states.
def encoded(lane,t,root,road):
 ix=min(int(np.floor((t+1e-6)/.02)),len(data['time'])-1)
 return planner.encode(np.asarray(root)[None],data['observation'][ix,lane:lane+1],data['controller_history'][ix,lane:lane+1],np.array([road]))

def angles(q):
 q=np.asarray(q);q=q/np.linalg.norm(q);x,y,z,w=q
 return np.array([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1)),np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))])

with torch.no_grad():
 for r in records:
  if r['method']!=a.method or r['status']!='published':continue
  lane=r['lane'];selected=r['selected'];case=summary['cases'][lane];query_t=r['query_step']*.02;parameter=params[selected:selected+1];jump=parameter[0,-1]>.5;prediction=r['prediction'];points=[(query_t,np.asarray(r['query_root']),False)]
  entry=case['entry_s']
  if entry is None:points.append(None)
  else:
   ix=max(0,round(entry/.02)-1);points.append((entry,data['root_xyzw'][ix,lane],bool(data['road_violation_history'][ix,lane])))
  for event in (0,2,3,4):
   t=float(data['event_time'][lane,event]);points.append((t,data['event_root_xyzw'][lane,event],bool(data['event_road_violation'][lane,event])) if np.isfinite(t) else None)
  if a.model:
   query_state=encoded(lane,query_t,np.asarray(r['query_root']),bool(data['road_violation_history'][max(0,r['query_step']-1),lane]));prediction=planner.score(np.repeat(query_state,len(params),axis=0),params,particles=32)
  active=range(5) if jump else [4]
  for mode in active:
   source=points[mode] if jump else points[0];target=points[mode+1]
   if source is None:continue
   t,root,road=source;s=encoded(lane,t,root,road);inp=torch.from_numpy(s);plan=torch.from_numpy(planner.plan_features(s,parameter));crop=torch.from_numpy(planner.map.crop(s[:,:13]));mean,var,logtime,logit=planner.model(inp,plan,crop,torch.tensor([mode]));dt=torch.expm1(logtime.clamp(0,np.log(11.)));one=(planner.model.physical_baseline(inp,dt,torch.tensor([mode]))+mean*planner.model.residual_std+planner.model.residual_mean)[0].numpy();rollout=np.asarray(prediction['events'][mode]['mean_root'][selected]);hit=target is not None
   row=dict(lane=lane,root=protocol['cases'][lane]['root'],decision_x=protocol['cases'][lane]['x'],skill='jump' if jump else 'ground',mode=mode,event_exists=hit,one_step_probability=float(logit.sigmoid()),rollout_conditional_probability=prediction['events'][mode]['mean_event_probability'][selected])
   if hit:
    actual_t,actual,_=target;delta=angles(one[3:7])-angles(actual[3:7]);delta=np.arctan2(np.sin(delta),np.cos(delta))
    row.update(one_step_time_error_s=abs(float(dt)-(actual_t-t)),one_step_xy_error_m=float(np.linalg.norm(one[:2]-actual[:2])),rollout_xy_error_m=float(np.linalg.norm(rollout[:2]-actual[:2])),one_step_height_error_m=float(abs(one[2]-actual[2])),rollout_height_error_m=float(abs(rollout[2]-actual[2])),one_step_rpy_error_deg=np.abs(np.rad2deg(delta)).tolist(),one_step_velocity_error_m_s=float(np.linalg.norm(one[7:10]-actual[7:10])),predicted_rollout_xy=rollout[:2].tolist(),actual_xy=actual[:2].tolist(),one_step_xy=one[:2].tolist())
   rows.append(row)
  completion.append(dict(lane=lane,root=protocol['cases'][lane]['root'],decision_x=protocol['cases'][lane]['x'],success=case['success'],probability=prediction['predicted_completion'][selected],predicted_duration_s=prediction['predicted_time_s'][selected],actual_duration_s=float(data['event_time'][lane,4])-query_t if case['success'] else None))
metrics=[]
for skill,mode in sorted(set((r['skill'],r['mode']) for r in rows)):
 subset=[r for r in rows if r['skill']==skill and r['mode']==mode];positive=[r for r in subset if r['event_exists']];m=dict(skill=skill,mode=mode,n=len(subset),events=len(positive),one_step_brier=float(np.mean([(r['one_step_probability']-r['event_exists'])**2 for r in subset])))
 for key in ('one_step_time_error_s','one_step_xy_error_m','rollout_xy_error_m','one_step_height_error_m','rollout_height_error_m','one_step_velocity_error_m_s'):m[key]=float(np.mean([r[key] for r in positive])) if positive else None
 metrics.append(m)
result=dict(metrics=metrics,completion_brier=float(np.mean([(r['probability']-r['success'])**2 for r in completion])),completion=completion,events=rows,source=str(a.run),model_override=str(a.model) if a.model else None,executed_method=a.method,scope='Frozen model. One-step prediction receives measured previous event; rollout uses only original query state. Positive-event errors exclude missing targets; all observed failures remain in probabilities. Prefix-conditioned diagnostics, not a new planning test.')
(a.output/'results.json').write_text(json.dumps(result,indent=2));np.savez_compressed(a.output/'diagnostics.npz',**{k:np.array([r[k] for r in rows if r['event_exists']]) for k in ('one_step_xy_error_m','rollout_xy_error_m','one_step_time_error_s')})
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
for j,m in enumerate(metrics):
 positive=[r for r in rows if r['skill']==m['skill'] and r['mode']==m['mode'] and r['event_exists']]
 axes[0].scatter([r['one_step_xy_error_m'] for r in positive],[r['rollout_xy_error_m'] for r in positive],s=20,alpha=.65,label=f"{m['skill']} mode {m['mode']} n={len(positive)}")
axes[0].set(xlabel='Measured-prefix one-step XY error (m)',ylabel='Query-only rollout XY error (m)',title='Every observed event; missing events excluded');axes[0].legend(fontsize=8)
for x in sorted(set(r['decision_x'] for r in completion)):
 sub=[r for r in completion if r['decision_x']==x];axes[1].scatter([r['probability'] for r in sub],[int(r['success']) for r in sub],alpha=.5,label=f'x={x}m')
axes[1].set(xlabel='Predicted completion probability',ylabel='Measured completion (0/1)',title='All selected plans; no hypothetical outcomes');axes[1].legend()
for ax in axes:ax.grid(alpha=.2)
fig.savefig(a.output/'prediction_errors.png',dpi=160);plt.close(fig);print(json.dumps(dict(metrics=metrics,completion_brier=result['completion_brier']),indent=2))
