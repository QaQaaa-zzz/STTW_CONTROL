"""Query five alphas from identical saved observations and correction state.

This is a policy/reference-map query, not a physical intervention experiment.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from sttw_control.torch_path_policy import TorchPathPolicy
p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--times',type=float,nargs='+',default=[.5,1.5,2.5,3.5]);p.add_argument('--all-decisions',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
r=json.loads(a.receipt.read_text());raw=np.load(a.receipt.with_suffix('.npz'));middle=raw['middle_trace'];identity=json.loads((a.bundle/'identity.json').read_text());cfg=json.loads((a.bundle/'config.json').read_text());scales=np.array([f['scale'] for f in cfg['network']['context_fields']]);alphas=[0,.25,.5,.75,1];rows=[];snapshots=[]
policy=TorchPathPolicy(a.bundle/'actor.npz',identity['files']['actor.npz'],cfg,5,'cpu');interface=cfg['residual_interface']
with torch.no_grad():
 for requested in (middle[::4,0] if a.all_decisions else a.times):
  # Select the actual 50 Hz decision boundary, not a held intermediate observation.
  index=4*int(np.argmin(np.abs(middle[::4,0]-requested)));saved=middle[index];obs=np.repeat(saved[None,7:358],5,axis=0);context=obs[0,336:]*scales;previous=context[10:12].copy();filtered=context[12:14].copy();nominal=np.repeat(saved[None,1:3],5,axis=0)
  obs[:,336]=np.array(alphas)/scales[0];observations=torch.tensor(obs,dtype=torch.float32);policy.offsets[:]=torch.tensor(previous);policy.filtered[:]=torch.tensor(filtered);latent=policy.latent(observations);governed,flags=policy.correct(latent,torch.tensor(nominal,dtype=torch.float32));target=flags['target'];f=policy.decay*torch.tensor(filtered)+(1-policy.decay)*target;delta=f-torch.tensor(previous)
  budget=torch.stack((torch.where(delta[:,0]>=0,torch.full((5,),interface['correction_speed_up_m_s2']*.005),torch.full((5,),interface['correction_speed_down_m_s2']*.005)),torch.full((5,),interface['correction_steer_rate_rad_s']*.005)),dim=-1);eta=torch.where(delta.abs()>0,(budget/delta.abs()).clamp_max(1),torch.ones_like(delta));common=eta.min(-1).values;steer_priority=torch.stack((common*delta[:,0],eta[:,1]*delta[:,1]),dim=-1)
  for i,alpha in enumerate(alphas):
   row=dict(time_s=float(saved[0]),alpha=alpha,target_speed_correction=float(target[i,0]),target_steer_correction=float(target[i,1]),previous_speed_correction=float(previous[0]),previous_steer_correction=float(previous[1]),filtered_speed_target=float(f[i,0]),filtered_steer_target=float(f[i,1]),eta_speed=float(eta[i,0]),eta_steer=float(eta[i,1]),rate_active=bool(flags['rate_clipped'][i]),speed_reference=float(governed[i,0]),steer_reference=float(governed[i,1]),steer_priority_speed_increment=float(steer_priority[i,0]),steer_priority_steer_increment=float(steer_priority[i,1]),common_speed_increment=float(common[i]*delta[i,0]),common_steer_increment=float(common[i]*delta[i,1]));rows.append(row)
  snapshots.append(dict(time_s=float(saved[0]),trace_index=index,observation=saved[7:358].tolist(),nominal=nominal[0].tolist(),offsets=previous.tolist(),filtered=filtered.tolist(),alpha_only_changed=True))
with (a.output/'queries.csv').open('w') as f:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(a.output/'identity.json').write_text(json.dumps(dict(source_receipt=str(a.receipt.resolve()),source_policy=identity['files']['actor.npz'],source_config=identity['files']['config.json'],source_alpha=r['path_contract']['alpha'],alphas=alphas,snapshots=snapshots,physics_steps=0,optimizer_updates=0,role='same saved policy observation/history/filter query; not closed-loop benefit or physical intervention',candidate='steer-priority increments are before final reference clamping; no candidate executed'),indent=2))
summary=[]
for t in sorted({r['time_s'] for r in rows}):
 group=[r for r in rows if r['time_s']==t];summary.append(dict(time_s=t,target_speed_span=max(r['target_speed_correction'] for r in group)-min(r['target_speed_correction'] for r in group),target_steer_span=max(r['target_steer_correction'] for r in group)-min(r['target_steer_correction'] for r in group),rate_active_count=sum(r['rate_active'] for r in group),steer_limited_more_than_speed=sum(r['eta_steer']<r['eta_speed'] for r in group)))
(a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
