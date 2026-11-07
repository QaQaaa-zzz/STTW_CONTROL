"""CPU-only closeout: actual5ms contact torque/geometry summaries, no simulation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from sttw_control.contact_torque_geometry import contact_torque_time_batch
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,action='append',required=True);a=p.parse_args();torch.set_num_threads(2)
for run in a.run:
 started=time.perf_counter();raw=run/'physics_contacts';meta=json.loads((raw/'index.json').read_text());dt=meta['physics_dt_s'];width=round(.005/dt);steps=[];blocks=[];alive=[]
 for chunk in meta['chunks']:
  with np.load(raw/chunk['file']) as f:d={k:f[k] for k in f.files}
  moments=contact_torque_time_batch({k:torch.as_tensor(v) for k,v in d.items() if k!='physics_step'},meta['body_order']);live=d['alive_before'].astype(bool);values={k:v.numpy() for k,v in moments.items()};steer=meta['joint_order'].index('steering_joint');values['steering_motor_torque']=d['motor_torque'][:,:,steer];values['steering_rate']=d['joint_velocity'][:,:,steer]
  for wheel in ('frontwheel','rearwheel'):
   count=d[wheel+'_normal_count'][:,:,0];start=d[wheel+'_normal_start'][:,:,0];slots=np.arange(int(count.max(initial=0)));ids=np.clip(start[:,:,None]+slots,0,d[wheel+'_normal_force'].shape[1]-1);rows=np.arange(len(count))[:,None,None];force=d[wheel+'_normal_force'][rows,ids,0]*(slots<count[:,:,None]);values[wheel+'_vertical_load']=(force*d[wheel+'_normal_direction'][rows,ids,2]).sum(2)
  steps.append(d['physics_step']);blocks.append(values);alive.append(live)
 physical=np.concatenate(steps);live=np.concatenate(alive);series={k:np.concatenate([b[k] for b in blocks]) for k in blocks[0]};bins=(physical-1)//width;unique=np.unique(bins);result=dict(physics_bin=unique,time_start_s=unique*.005,recorded_physics_samples=np.array([(bins==i).sum() for i in unique]),alive_physics_samples=np.stack([live[bins==i].sum(0) for i in unique]))
 for key,value in series.items():
  means=[];minimum=[];maximum=[];impulses=[];finite_counts=[]
  for i in unique:
   use=bins==i;v=value[use];valid=live[use]&np.isfinite(v);count=valid.sum(0);total=np.where(valid,v,0.).sum(0);lo=np.where(valid,v,np.inf).min(0);hi=np.where(valid,v,-np.inf).max(0)
   means.append(np.divide(total,count,out=np.full_like(total,np.nan,dtype=float),where=count>0));minimum.append(np.where(count>0,lo,np.nan));maximum.append(np.where(count>0,hi,np.nan));finite_counts.append(count)
   if key.endswith('moment') or key in ('negative_lateral','reference_delta','steering_motor_torque'):impulses.append(total*dt)
  result.update({key+'_mean':np.stack(means),key+'_min':np.stack(minimum),key+'_max':np.stack(maximum),key+'_finite_samples':np.stack(finite_counts)})
  if impulses:result[key+'_impulse_Nms']=np.stack(impulses)
 np.savez_compressed(run/'contact_5ms_summary.npz',**result)
 identity=dict(source=str(raw.resolve()),run=str(run.resolve()),record_window_s=meta.get('record_window_s'),physics_dt_s=dt,bin_dt_s=.005,recorded_steps=len(physical),lanes=live.shape[1],bins=len(unique),alive_filter='alive_before_this_physics_step; post-failure samples excluded',boundary_bins='partial first/last bins retain actual sample counts, never zero padded',torque_units='N m; impulses N m s; means/min/max over finite alive observations',trail_units='m; actual friction patch local-normal projected steering-axis trail; missing/singular samples remain NaN; reference_trail is a geometry reference only',load_units='N from actual native normal buffers; not actuator power or task success',kernel='existing contact_torque_time_batch/front_contact_torques; normal/friction own points, full vector steering-axis projection',elapsed_s=time.perf_counter()-started)
 (run/'contact_5ms_summary.json').write_text(json.dumps(identity,indent=2)+'\n');print(json.dumps(identity),flush=True)
