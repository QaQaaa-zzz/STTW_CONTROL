"""Reconstruct solved-step wheel forces and front steering-axis moments."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
idx=json.loads((a.run/'physics_contacts/index.json').read_text());c=json.loads((a.run/'contract.json').read_text());dt=idx['physics_dt_s'];n=c['settings']['num_envs'];fork=idx['body_order'].index('frontfork')
frames=[];times=[];maxcounts=np.zeros((2,2),int)
for chunk in idx['chunks']:
 d=np.load(a.run/'physics_contacts'/chunk['file']);count=len(d['physics_step']);f=np.zeros((count,n,2,3));moment=np.zeros((count,n,2));front=d['body_state'][:,:,fork];axis=Rotation.from_quat(front[:,:,3:7].reshape(-1,4)).apply([0,1,0]).reshape(count,n,3)
 for wi,wheel in enumerate(['frontwheel','rearwheel']):
  for kind in ['normal','friction']:
   counts=d[wheel+'_'+kind+'_count'];starts=d[wheel+'_'+kind+'_start'];force=d[wheel+'_'+kind+'_force'];points=d[wheel+'_'+kind+'_point'];maxcounts[wi,int(kind=='friction')]=max(maxcounts[wi,int(kind=='friction')],int(counts.max()))
   if kind=='normal':force=force*d[wheel+'_normal_direction']
   for t in range(count):
    for lane in range(n):
     for filter_id in range(counts.shape[2]):
      start=int(starts[t,lane,filter_id]);k=int(counts[t,lane,filter_id]);sl=slice(start,start+k);values=force[t,sl];f[t,lane,wi]+=values.sum(0)
      if wi==0:moment[t,lane,int(kind=='friction')]+=(np.cross(points[t,sl]-front[t,lane,:3],values)*axis[t,lane]).sum()
 if not np.isfinite(f).all() or not np.isfinite(moment).all():raise ValueError('Nonfinite active contact buffer')
 frames.append(dict(force=f,moment=moment,steer_velocity=d['joint_velocity'][:,:,idx['joint_order'].index('steering_joint')],motor=d['motor_torque'],alive=d['alive_before']))
 times.append(d['physics_step']*dt)
out={key:np.concatenate([row[key] for row in frames]) for key in frames[0]};out['time_s']=np.concatenate(times)
np.savez_compressed(a.run/'physics_wrench.npz',**out)
report=dict(physics_steps=len(out['time_s']),dt_s=dt,max_patch_count=maxcounts.tolist(),scope='normal and friction moments at their own world points, projected onto frontfork local+y steering axis; no causal compensation',lanes=[])
for lane,alpha in enumerate(c['settings']['alphas']):
 valid=out['alive'][:,lane];m=out['moment'][valid,lane];report['lanes'].append(dict(alpha=alpha,signed_impulse_Nms=(m.sum(0)*dt).tolist(),absolute_impulse_Nms=(abs(m).sum(0)*dt).tolist(),peak_abs_moment_Nm=abs(m).max(0).tolist(),peak_abs_steer_rate_rad_s=float(abs(out['steer_velocity'][valid,lane]).max())))
(a.run/'physics_wrench_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
