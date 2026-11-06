"""Reconstruct solved-step wheel forces and front steering-axis moments."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
idx=json.loads((a.run/'physics_contacts/index.json').read_text());c=json.loads((a.run/'contract.json').read_text());dt=idx['physics_dt_s'];n=c['settings']['num_envs'];fork=idx['body_order'].index('frontfork')
frames=[];times=[];maxcounts=np.zeros((2,2),int)
for chunk in idx['chunks']:
 archive=np.load(a.run/'physics_contacts'/chunk['file']);d={key:archive[key] for key in archive.files};archive.close();count=len(d['physics_step']);f=np.zeros((count,n,2,3));moment=np.zeros((count,n,2));normal_force=np.zeros_like(f);ridge=np.zeros((count,n,2),bool);components=np.zeros((count,n,3));front=d['body_state'][:,:,fork];axis=Rotation.from_quat(front[:,:,3:7].reshape(-1,4)).apply([0,1,0]).reshape(count,n,3)
 for wi,wheel in enumerate(['frontwheel','rearwheel']):
  for kind in ['normal','friction']:
   counts=d[wheel+'_'+kind+'_count'];starts=d[wheel+'_'+kind+'_start']
   maxcounts[wi,int(kind=='friction')]=max(maxcounts[wi,int(kind=='friction')],int(counts.max()))
   # Gather only active patches; one vectorized reduction replaces per-step/lane Python loops.
   patch=np.arange(int(counts.max()))
   ti,li,fi,pi=np.nonzero(patch[None,None,None,:]<counts[...,None])
   indices=starts[ti,li,fi]+pi
   values=d[wheel+'_'+kind+'_force'][ti,indices]
   points=d[wheel+'_'+kind+'_point'][ti,indices]
   if kind=='normal':values=values*d[wheel+'_normal_direction'][ti,indices]
   np.add.at(f[:,:,wi],(ti,li),values)
   if kind=='normal':
    np.add.at(normal_force[:,:,wi],(ti,li),values)
    raised=(points[:,2]>.003)&(np.linalg.norm(values,axis=1)>.01)
    np.logical_or.at(ridge[:,:,wi],(ti,li),raised)
   if wi==0:
    lever=points-front[ti,li,:3];steer_axis=axis[ti,li]
    np.add.at(moment[:,:,int(kind=='friction')],(ti,li),(np.cross(lever,values)*steer_axis).sum(1))
    if kind=='friction' and len(ti):
     ns=d[wheel+'_normal_start'][ti,li,fi];nk=d[wheel+'_normal_count'][ti,li,fi]
     keep=nk>0;ti,li,fi,ns,nk,values,points,lever,steer_axis=[x[keep] for x in (ti,li,fi,ns,nk,values,points,lever,steer_axis)]
     if len(ti):
      slots=np.arange(int(nk.max()));ni=ns[:,None]+np.minimum(slots[None,:],nk[:,None]-1)
      candidates=d[wheel+'_normal_point'][ti[:,None],ni]
      distance=np.linalg.norm(points[:,None,:]-candidates,axis=-1)
      distance=np.where(slots[None,:]<nk[:,None],distance,np.inf)
      nearest=ni[np.arange(len(ti)),distance.argmin(1)]
      normal=d[wheel+'_normal_direction'][ti,nearest]
      axle=Rotation.from_quat(d['body_state'][ti,li,idx['body_order'].index(wheel),3:7]).apply([0,0,1])
      longitudinal=np.cross(axle,normal);longitudinal/=np.linalg.norm(longitudinal,axis=1,keepdims=True)
      lateral=np.cross(normal,longitudinal)
      for ci,direction in enumerate([longitudinal,lateral,normal]):
       part=(values*direction).sum(1)[:,None]*direction
       np.add.at(components[:,:,ci],(ti,li),(np.cross(lever,part)*steer_axis).sum(1))

 if not np.isfinite(f).all() or not np.isfinite(moment).all():raise ValueError('Nonfinite active contact buffer')
 frames.append(dict(force=f,normal_force=normal_force,ridge_contact=ridge,friction_components=components,moment=moment,steer_velocity=d['joint_velocity'][:,:,idx['joint_order'].index('steering_joint')],motor=d['motor_torque'],alive=d['alive_before']))
 times.append(d['physics_step']*dt)
out={key:np.concatenate([row[key] for row in frames]) for key in frames[0]};out['time_s']=np.concatenate(times)
np.savez_compressed(a.run/'physics_wrench.npz',**out)
report=dict(physics_steps=len(out['time_s']),dt_s=dt,max_patch_count=maxcounts.tolist(),scope='normal and friction moments at their own world points, projected onto frontfork local+y steering axis; no causal compensation',lanes=[])
for lane,alpha in enumerate(c['settings']['alphas']):
 valid=out['alive'][:,lane];m=out['moment'][valid,lane];report['lanes'].append(dict(alpha=alpha,signed_impulse_Nms=(m.sum(0)*dt).tolist(),absolute_impulse_Nms=(abs(m).sum(0)*dt).tolist(),peak_abs_moment_Nm=abs(m).max(0).tolist(),peak_abs_steer_rate_rad_s=float(abs(out['steer_velocity'][valid,lane]).max())))
(a.run/'physics_wrench_summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

# Saturation uses the actually applied steering servo torque; low load uses vertical normal force.
def durations(mask):
 edges=np.diff(np.r_[False,mask,False].astype(int));spans=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
 return dict(total_s=float(mask.sum()*dt),longest_s=float(spans.max()*dt) if len(spans) else 0.,intervals=int(len(spans)))
margin=dict(scope='natural acquisition; first150ms mean vertical front normal load is operational pre-obstacle reference, not theoretical static load',friction_basis='wheel local z axle crossed with nearest normal-patch normal at each friction point; normal residual retained; nearest association is geometric, not native patch linkage',lanes=[])
for lane,alpha in enumerate(c['settings']['alphas']):
 load=out['normal_force'][:,lane,0,2];baseline=float(load[:round(.15/dt)].mean());valid=out['alive'][:,lane];sat=(abs(out['motor'][:,lane,idx['joint_order'].index('steering_joint')])>=.999*max(abs(v) for v in idx['motor_limits'][0]))&valid
 event=out['ridge_contact'][:,lane];events=[]
 for wi in range(2):
  ids=np.flatnonzero(event[:,wi]&valid);events.append(dict(first_s=float(out['time_s'][ids[0]]) if len(ids) else None,last_s=float(out['time_s'][ids[-1]]) if len(ids) else None))
 comp=out['friction_components'][valid,lane];row=dict(alpha=alpha,front_reference_load_N=baseline,steering_saturation=durations(sat),front_low_load=durations((load<.2*baseline)&valid),ridge_events=events,friction_component_signed_impulse_Nms=(comp.sum(0)*dt).tolist(),friction_component_absolute_impulse_Nms=(abs(comp).sum(0)*dt).tolist(),friction_decomposition_max_residual_Nm=float(abs(comp.sum(1)-out['moment'][valid,lane,1]).max()))
 margin['lanes'].append(row)
(a.run/'control_margin_summary.json').write_text(json.dumps(margin,indent=2)+'\n')
