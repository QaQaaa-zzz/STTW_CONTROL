"""G0 actual-contact geometry check and paired mesh-resolution acceptance."""
import argparse,json
from pathlib import Path
import numpy as np
from sttw_control.gravel_contact_events import raised_contact_and_load,intervals
from sttw_control.route_pebbles import project_route
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);a=p.parse_args()
c=json.loads((a.run/'contract.json').read_text());s=c['settings'];index=json.loads((a.run/'physics_contacts/index.json').read_text());dt=index['physics_dt_s'];n=s['num_envs'];terrain_paths=list(s['evaluation_terrains'].values());terrains=[np.load(Path(x)/'terrain.npz') for x in terrain_paths]
route=np.load(Path(c['bundle'])/'route.npz');parts=[];residuals=[[] for _ in range(n)];moments=[]
import torch
from sttw_control.contact_torque_geometry import front_contact_torques
for chunk in index['chunks']:
 with np.load(a.run/'physics_contacts'/chunk['file']) as f:d={k:f[k] for k in f.files}
 contacts=[];normal=[];vertical=[]
 for lane in range(n):
  cr=[];nr=[];vr=[]
  for wheel in ['frontwheel','rearwheel']:
   raised,load,vload,res=raised_contact_and_load(d,wheel,lane,np.asarray(index['origins'][lane]),terrains[lane]);cr.append(raised);nr.append(load);vr.append(vload);residuals[lane].extend(res.tolist())
  contacts.append(np.stack(cr,1));normal.append(np.stack(nr,1));vertical.append(np.stack(vr,1))
 body=d['body_state'][...,:3]-np.array(index['origins'])[None,:,None,:]
 parts.append(dict(time_s=d['physics_step']*dt,contact=np.stack(contacts,1),normal=np.stack(normal,1),vertical=np.stack(vertical,1),wheel_height=body[:,:,[index['body_order'].index('frontwheel'),index['body_order'].index('rearwheel')]],steer_velocity=d['joint_velocity'][:,:,index['joint_order'].index('steering_joint')],motor=d['motor_torque'][:,:,index['joint_order'].index('steering_joint')]))
 torques=front_contact_torques({k:torch.as_tensor(v) for k,v in d.items() if k!='physics_step'},index['body_order']);moments.append({k:v.numpy() for k,v in torques.items()})
x={k:np.concatenate([d[k] for d in parts]) for k in parts[0]};torques={k:np.concatenate([d[k] for d in moments]) for k in moments[0]};time=x['time_s'];out=[]
for lane in range(n):
 contact=x['contact'][:,lane];wheel_s=project_route(x['wheel_height'][:,lane,:,:2].reshape(-1,2),route)[0].reshape(-1,2);rough=(wheel_s>=3)&(wheel_s<=7)
 events=[intervals(contact[:,k],time,dt) for k in (0,1)];union=contact.any(1)
 out.append(dict(lane=lane,label=s['case_labels'][lane],wheel_events=events,raised_contact_frames=contact.sum(0).tolist(),overlap_ratio=float(contact.all(1).sum()/max(1,union.sum())),normal_load_integral_rough_Ns=(x['normal'][:,lane]*rough).sum(0).tolist(),contact_height_residual_abs_p95_m=float(np.percentile(abs(np.array(residuals[lane])),95)) if residuals[lane] else None))
 out[-1]['normal_load_integral_rough_Ns']=((x['normal'][:,lane]*rough).sum(0)*dt).tolist()
pairs=[]
for coarse,fine in [(0,1),(2,3)]:
 checks=[]
 for wheel in (0,1):
  ca=out[coarse]['wheel_events'][wheel];fi=out[fine]['wheel_events'][wheel]
  timing=[abs(ca[0]['on_s']-fi[0]['on_s']),abs(ca[-1]['off_s']-fi[-1]['off_s'])] if ca and fi else [None,None]
  coarse_load=out[coarse]['normal_load_integral_rough_Ns'][wheel];fine_load=out[fine]['normal_load_integral_rough_Ns'][wheel]
  rel=abs(coarse_load-fine_load)/max(abs(fine_load),1e-12)
  height_rmse=float(np.sqrt(np.mean((x['wheel_height'][:,coarse,wheel,2]-x['wheel_height'][:,fine,wheel,2])**2)))
  checks.append(dict(wheel=['front','rear'][wheel],first_last_contact_difference_s=timing,normal_integral_relative_difference=rel,wheel_center_height_rmse_m=height_rmse,pass_engineering=all(t is not None and t<=.01 for t in timing) and rel<=.05 and height_rmse<=.002))
 pairs.append(dict(scene=s['case_labels'][coarse].split('_')[0],wheels=checks,selected_grid_m=.01 if all(q['pass_engineering'] for q in checks) else .005))
status=json.loads((a.run/'status.json').read_text());report=dict(role='G0 engineering DEV; unchanged frozen model, not task capability proof',lanes=out,resolution_comparisons=pairs,all_actual_stone_contact=all(all(v>0 for v in r['raised_contact_frames']) for r in out),failures=status['failures'],record_window_s=index.get('record_window_s'),observed_physics_s=[float(time[0]),float(time[-1])])
(a.run/'gravel_contact_events.json').write_text(json.dumps(report,indent=2)+'\n')
# Summaries use exact25-sample/5ms blocks; raw0.2ms patches remain on disk.
count=len(time)//25
agg={}
for key,v in torques.items():
 b=v[:count*25].reshape(count,25,n);agg[key+'_mean']=b.mean(1);agg[key+'_min']=b.min(1);agg[key+'_max']=b.max(1);agg[key+'_impulse']=b.sum(1)*dt
np.savez_compressed(a.run/'contact_5ms.npz',time_s=time[:count*25].reshape(count,25)[:,-1],**agg)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with np.load(a.run/'trajectory.npz') as f:
 pose=f['pose'];alive=f['alive'].astype(bool);rewards=f['reward'];valid_policy=f['valid'].astype(bool);trace=f['lower_trace'].reshape(-1,n,11);measurement=f['measurement']
fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
for lane,ax in enumerate(axes.flat):
 ax.plot(*route['xy'].T,':',color='.5',label='Original reference');valid=alive[:,lane];ax.plot(*pose[valid,lane,:2].T,color='#0072B2',label='Actual');ax.plot(*pose[np.flatnonzero(valid)[-1],lane,:2],marker='x',color='#D55E00');ax.set(xlabel='World x (m)',ylabel='World y (m)',title=s['case_labels'][lane]);ax.set_aspect('equal');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig(a.run/'xy_g0.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(5,1,figsize=(12,12),layout='constrained')
for lane,label in enumerate(s['case_labels']):
 style=['-','--','-.',':'][lane]
 axes[0].plot(time,x['normal'][:,lane,0],ls=style,label=label,lw=.7);axes[1].plot(time,x['steer_velocity'][:,lane],ls=style,lw=.7);axes[2].plot(time,x['wheel_height'][:,lane,0,2],ls=style,lw=.7);axes[3].plot(time,x['contact'][:,lane,0].astype(int)+lane*1.3,ls=style,lw=.7);axes[4].plot((np.arange(len(measurement))+1)*.005,np.rad2deg(measurement[:,lane,0]),ls=style,lw=.7)
for ax,y in zip(axes,['Front normal load (N)','Steering rate (rad/s)','Front wheel center z (m)','Actual stone contact + lane offset','Roll (deg)']):ax.set(xlabel='Time (s)',ylabel=y);ax.grid(alpha=.2)
axes[0].legend();fig.savefig(a.run/'contact_g0.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(3,1,figsize=(11,9),layout='constrained')
for lane,label in enumerate(s['case_labels']):
 ids=np.flatnonzero(valid_policy[:,lane]);axes[0].plot(ids+1,rewards[ids,lane],label=label);axes[1].plot(ids+1,np.cumsum(rewards[ids,lane]));axes[2].plot((np.arange(len(trace))+1)*.005,trace[:,lane,2]-2.3,label=label)
for ax,y in zip(axes,['Reward / policy step','Cumulative reward','True speed error (m/s)']):ax.set(xlabel='Policy step' if ax is not axes[2] else 'Time (s)',ylabel=y);ax.grid(alpha=.2)
axes[0].legend();fig.savefig(a.run/'reward_g0.png',dpi=150);plt.close(fig)
assets=json.loads(a.assets.read_text());assets['terrains'].update({r['scene']:str(Path(terrain_paths[0]).parent/(r['scene']+'_'+str(int(r['selected_grid_m']*1000))+'mm')) for r in pairs});a.assets.write_text(json.dumps(assets,indent=2)+'\n')
(a.run/'RESULTS.md').write_text('# G0 embedded pebbles\n\n![Actual/reference XY](xy_g0.png)\n\n![Contact responses](contact_g0.png)\n\n'+json.dumps(report,indent=2)+'\n\nP0 not launched. Fine grid selected when declared coarse/fine tolerance is exceeded; this is not an independent5mm convergence proof.\n')
print(json.dumps(report))
