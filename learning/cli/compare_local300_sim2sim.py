"""Plot saved MuJoCo and Isaac Lab Local300 standard cases without simulation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--start-relative',action='store_true');p.add_argument('--historical-lab',type=Path);p.add_argument('--zero',type=Path);p.add_argument('--reference-label',default='MuJoCo Local300');p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--label',default='Lab Local300');a=p.parse_args();s=json.loads(a.config.read_text());d=np.load(a.run/'trajectory.npz');rows=[]
d={key:d[key].copy() for key in d.files}
zero=dict(np.load(a.zero/'trajectory.npz')) if a.zero else None
historical=dict(np.load(a.historical_lab/'trajectory.npz')) if a.historical_lab else None
def relative(xy,origin,yaw):
 c,sn=np.cos(yaw),np.sin(yaw)
 return (xy-origin)@np.array([[c,-sn],[sn,c]])
if a.start_relative:
 for data in (d,zero,historical):
  if data is not None:
   for lane in range(data['pose'].shape[1]):
    origin=data['pose'][0,lane,:2].copy();yaw=float(data['pose'][0,lane,2])
    data['pose'][:,lane,:2]=relative(data['pose'][:,lane,:2],origin,yaw)
def path_distance(xy,reference):
 segment=np.diff(reference,axis=0);norm=np.sum(segment**2,axis=1)
 out=[]
 for points in np.array_split(xy,max(1,(len(xy)+127)//128)):
  offset=points[:,None,:]-reference[None,:-1,:]
  fraction=np.clip(np.sum(offset*segment[None],axis=-1)/np.maximum(norm,1e-20),0,1)
  out.append(np.sqrt(np.min(np.sum((offset-fraction[:,:,None]*segment[None])**2,axis=-1),axis=1)))
 return np.concatenate(out)
paths,paxs=plt.subplots(2,len(s['cases']),figsize=(21,6),squeeze=False)
path_rows=[];path_arrays={}
fig,axes=plt.subplots(2,4,figsize=(17,9));states,axs=plt.subplots(3,len(s['cases']),figsize=(21,9));axes.flat[-1].axis('off')
for i,case in enumerate(s['cases']):
 old=dict(np.load(Path(s['mujoco_reference'])/(case+'.npz')))
 if a.start_relative:
  origin=old['actual_xy'][0].copy();yaw=float(old['yaw_unwrapped'][0])
  old['actual_xy']=relative(old['actual_xy'],origin,yaw);old['reference_xy']=relative(old['reference_xy'],origin,yaw)
 valid=d['valid'][:,i];n=int(valid.sum());t=d['time'][:n,i]+d['stop_tick'][:n,i]*.0002;ot=old['time']+.005;common=min(n,len(ot));new=np.column_stack((d['speed'][:n,i],d['measurement'][:n,i,2],d['measurement'][:n,i,0]));previous=np.column_stack((old['actual_forward_speed'],old['actual_delta'],old['phi']));ax=axes.flat[i]
 ax.plot(*old['reference_xy'].T,':',color='gray',label='Original command reference')
 for xy,label,color,style in [(old['actual_xy'],a.reference_label,'#0072B2','--'),(d['pose'][:n,i,:2],a.label,'#D55E00','-')]:
  ax.plot(*xy.T,style,color=color,label=label);ax.scatter(*xy[-1],marker='x',color=color)
 if zero is not None:
  nz=int(zero['valid'][:,i].sum());ax.plot(*zero['pose'][:nz,i,:2].T,'-.',color='#009E73',label='Isaac Lab ECBC+ESO');ax.scatter(*zero['pose'][nz-1,i,:2],marker='x',color='#009E73')
 if historical is not None:
  nh=int(historical['valid'][:,i].sum());ax.plot(*historical['pose'][:nh,i,:2].T,':',color='#CC79A7',label='Original chassis Lab Local300');ax.scatter(*historical['pose'][nh-1,i,:2],marker='x',color='#CC79A7')
 curves=[('mujoco',old['actual_xy'],ot,old['actual_forward_speed']-old['governed'][:,0],'#0072B2','--'),('lab',d['pose'][:n,i,:2],t,d['speed'][:n,i]-d['reference'][:n,i,0],'#D55E00','-')]
 if zero is not None:curves.append(('ecbc',zero['pose'][:nz,i,:2],zero['time'][:nz,i]+zero['stop_tick'][:nz,i]*.0002,zero['speed'][:nz,i]-zero['reference'][:nz,i,0],'#009E73','-.'))
 if historical is not None:curves.append(('original_lab',historical['pose'][:nh,i,:2],historical['time'][:nh,i]+historical['stop_tick'][:nh,i]*.0002,historical['speed'][:nh,i]-historical['reference'][:nh,i,0],'#CC79A7',':'))
 for name,xy,clock,ev,color,style in curves:
  distance=path_distance(xy,old['reference_xy']);dt=np.diff(np.r_[0,clock])
  paxs[0,i].plot(clock,distance,style,color=color,label=name);paxs[1,i].plot(clock,ev,style,color=color)
  path_arrays[case+'_'+name+'_time']=clock;path_arrays[case+'_'+name+'_distance']=distance;path_arrays[case+'_'+name+'_speed_error']=ev
  path_rows.append(dict(case=case,method=name,observed_s=float(clock[-1]),path_rmse_m=float(np.sqrt(np.mean(distance**2))),path_peak_m=float(distance.max()),seconds_over_half_m=float(dt[distance>.5].sum()),speed_rmse_m_s=float(np.sqrt(np.mean(ev**2)))))
 paxs[0,i].set_title(case,fontsize=8)
 for j in range(2):paxs[j,i].grid(alpha=.2);paxs[j,i].set_xlabel('Time (s)')
 ax.set(title=case,xlabel=('Start-relative x (m)' if a.start_relative else 'World x (m)'),ylabel=('Start-relative y (m)' if a.start_relative else 'World y (m)'));ax.set_aspect('equal',adjustable='datalim')
 for j,unit in enumerate(['Speed (m/s)','Steer angle (rad)','Roll (rad)']):
  axs[j,i].plot(ot,previous[:,j],'--',color='#0072B2');axs[j,i].plot(t,new[:,j],color='#D55E00');axs[j,i].set_ylabel(unit);axs[j,i].grid(alpha=.2)
  if j<2:axs[j,i].plot(ot,old['governed'][:,j],':',color='gray')
 axs[0,i].set_title(case,fontsize=8);axs[2,i].set_xlabel('Time (s)')
 delta=new[:common]-previous[:common];xy_delta=d['pose'][:common,i,:2]-old['actual_xy'][:common]
 rows.append(dict(case=case,lab_end_s=float(t[-1]),lab_failed=bool(d['failed'][:,i].any()),mujoco_end_s=float(ot[-1]),mujoco_failed=bool(old['physical_failure'].any()),common_s=float(min(t[-1],ot[-1])),xy_difference_rmse_m=float(np.sqrt(np.mean(np.sum(xy_delta**2,axis=1)))),state_difference_rmse=np.sqrt(np.mean(delta**2,axis=0)).tolist()))
 row=rows[-1];row['end_xy_difference_m']=float(np.linalg.norm(d['pose'][n-1,i,:2]-old['actual_xy'][-1]))
 row['tracking']={}
 series=[('lab_local300',new[:,:2],d['reference'][:n,i]),('mujoco_reference',previous[:,:2],old['governed'])]
 if zero is not None:series.append(('lab_ecbc',np.column_stack((zero['speed'][:nz,i],zero['measurement'][:nz,i,2])),zero['reference'][:nz,i]))
 for name,actual,reference in series:
  error=actual-reference;tol=.01 if case.startswith('post_return_small') else .02
  row['tracking'][name]=dict(rmse=np.sqrt(np.mean(error**2,axis=0)).tolist(),final_tracking_hold=bool(len(error)==2000 and (np.abs(error[-100:,0])<=.05).all() and (np.abs(error[-100:,1])<=tol).all()))
handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',fontsize=9)
fig.suptitle(a.reference_label+' vs '+a.label+', standard flat ground\n'+('Each first recorded pose aligned; different warmup states; ' if a.start_relative else 'Same saved initial state; ')+'same issued commands; crosses mark endpoints');fig.tight_layout();fig.savefig(a.run/'mujoco_lab_xy.png',dpi=150)
states.suptitle('MuJoCo dashed blue / Isaac Lab solid orange / issued reference dotted gray');states.tight_layout();states.savefig(a.run/'mujoco_lab_states.png',dpi=150)
(a.run/'comparison.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

paxs[0,0].set_ylabel('Nearest path distance (m)');paxs[1,0].set_ylabel('Speed error (m/s)')
paths.suptitle('Full-window tracking; nearest original polyline distance, not timed position error\n0.5 m exceedance duration is descriptive, not a new success threshold; curves end at recorded termination')
paxs[0,0].legend(fontsize=7)
paths.tight_layout();paths.savefig(a.run/'path_tracking.png',dpi=150)
(a.run/'path_tracking.json').write_text(json.dumps(dict(start_relative=a.start_relative,reference='original command-integrated polyline; not a feasibility guarantee',metrics=path_rows),indent=2))
np.savez_compressed(a.run/'path_tracking.npz',**path_arrays)
