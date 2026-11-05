"""Plot saved MuJoCo and Isaac Lab Local300 standard cases without simulation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--zero',type=Path);p.add_argument('--reference-label',default='MuJoCo Local300');p.add_argument('--config',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--label',default='Lab Local300');a=p.parse_args();s=json.loads(a.config.read_text());d=np.load(a.run/'trajectory.npz');rows=[]
zero=np.load(a.zero/'trajectory.npz') if a.zero else None
fig,axes=plt.subplots(2,4,figsize=(17,9));states,axs=plt.subplots(3,len(s['cases']),figsize=(21,9));axes.flat[-1].axis('off')
for i,case in enumerate(s['cases']):
 old=np.load(Path(s['mujoco_reference'])/(case+'.npz'));valid=d['valid'][:,i];n=int(valid.sum());t=d['time'][:n,i]+d['stop_tick'][:n,i]*.0002;ot=old['time']+.005;common=min(n,len(ot));new=np.column_stack((d['speed'][:n,i],d['measurement'][:n,i,2],d['measurement'][:n,i,0]));previous=np.column_stack((old['actual_forward_speed'],old['actual_delta'],old['phi']));ax=axes.flat[i]
 ax.plot(*old['reference_xy'].T,':',color='gray',label='Original command reference')
 for xy,label,color,style in [(old['actual_xy'],a.reference_label,'#0072B2','--'),(d['pose'][:n,i,:2],a.label,'#D55E00','-')]:
  ax.plot(*xy.T,style,color=color,label=label);ax.scatter(*xy[-1],marker='x',color=color)
 if zero is not None:
  nz=int(zero['valid'][:,i].sum());ax.plot(*zero['pose'][:nz,i,:2].T,'-.',color='#009E73',label='Isaac Lab ECBC+ESO');ax.scatter(*zero['pose'][nz-1,i,:2],marker='x',color='#009E73')
 ax.set(title=case,xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim')
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
fig.suptitle(a.reference_label+' vs '+a.label+', standard flat ground\nSame saved initial state and issued commands; no H75; crosses mark observation endpoints');fig.tight_layout();fig.savefig(a.run/'mujoco_lab_xy.png',dpi=150)
states.suptitle('MuJoCo dashed blue / Isaac Lab solid orange / issued reference dotted gray');states.tight_layout();states.savefig(a.run/'mujoco_lab_states.png',dpi=150)
(a.run/'comparison.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
