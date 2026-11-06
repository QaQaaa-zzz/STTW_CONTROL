"""Offline front-contact trail proxies from synchronized saved link/contact data."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--root',type=Path);g.add_argument('--run',type=Path);a=p.parse_args();out=(a.root/'analysis'/'trail_geometry' if a.root else a.run/'trail_geometry');out.mkdir(exist_ok=True)
rows=[];contact_times=[];fig,axs=plt.subplots(5,1,figsize=(11,14),layout='constrained')
for lane_number,(alpha,color) in enumerate(zip([0,.25,.5,.75,1],['#0072B2','#D55E00','#009E73','#CC79A7','#8A6500'])):
 lane=0 if a.root else lane_number;run=a.root/'entry_projection'/f'alpha{alpha:g}'/'evaluation' if a.root else a.run;d=np.load(run/'trajectory.npz');c=json.loads((run/'contract.json').read_text());b=d['body_state'][:,lane];o=b[:,c['bodies'].index('frontfork'),:3];axis=Rotation.from_quat(b[:,c['bodies'].index('frontfork'),3:7]).apply([0,1,0]);axle=Rotation.from_quat(b[:,c['bodies'].index('frontwheel'),3:7]).apply([0,0,1]);forward=Rotation.from_quat(b[:,0,3:7]).apply([1,0,0])
 # Last 0.2ms sample aligns with each saved 5ms post-step body state.
 point=d['wheel_contact_points'][:,-1,lane,0,0];force=d['wheel_contact_forces'][:,-1,lane,0,0];norm=np.linalg.norm(force,axis=1);valid=np.isfinite(point).all(1)&(norm>1e-3)&d['lower_trace'].reshape(len(point),-1,11)[:,lane,6].astype(bool);normal=np.full_like(force,np.nan);normal[valid]=force[valid]/norm[valid,None]
 up=np.broadcast_to([0.,0.,1.],axis.shape);values={}
 for name,n in [('horizontal',up),('resultant_normal',normal)]:
  direction=np.cross(axle,n);direction/=np.linalg.norm(direction,axis=1)[:,None];direction*=np.where((direction*forward).sum(1)>=0,1.,-1.)[:,None]
  dot=(axis*n).sum(1);ok=valid&(abs(dot)>1e-6);intersection=np.full_like(o,np.nan);intersection[ok]=o[ok]+axis[ok]*(((point[ok]-o[ok])*n[ok]).sum(1)/dot[ok])[:,None];values[name]=((intersection-point)*direction).sum(1)
 moment=(np.cross(point-o,force)*axis).sum(1);moment[~valid]=np.nan;t=(np.arange(len(o))+1)*.005
 raised=valid&(point[:,2]>.0025);flat=valid&(point[:,2]<=.0025);contact_times.extend(t[raised].tolist())
 row={'alpha':alpha,'terrain_height_m':c['terrain'][lane]['height_m'],'speed_request_m_s':c['settings'].get('user_speed_m_s',2.3),'role':'saved natural trajectory; mean contact point and normal resultant; not full contact wrench','valid_samples':int(valid.sum())}
 for name,v in values.items():
  negative=valid&(v<0);edges=np.diff(np.r_[False,negative,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
  row[name]={'negative_intervals_s':[[float((start+1)*.005),float(end*.005)] for start,end in zip(starts,ends)],'longest_negative_sample_duration_s':float(max(ends-starts,default=0)*.005)}
  for phase,mask in [('all',valid),('raised',raised),('low_contact',flat)]:
   w=v[mask];row[name][phase]={'n':len(w),'min_m':float(np.nanmin(w)),'median_m':float(np.nanmedian(w)),'negative_samples':int((w<0).sum())}
 row['negative_response']=[{'time_s':float(t[i]),'roll_rad':float(d['measurement'][i,lane,0]),'roll_rate_rad_s':float(d['measurement'][i,lane,1]),'steer_rad':float(d['measurement'][i,lane,2]),'steer_rate_rad_s':float(d['measurement'][i,lane,3]),'normal_moment_proxy_Nm':float(moment[i])} for i in np.flatnonzero(valid&(values['resultant_normal']<0))];rows.append(row);np.savez_compressed(out/f'alpha{alpha:g}.npz',time=t,valid=valid,raised=raised,point=point,force=force,axis=axis,axis_origin=o,normal_resultant_moment=moment,**values)
 axs[0].plot(t,values['horizontal']*1000,color=color,label=f'alpha {alpha:g}');axs[1].plot(t,values['resultant_normal']*1000,color=color);axs[2].plot(t,moment,color=color);axs[3].plot(t,d['measurement'][:,lane,3],color=color);axs[4].plot(t,d['measurement'][:,lane,1],color=color)
for ax in axs:ax.axhline(0,color='k',lw=.7);ax.grid(alpha=.2);ax.set_xlabel('Time (s)')
axs[0].set_ylabel('Horizontal-plane trail (mm)');axs[1].set_ylabel('Resultant-normal-plane trail (mm)');axs[2].set_ylabel('Mean-point normal moment (Nm)');axs[3].set_ylabel('Steering rate (rad/s)');axs[4].set_ylabel('Roll rate (rad/s)');axs[0].legend(ncol=5)
fig.suptitle(f"{row['terrain_height_m']*100:g}cm terrain, {row['speed_request_m_s']:g} m/s"+' | original H75 + Local300 | DEV, one state per alpha\n5ms synchronized contact/link samples; positive trail = contact behind axis intersection\nMean-point geometry proxies, not exact multi-contact wrench or proof of instability')
fig.savefig(out/'trail_geometry.png',dpi=150)
for ax in axs:ax.set_xlim(min(contact_times)-.15,max(contact_times)+.4)
fig.savefig(out/'trail_contact_detail.png',dpi=150);plt.close(fig);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
