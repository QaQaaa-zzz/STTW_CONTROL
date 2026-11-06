"""Offline front-contact trail proxies from synchronized saved link/contact data."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root/'analysis'/'trail_geometry';out.mkdir(exist_ok=True)
rows=[];fig,axs=plt.subplots(3,1,figsize=(11,10),layout='constrained')
for alpha,color in zip([0,.25,.5,.75,1],['#0072B2','#D55E00','#009E73','#CC79A7','#8A6500']):
 run=a.root/'entry_projection'/f'alpha{alpha:g}'/'evaluation';d=np.load(run/'trajectory.npz');c=json.loads((run/'contract.json').read_text());b=d['body_state'][:,0];o=b[:,c['bodies'].index('frontfork'),:3];axis=Rotation.from_quat(b[:,c['bodies'].index('frontfork'),3:7]).apply([0,1,0]);axle=Rotation.from_quat(b[:,c['bodies'].index('frontwheel'),3:7]).apply([0,0,1]);forward=Rotation.from_quat(b[:,0,3:7]).apply([1,0,0])
 # Last 0.2ms sample aligns with each saved 5ms post-step body state.
 point=d['wheel_contact_points'][:,-1,0,0,0];force=d['wheel_contact_forces'][:,-1,0,0,0];norm=np.linalg.norm(force,axis=1);valid=np.isfinite(point).all(1)&(norm>1e-3);normal=np.full_like(force,np.nan);normal[valid]=force[valid]/norm[valid,None]
 up=np.broadcast_to([0.,0.,1.],axis.shape);values={}
 for name,n in [('horizontal',up),('resultant_normal',normal)]:
  direction=np.cross(axle,n);direction/=np.linalg.norm(direction,axis=1)[:,None];direction*=np.where((direction*forward).sum(1)>=0,1.,-1.)[:,None]
  dot=(axis*n).sum(1);ok=valid&(abs(dot)>1e-6);intersection=np.full_like(o,np.nan);intersection[ok]=o[ok]+axis[ok]*(((point[ok]-o[ok])*n[ok]).sum(1)/dot[ok])[:,None];values[name]=((intersection-point)*direction).sum(1)
 moment=(np.cross(point-o,force)*axis).sum(1);moment[~valid]=np.nan;t=(np.arange(len(o))+1)*.005
 raised=valid&(point[:,2]>.0025);flat=valid&(point[:,2]<=.0025)
 row={'alpha':alpha,'terrain_height_m':.04,'speed_request_m_s':2.3,'role':'natural lane0 from entry_projection; mean contact point and normal resultant; not full contact wrench','valid_samples':int(valid.sum())}
 for name,v in values.items():
  row[name]={}
  for phase,mask in [('all',valid),('raised',raised),('low_contact',flat)]:
   w=v[mask];row[name][phase]={'n':len(w),'min_m':float(np.nanmin(w)),'median_m':float(np.nanmedian(w)),'negative_samples':int((w<0).sum())}
 rows.append(row);np.savez_compressed(out/f'alpha{alpha:g}.npz',time=t,valid=valid,raised=raised,point=point,force=force,axis=axis,axis_origin=o,normal_resultant_moment=moment,**values)
 axs[0].plot(t,values['horizontal']*1000,color=color,label=f'alpha {alpha:g}');axs[1].plot(t,values['resultant_normal']*1000,color=color);axs[2].plot(t,moment,color=color)
for ax in axs:ax.axhline(0,color='k',lw=.7);ax.grid(alpha=.2);ax.set_xlabel('Time (s)')
axs[0].set_ylabel('Horizontal-plane trail (mm)');axs[1].set_ylabel('Resultant-normal-plane trail (mm)');axs[2].set_ylabel('Mean-point normal moment (Nm)');axs[0].legend(ncol=5)
fig.suptitle('4cm oblique terrain, 2.3 m/s | original H75 + Local300 | DEV, one state per alpha\n5ms synchronized contact/link samples; positive trail = contact behind axis intersection\nMean-point geometry proxies, not exact multi-contact wrench or proof of instability')
fig.savefig(out/'trail_geometry.png',dpi=150);plt.close(fig);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
