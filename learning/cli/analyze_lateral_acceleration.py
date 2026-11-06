"""Check sustained lateral acceleration against the proposed 2.4 m/s2 bound."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
index=json.loads((a.evaluation/'index.json').read_text());rows=[];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8A6500'];fig,axes=plt.subplots(2,5,figsize=(20,8),layout='constrained')
for case,ax in zip(index,axes.flat):
 d=np.load(Path(case['run'])/'trajectory.npz');c=json.loads((Path(case['run'])/'contract.json').read_text());tr=d['lower_trace'].reshape(-1,6,11)
 for lane,alpha in enumerate([0,.25,.5,.75,1]):
  count=int(tr[:,lane,6].sum());vel=d['body_state'][:count,lane,c['bodies'].index('chasis'),7:9];speed=np.linalg.norm(vel,axis=1);course=np.unwrap(np.arctan2(vel[:,1],vel[:,0]));ay=speed*np.gradient(course,.005,edge_order=2)
  # 0.2s centered average for diagnosis; no policy signal is changed.
  avg=np.convolve(ay,np.ones(40)/40,mode='valid');time=(np.arange(len(avg))+20.5)*.005;progress=np.asarray(d['progress'][:count,lane]).reshape(-1);segment=(progress[20:20+len(avg)]>=3.499)&(progress[20:20+len(avg)]<=5.819);above=(avg>2.4)&segment
  edges=np.diff(np.r_[False,above,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1);longest=float(max(ends-starts,default=0)*.005)
  r=dict(terrain=case['terrain'],speed=case['user_speed_m_s'],alpha=alpha,constant_segment_s=float(segment.sum()*.005),mean_ay=float(avg[segment].mean()),max_200ms_ay=float(avg[segment].max()),above_2p4_s=float(above.sum()*.005),longest_above_2p4_s=longest)
  rows.append(r);ax.plot(time,avg,color=colors[lane],lw=.9,label=f'alpha {alpha:g}')
 ax.axhline(2.4,color='k',ls='--',label='Proposed 2.4');ax.axhline(9.81*np.tan(.26),color='grey',ls=':',label='g tan(roll cost onset)');ax.set(xlim=(.5,4.5),title=f"{case['terrain']} | {case['user_speed_m_s']} m/s",xlabel='Time (s)',ylabel='0.2s mean lateral acceleration (m/s²)');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=6);fig.suptitle('Original H75 + Local300 | measured planar velocity/course | DEV, all five alphas\nCentered 0.2s average; 2.4 is a hypothesis line, roll-cost line is a quasi-steady equivalent, not a physical limiter')
fig.savefig(a.output/'lateral_acceleration.png',dpi=150);plt.close(fig);(a.output/'lateral_acceleration.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(sorted(rows,key=lambda r:r['longest_above_2p4_s'],reverse=True)[:6],indent=2))
