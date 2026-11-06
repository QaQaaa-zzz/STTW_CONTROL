"""Plot saved single/batched ground diagnostics without rerunning physics."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
d=np.load(a.run/'trajectory.npz');tr=d['trace'];protocol=json.loads((a.run/'protocol.json').read_text());results=json.loads((a.run/'results.json').read_text())
if tr.ndim==2:tr=tr[:,None,:]
n=tr.shape[1];valid=d['valid'] if 'valid' in d else np.ones(tr.shape[:2],bool)
cases=protocol.get('spec',{}).get('cases',[{'name':'ecbc'}]);colors=plt.get_cmap('tab10').colors
f,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for i in range(n):
 x=tr[valid[:,i],i];t=x[:,0];label=cases[i]['name'];color=colors[i%len(colors)];style='--' if 'local300' in label else '-'
 axes[0,0].plot(x[:,1],x[:,2],style,color=color,label=label);axes[0,0].plot(x[-1,1],x[-1,2],'x',color=color)
 axes[0,1].plot(t,x[:,4],style,color=color,label=label)
 axes[1,0].plot(t,np.rad2deg(x[:,5]),style,color=color,label=label)
 axes[1,1].plot(t,x[:,9],style,color=color,label=label+' hip');axes[1,1].plot(t,x[:,10],':',color=color,label=label+' knee')
 if 'reference' in d and (i==0 or cases[i]['steer_rad']!=cases[i-1]['steer_rad']):
  refs=d['reference'][valid[:,i],i];path=[x[0,1:3].copy()];yaw=x[0,3]
  for k in range(1,len(x)):
   dt=t[k]-t[k-1];dyaw=refs[k,0]*np.tan(refs[k,1])*np.cos(protocol['controller']['caster'])/protocol['controller']['wheelbase']*dt
   path.append(path[-1]+refs[k,0]*dt*np.array([np.cos(yaw+.5*dyaw),np.sin(yaw+.5*dyaw)]));yaw+=dyaw
  path=np.asarray(path);axes[0,0].plot(path[:,0],path[:,1],':',color='gray',label='kinematic command reference' if i==0 else None)
 elif 'reference' not in d:
  axes[0,0].plot([x[0,1],x[0,1]+2*t[-1]],[x[0,2],x[0,2]],':',color='gray',label='2 m/s straight reference')
axes[0,0].set(xlabel='World x (m)',ylabel='World y (m)',aspect='equal');axes[0,0].legend(fontsize=8)
if n==1:axes[0,0].set_ylim(-1,1)
axes[0,1].axhline(protocol.get('spec',{}).get('speed_m_s',2),color='gray',ls=':');axes[0,1].set(xlabel='Time (s)',ylabel='Forward speed (m/s)')
axes[1,0].set(xlabel='Time (s)',ylabel='Roll (deg)');axes[1,1].set(xlabel='Time (s)',ylabel='Pendulum angle (rad)');axes[1,1].legend(fontsize=7,ncol=2)
for ax in [axes[0,1],axes[1,0],axes[1,1]]:
 for key in ('warmup_s','return_s'):
  if key in protocol.get('spec',{}):ax.axvline(protocol['spec'][key],color='gray',ls=':',lw=.7)
f.suptitle('Isaac Lab ground control / ECBC and frozen Local300\nNominal DEV; controller configuration recorded in protocol; endpoints marked x')
f.savefig(a.run/'xy_response.png',dpi=160);plt.close(f)
(a.run/'RESULTS.md').write_text('# Isaac Lab pendulum ground diagnostic\n\n![Actual XY and response](xy_response.png)\n\n```json\n'+json.dumps(results,indent=2)+'\n```\n\nSingle nominal initialization, fixed hip/knee targets. No H75, training or jump handoff. Gray XY reference integrates requested speed and steering through the original bicycle kinematics; it is not a feasibility guarantee. Each method stops at its actual recorded endpoint. Raw states, applied references and commands remain in trajectory.npz; configuration in protocol.json.\n')
