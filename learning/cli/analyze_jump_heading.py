"""Compare actual jump/handback XY and body heading; retain every batch lane."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runs',nargs='+',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
records=[]
for run in a.runs:
 spec=json.loads((run/'protocol.json').read_text())['spec']
 mount=json.loads(Path(spec['model_config']).read_text())['mount_rotation_wxyz']
 rotation=Rotation.from_quat(mount[1:]+mount[:1])
 fig,axes=plt.subplots(3,2,figsize=(13,12),layout='constrained')
 for lane,d in enumerate(sorted(run.glob('lane_*'))):
  data=np.load(d/'trajectory.npz');mask=data['valid'][:,0];trace=data['trace'][mask,0]
  body=np.load(d/'body_contact.npz');root=body['body_state'][mask,0,list(body['body_names']).index('chasis')]
  yaw=(Rotation.from_quat(root[:,3:7])*rotation).as_euler('xyz')[:,2]
  model=spec['skills'][lane];col=0 if lane<len(spec['skills'])//2 else 1
  candidate=spec['batch_candidates'][lane];label=f"v={candidate['speed_m_s']}, alpha={candidate['alpha']}"
  color={0:'#0072B2',.5:'#D55E00',1:'#009E73'}[candidate['alpha']]
  style='-' if candidate['speed_m_s']==2 else '--'
  route=np.load(d/'ground_route.npz')['xy']
  axes[0,col].plot(trace[:,1],trace[:,2],style,color=color,label=label)
  axes[0,col].plot(trace[-1,1],trace[-1,2],'x',color=color)
  axes[1,col].plot(trace[:,0],np.rad2deg(yaw),style,color=color)
  axes[2,col].plot(trace[:,0],trace[:,2],style,color=color)
  entry=dict(run=str(run),lane=lane,model=model,candidate=candidate,events={})
  for name,marker in [('handoff','o'),('landing_handoff','s')]:
   f=d/(name+'.json')
   if f.exists():
    event=json.loads(f.read_text());idx=np.argmin(abs(trace[:,0]-event['time']))
    entry['events'][name]=dict(time_s=float(trace[idx,0]),y_m=float(trace[idx,2]),yaw_deg=float(np.rad2deg(yaw[idx])),vy_m_s=float(root[idx,8]))
    axes[0,col].plot(trace[idx,1],trace[idx,2],marker,color=color,ms=5)
    axes[1,col].plot(trace[idx,0],np.rad2deg(yaw[idx]),marker,color=color,ms=5)
  if (d/'handoff.json').exists():
   selected=trace[:,0]>=entry['events']['handoff']['time_s']
   entry['post_jump']=dict(max_abs_y_m=float(abs(trace[selected,2]).max()),max_abs_yaw_deg=float(np.rad2deg(abs(yaw[selected])).max()),final_y_m=float(trace[-1,2]),final_yaw_deg=float(np.rad2deg(yaw[-1])),end_time_s=float(trace[-1,0]))
  records.append(entry);axes[0,col].set_title(model)
 for col in range(2):
  axes[0,col].plot(route[:,0],route[:,1],':',color='black',label='Original route');axes[0,col].set_aspect('equal',adjustable='datalim');axes[0,col].set(xlabel='World x (m)',ylabel='World y (m)');axes[0,col].legend(fontsize=7)
  axes[1,col].set(xlabel='Time (s)',ylabel='Body yaw (deg)')
  axes[2,col].set(xlabel='Time (s)',ylabel='World y (m); jump centreline y=0')
 for ax in axes.flat:ax.grid(alpha=.25)
 fig.suptitle(f'{run.name}: all lanes, actual failure endpoints\nCircle: jump handoff; square: ground handback; cross: observation end')
 fig.savefig(a.output/(run.name+'.png'),dpi=140);plt.close(fig)
(a.output/'heading.json').write_text(json.dumps(records,indent=2))
