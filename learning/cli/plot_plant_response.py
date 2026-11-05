"""Compare saved identical-command responses; no simulation or smoothing."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--candidate-label',default='Isaac Lab');p.add_argument('--run',type=Path,required=True);p.add_argument('--lab-name',default='lab');a=p.parse_args()
height=json.loads((a.run/'mujoco/status.json').read_text()).get('height_offset',0.);m=np.load(a.run/'mujoco/response.npz');l=np.load(a.run/a.lab_name/'response.npz');out=a.run if a.lab_name=='lab' else a.run/(a.lab_name+'_comparison');out.mkdir(exist_ok=True);implicit=json.loads((a.run/a.lab_name/'status.json').read_text())['identity'].get('servo_mode')=='implicit';np.testing.assert_array_equal(m['command'],l['command']);t=m['time'];fields=m['fields'].tolist();results=[]
fig,axs=plt.subplots(1,4,figsize=(17,4.5))
for i,case in enumerate(m['cases']):
 for d,label,color,style in [(m,'MuJoCo','#0072B2','--'),(l,a.candidate_label,'#D55E00','-')]:
  xy=d['values'][:,i,:2];axs[i].plot(*xy.T,style,color=color,label=label);axs[i].scatter(*xy[-1],marker='x',color=color)
 axs[i].set(title=case,xlabel='World x (m)',ylabel='World y (m)');axs[i].set_aspect('equal',adjustable='datalim');axs[i].grid(alpha=.2);axs[i].legend()
fig.suptitle(('Airborne diagnostic, initial height +'+str(height)+'m | ' if height else '')+'Plant response | identical velocity commands and physical initial state | no ECBC / residual / H75\n1 second, 5000 Hz; no geometric path target; crosses mark budget endpoints')
fig.tight_layout();fig.savefig(out/'xy.png',dpi=160);plt.close(fig)
for i,case in enumerate(m['cases']):
 fig,axs=plt.subplots(3,3,figsize=(15,10));cols=[4,7,8,5,6,3,9,10,2];units=['m/s','rad/s','rad/s','rad','rad/s','rad','Nm','Nm','rad']
 for ax,k,unit in zip(axs.flat,cols,units):
  for d,label,color,style in [(m,'MuJoCo','#0072B2','--'),(l,a.candidate_label,'#D55E00','-')]:ax.plot(t,d['values'][:,i,k],style,color=color,label=label,lw=.8)
  if k in [6,7]:ax.plot(t,m['command'][:,i,0 if k==6 else 1],':',color='.4',label='Requested rate')
  ax.axvspan(.1,.6 if i==1 else .3,color='.7',alpha=.15) if i else None
  ax.set(title=fields[k]+(' (Lab estimate only)' if implicit and k in [9,10] else ''),xlabel='Time (s)',ylabel=unit);ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.suptitle(str(case)+(' | airborne initial height +'+str(height)+'m' if height else '')+' | same input, no feedback | unsmoothed physics-step data\nActuator torques use original joint sign; rear reported rate is forward-positive')
 fig.tight_layout();fig.savefig(out/(str(case)+'_response.png'),dpi=150);plt.close(fig)
 result=dict(case=str(case),candidate_label=a.candidate_label,candidate_directory=a.lab_name,windows={})
 for end in [.005,.1,1.]:
  n=round(end/.0002);delta=l['values'][:n,i]-m['values'][:n,i]
  delta[:,2:4]=np.arctan2(np.sin(delta[:,2:4]),np.cos(delta[:,2:4]))
  result['windows'][str(end)]=dict(rmse=dict(zip(fields,np.sqrt(np.mean(delta**2,axis=0)).tolist())),mujoco_endpoint=dict(zip(fields,m['values'][n-1,i].tolist())),lab_endpoint=dict(zip(fields,l['values'][n-1,i].tolist())))
 result['last_half_second']={}
 for d,label in [(m,'mujoco'),(l,'lab')]:
  z=d['values'][2500:,i];result['last_half_second'][label]=dict(body_speed_mean=float(z[:,4].mean()),rear_speed_proxy_mean=float(z[:,7].mean()*.1),rear_rate_std=float(z[:,7].std()),steer_torque_saturated_fraction=float((abs(z[:,9])>=.999).mean()),rear_torque_saturated_fraction=float((abs(z[:,10])>=2.999).mean()))
 if implicit:
  result['last_half_second']['lab'].pop('steer_torque_saturated_fraction');result['last_half_second']['lab'].pop('rear_torque_saturated_fraction')
  result['torque_limitation']='Implicit Lab torque columns are prestep estimates, not solver applied torque.'
 results.append(result)
(out/'comparison.json').write_text(json.dumps(results,indent=2));print(json.dumps([dict(case=r['case'],last_half_second=r['last_half_second']) for r in results],indent=2))
