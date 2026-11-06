"""Align saved natural-branch contact vectors and controller responses; no physics."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
status=json.loads((a.run/'status.json').read_text());spec=json.loads((a.run/'protocol.json').read_text())
# Natural continuation is unaffected by the interventions in sibling lanes.
phase=next(iter(spec['fork_phases_m']));cases=[c for c in status['cases'] if c['phase']==phase]
out=a.run/'analysis/contact_alignment';out.mkdir(parents=True,exist_ok=True)
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#8A6500'];series=[];summary=[]
for case in cases:
 folder=Path(case['run']);d=np.load(folder/'trajectory.npz');tr=d['lower_trace'][:,:,0].reshape(-1,11)
 metrics=json.loads((a.run/'analysis'/phase/('alpha'+str(case['alpha']))/'metrics.json').read_text());contacts=metrics['branches'][0]['contacts']
 forces=d['wheel_contact_forces'][:,:,0].reshape(-1,2,d['wheel_contact_forces'].shape[-2],3)
 # Sum filtered partner vectors, then magnitude. This is not the sum of
 # individual contact-point force magnitudes if their normals differ.
 resultant=np.linalg.norm(forces.sum(axis=2),axis=-1)
 active=np.repeat(tr[:,6].astype(bool),25)&(np.tile(np.arange(25),len(tr))<np.repeat(tr[:,7],25))
 ht=(np.arange(len(forces))+1)*.0002;ids=np.flatnonzero(tr[:,6]);lt=ids*.005+tr[ids,7]*.0002
 row=dict(alpha=case['alpha'],contacts=contacts,run=str(folder),physical_samples=int(active.sum()),front_peak_resultant_N=float(resultant[active,0].max()),rear_peak_resultant_N=float(resultant[active,1].max()))
 summary.append(row);series.append(dict(alpha=case['alpha'],contacts=contacts,ht=ht[active],load=resultant[active],lt=lt,offsets=d['offsets'][ids,0],roll_rate=d['measurement'][ids,0,1],cross=tr[ids,3]))
 np.savez_compressed(out/('alpha'+str(case['alpha'])+'.npz'),physics_time=ht[active],normal_resultant_N=resultant[active],control_time=lt,offsets=d['offsets'][ids,0],roll_rate=d['measurement'][ids,0,1],cross_track=tr[ids,3])
for wheel in ('front','rear'):
 fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
 for c,color in zip(series,colors):
  t0=c['contacts'][wheel+'_first_s'];high=(c['ht']>=t0-.15)&(c['ht']<=t0+.8);low=(c['lt']>=t0-.15)&(c['lt']<=t0+.8)
  values=[c['load'][high,0],c['load'][high,1],c['offsets'][low,0],c['offsets'][low,1],c['roll_rate'][low],c['cross'][low]]
  for i,(ax,v) in enumerate(zip(axes.flat,values)):
   time=c['ht'][high] if i<2 else c['lt'][low];ax.plot(time-t0,v,color=color,label='alpha '+str(c['alpha']),lw=.9)
  for key,marker in [('front_first_s','>'),('front_last_s','<'),('rear_first_s','^'),('rear_last_s','v')]:
   t=c['contacts'][key]-t0
   axes[0,0].plot(t,.96,marker=marker,color=color,transform=axes[0,0].get_xaxis_transform(),markersize=5)
 for ax,ylabel in zip(axes.flat,['Front normal resultant (N)','Rear normal resultant (N)','Executed speed correction (m/s)','Executed steer correction (rad)','Roll rate (rad/s)','Signed cross-track (m)']):
  ax.set(xlabel='Time from '+wheel+' first raised contact (s)',ylabel=ylabel,xlim=(-.15,.8));ax.axvline(0,color='.4',ls=':');ax.grid(alpha=.2)
 axes[0,0].legend(ncol=3,fontsize=8);fig.suptitle('Original H75 / Local300 | natural branches, all five alpha | '+wheel+' contact aligned\nNormal resultant at0.2ms; control response at5ms; no smoothing; one initial state, DEV')
 fig.savefig(out/(wheel+'_aligned.png'),dpi=150);plt.close(fig)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
(out/'RESULTS.md').write_text('''# Contact-phase aligned natural responses

[Actual XY / paired intervention evidence](../RESULTS.md)

![Front aligned](front_aligned.png)
![Rear aligned](rear_aligned.png)

Original H75 u77 and selected Local300; natural lane0 from each entry-phase run, all five alpha. Sibling interventions do not define these natural trajectories. Each alpha reaches its own physical/contact history; aligning events does not make these states identical. Common-state hold/zero interventions remain separate causal evidence.

Saved `wheel_contact_forces` comes from `ContactSensor.data.force_matrix_w`: the installed IsaacLab contact_sensor_data.py specifies filtered NORMAL force vectors in world coordinates, not total friction-inclusive contact force. Installed IsaacLab commit090aed181; the sensor calls get_contact_force_matrix(dt=0.0002). Matching version documentation: https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.sensors.html . Newer main-branch API semantics differ; this report is tied to the installed source and recorded run.

Plot quantity: norm(sum of filtered partner normal vectors), in N. This wheel-level resultant is not the sum of individual contact-point scalar loads when normals differ. Existing Fz plots show the vertical component only. No terrain-normal projection is inferred from a mean contact position. Forces are raw0.2ms samples; controller/roll-rate signals are5ms. No smoothing or interpolation. Top-left >/< mark first/last raised front contact; ^/v mark rear contact, each alpha in its own color. Markers at the top indicate times, not force values. First/last raised contact follows the existing mean-point, positive-terrain-height,2.5mm-z and1mN threshold convention; the interval may contain intermittent contact.

Derived arrays and full-window peak resultants are saved as alpha*.npz and summary.json. Original trajectories and all negative outcomes are unchanged. These plots describe contact timing and response; alone they do not prove a channel mechanism or generalization.
''')
print(json.dumps(summary))
