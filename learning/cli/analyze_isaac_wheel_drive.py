"""Compare requested, actuator-limited, measured wheel speed and chassis motion."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
d=np.load(a.run/'trajectory.npz');protocol=json.loads((a.run/'protocol.json').read_text());contract=json.loads(Path(protocol['asset']).with_name('contract.json').read_text());act=next(x for x in contract['actuators'] if x['name']=='cmd_rearwheel_v');radius=protocol['plant']['wheel_radius_m'];cases=protocol['spec']['cases'];n=len(cases)
f,axes=plt.subplots(n,3,figsize=(15,3.2*n),squeeze=False,layout='constrained');summary=[]
for i,case in enumerate(cases):
 keep=d['valid'][:,i];z=d['trace'][keep,i];t=z[:,0];request=d['command'][keep,i,1];target=-np.clip(-request,*act['ctrlrange']);wheel=z[:,13]/radius;v=z[:,4]
 axes[i,0].plot(t,request,label='Requested rear rate');axes[i,0].plot(t,target,label='Limited target');axes[i,0].plot(t,wheel,label='Measured rear rate',ls='--');axes[i,0].set(ylabel='Forward-positive wheel rate (rad/s)',title=case['name'])
 axes[i,1].plot(t,d['reference'][keep,i,0],label='Requested chassis speed');axes[i,1].plot(t,target*radius,label='Wheel target × radius');axes[i,1].plot(t,wheel*radius,label='Measured wheel × radius');axes[i,1].plot(t,v,label='Actual chassis speed',lw=2);axes[i,1].set(ylabel='Speed (m/s)')
 axes[i,2].plot(z[:,1],z[:,2],label='Actual XY');axes[i,2].plot(z[-1,1],z[-1,2],'x');axes[i,2].set(xlabel='World x (m)',ylabel='World y (m)',aspect='equal')
 # Commands are saved as the held command that produced each post-step sample.
 mask=t>=.1;row=dict(name=case['name'],duration_s=float(t[-1]),wheel_tracking_rmse_rad_s=float(np.sqrt(np.mean((wheel[mask]-target[mask])**2))),wheel_chassis_speed_gap_mean_m_s=float(np.mean(wheel[mask]*radius-v[mask])),samples=[])
 for when in (.4,.8,1.2,1.6,2.,2.4,2.8,3.2):
  if when>t[-1]:continue
  k=int(np.argmin(abs(t-when)));row['samples'].append(dict(time_s=float(t[k]),requested_wheel_rad_s=float(request[k]),limited_wheel_rad_s=float(target[k]),measured_wheel_rad_s=float(wheel[k]),actual_chassis_m_s=float(v[k]),wheel_equivalent_m_s=float(wheel[k]*radius)))
 summary.append(row)
 for j in range(3):axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=7)
 for j in (0,1):axes[i,j].set_xlabel('Time (s)')
f.suptitle('Rear drive input/output audit — original motor bounds retained\nWheel circumferential speed is not chassis speed; no native drive-torque measurement claimed')
f.savefig(a.run/'wheel_drive_input_output.png',dpi=150);plt.close(f)
(a.run/'wheel_drive_audit.json').write_text(json.dumps(dict(actuator=act,radius_m=radius,cases=summary,scope='Target reconstructed from executed plant clamp; actual wheel/chassis values from recorded states; no inference of motor torque saturation from velocity saturation'),indent=2)+'\n')
