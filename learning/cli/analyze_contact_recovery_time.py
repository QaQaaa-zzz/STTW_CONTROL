"""Operational settling time relative to matched flat, for ridge-spacing design."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
# Declared descriptive tolerances, not a task-success or stability certificate.
limits=dict(roll_difference_rad=.03,speed_difference_m_s=.1,steer_rate_difference_rms_rad_s=.25,rms_window_s=.1,hold_s=.5)
lanes=json.loads((a.root/'lanes.json').read_text());flat=np.load(a.root/'flat/trajectory.npz');fw=np.load(a.root/'flat/physics_wrench.npz');rows=[]
dt=.005;window=round(limits['rms_window_s']/dt);hold=round(limits['hold_s']/dt)
def rolling_mean(x,k):
    total=np.r_[0.,np.cumsum(x)];return (total[k:]-total[:-k])/k
for terrain in ['oblique_4cm','oblique_6cm']:
    d=np.load(a.root/terrain/'trajectory.npz');w=np.load(a.root/terrain/'physics_wrench.npz');events=json.loads((a.root/terrain/'contact_event_windows.json').read_text())['lanes']
    trace=d['lower_trace'].reshape(-1,len(lanes),11);ft=flat['lower_trace'].reshape(trace.shape);m=d['measurement'];fm=flat['measurement'];alive=d['alive']&flat['alive']
    # Use each actual0.2ms steering rate in its5ms interval, then rolling100ms RMS.
    steer_difference=w['steer_velocity']-fw['steer_velocity'];steer_square=(steer_difference**2).reshape(len(trace),25,len(lanes)).mean(1)
    time=(np.arange(len(trace))+1)*dt
    for i,identity in enumerate(lanes):
        roll=abs(m[:,i,0]-fm[:,i,0]);speed=abs(trace[:,i,2]-ft[:,i,2]);steer=np.sqrt(rolling_mean(steer_square[:,i],window));tt=time[window-1:]
        first=events[i]['first_front_contact_s'];rear=events[i]['last_rear_contact_s']
        acceptable=(roll[window-1:]<=limits['roll_difference_rad'])&(speed[window-1:]<=limits['speed_difference_m_s'])&(steer<=limits['steer_rate_difference_rms_rad_s'])&alive[window-1:,i]&(tt>rear)
        holds=np.flatnonzero(rolling_mean(acceptable.astype(float),hold)>=1.-1e-10)
        confirmation=float(tt[holds[0]+hold-1]) if len(holds) else None
        rows.append(dict(terrain=terrain,**identity,first_front_contact_s=first,last_rear_contact_s=rear,settled_confirmation_s=confirmation,T_front_to_confirmed_settling_s=confirmation-first if confirmation is not None else None,rear_to_confirmed_settling_s=confirmation-rear if confirmation is not None else None,censored_at_s=float(time[-1]),interpretation='matched-flat state-response settling, not original-path recovery or goal arrival'))
report=dict(role='DEV operational recovery-time estimate for spacing; tolerances declared before calculation, no model fitting',limits=limits,rows=rows)
(a.root/'recovery_time.json').write_text(json.dumps(report,indent=2)+'\n')
for terrain in ['oblique_4cm','oblique_6cm']:
    group=[r for r in rows if r['terrain']==terrain];values=[r['T_front_to_confirmed_settling_s'] for r in group if r['T_front_to_confirmed_settling_s'] is not None]
    print(terrain,'settled',len(values),'/',len(group),'T median/min/max',np.median(values) if values else None,min(values) if values else None,max(values) if values else None)
