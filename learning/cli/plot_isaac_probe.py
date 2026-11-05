"""Plot raw short engineering trajectories, not a learned-policy evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--receipt', type=Path, required=True)
parser.add_argument('--baseline', type=Path)
args = parser.parse_args()
p = json.loads(args.receipt.read_text())
method = 'LOCAL300 + ECBC/ESO' if 'lower_actor' in p else 'ECBC/ESO'
d = np.load(args.receipt.with_suffix('.npz'))
states = d['body_states'][:, 0]
control = d['control_trace']
time = np.arange(1, len(states)+1)*float(d['dt'])
initial = np.asarray(p['initial_states'][0][0][0])
yaw = control[0, 10]
reference = initial[:2]+p['drive_speed']*time[:, None]*np.array([np.cos(yaw), np.sin(yaw)])
fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
axes[0, 0].plot(reference[:, 0], reference[:, 1], 'k--', label='Original straight command integral')
axes[0, 0].plot(states[:, 0], states[:, 1], color='#0072B2', label=method+' actual')
axes[0, 0].plot(states[-1, 0], states[-1, 1], 'o', color='#D55E00', label=p['termination'])
axes[0, 0].set(xlabel='World X [m]', ylabel='World Y [m]', title='Actual / original reference XY')
axes[0, 0].set_aspect('equal', adjustable='datalim')
axes[0, 0].legend(fontsize=7)
axes[0, 1].plot(control[:, 0], control[:, 11], color='#0072B2', label='True forward speed')
axes[0, 1].plot(control[:, 0], control[:, 6]*.1, '--', color='#D55E00', label='Rear wheel proxy')
axes[0, 1].axhline(p['drive_speed'], color='k', linestyle=':', label='Original request')
axes[0, 1].set(xlabel='Time [s]', ylabel='Speed [m/s]')
axes[0, 1].legend(fontsize=8)
axes[1, 0].plot(control[:, 0], control[:, 1], color='#0072B2', label='Left-positive roll')
axes[1, 0].plot(control[:, 0], control[:, 3], '--', color='#D55E00', label='Steer angle')
axes[1, 0].set(xlabel='Time [s]', ylabel='Angle [rad]')
axes[1, 0].legend(fontsize=8)
axes[1, 1].plot(time, states[:, 2], color='#0072B2')
axes[1, 1].set(xlabel='Time [s]', ylabel='Chassis height [m]')
if args.baseline:
    baseline = json.loads(args.baseline.read_text())
    for field in ('asset_sha256','controller_config','drive_speed','ground','dt_s','initial_states','initial_dofs','shape_properties'):
        if p[field] != baseline[field]:
            raise ValueError('baseline mismatch: '+field)
    raw = np.load(args.baseline.with_suffix('.npz'))
    b = raw['body_states'][:,0]; c = raw['control_trace']
    axes[0,0].plot(b[:,0],b[:,1],color='#009E73',label='Matched ECBC/ESO')
    axes[0,0].plot(b[-1,0],b[-1,1],'x',color='#009E73')
    axes[0,1].plot(c[:,0],c[:,11],color='#009E73',label='ECBC true speed')
    axes[1,0].plot(c[:,0],c[:,1],color='#009E73',label='ECBC roll')
    axes[1,1].plot(np.arange(1,len(b)+1)*float(raw['dt']),b[:,2],color='#009E73',label='ECBC height')
    for ax in axes.ravel():ax.legend(fontsize=7)
for ax in axes.ravel():
    ax.grid(alpha=.2)
fig.suptitle('Isaac '+p['ground']+' engineering probe | '+method+' | one initial state\n'
             'No external disturbance; no task reward; ESO correction starts after 3 s')
fig.savefig(args.receipt.with_suffix('.png'), dpi=160)
summary = dict(explicit_physics_steps=p['physics_steps'], observed_seconds=float(time[-1]),
               termination=p['termination'], finite=p['finite'], final_xy_m=states[-1, :2].tolist(),
               max_abs_roll_rad=float(np.abs(control[:, 1]).max()),
               speed_min_m_s=float(control[:, 11].min()), speed_max_m_s=float(control[:, 11].max()),
               max_abs_original_cross_m=float(np.abs((states[:, :2]-initial[:2])@np.array([-np.sin(yaw), np.cos(yaw)])).max()),
               source_npz=str(args.receipt.with_suffix('.npz')), smoothing='none',
               scope='One short engineering rollout, not offroad recovery or task success')
args.receipt.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
