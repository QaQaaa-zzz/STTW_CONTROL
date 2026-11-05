"""Five-alpha fixed-path engineering panel using saved trajectories only."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);a=p.parse_args()
route=np.load(a.bundle/'route.npz');fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');colors=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00'];summary={};base=None
for name,label,color in [('baseline','Zero middle + LOCAL300','black')]+[(f'alpha{x}',f'H75 alpha={x}',co) for x,co in zip([0,.25,.5,.75,1],colors)]:
 receipt=json.loads((a.run/(name+'.json')).read_text());d=np.load(a.run/(name+'.npz'));states=d['body_states'][:,0];c=d['control_trace'];middle=d['middle_trace']
 if base is None:base=receipt
 for key in ('asset_sha256','controller_config','drive_speed','ground','dt_s','initial_states','initial_dofs','shape_properties','lower_actor'):
  if receipt[key]!=base[key]:raise ValueError('pair mismatch '+key)
 if receipt['path_contract']['bundle']!=base['path_contract']['bundle']:raise ValueError('route/policy bundle differs')
 axes[0,0].plot(states[:,0],states[:,1],color=color,label=label);axes[0,0].plot(states[-1,0],states[-1,1],'x',color=color)
 axes[0,1].plot(c[:,0],c[:,11],color=color,label=label)
 axes[1,0].plot(middle[:,0],middle[:,6],color=color,label=label)
 axes[1,1].plot(c[:,0],c[:,1],color=color,label=label)
 summary[name]=dict(termination=receipt['termination'],seconds=receipt['physics_steps']*receipt['dt_s'],physics_steps=receipt['physics_steps'],final_xy=states[-1,:2].tolist(),max_abs_roll=float(np.abs(c[:,1]).max()),max_abs_cross=float(np.abs(middle[:,6]).max()),mean_true_speed=float(c[:,11].mean()),alpha=receipt['path_contract']['alpha'],zero_middle=receipt['path_contract']['zero_middle'])
segment=route['s']<=12;axes[0,0].plot(route['xy'][segment,0],route['xy'][segment,1],'k--',label='Original fixed route (first 12 m)');axes[0,0].set_aspect('equal',adjustable='datalim');axes[0,0].set(xlabel='World X [m]',ylabel='World Y [m]',title='Actual/reference XY; x = actual endpoint');axes[0,0].legend(fontsize=7)
axes[0,1].axhline(base['drive_speed'],color='grey',linestyle='--');axes[0,1].set(xlabel='Time [s]',ylabel='True forward speed [m/s]')
axes[1,0].set(xlabel='Time [s]',ylabel='Signed cross-track error [m]');axes[1,1].set(xlabel='Time [s]',ylabel='Left-positive roll [rad]')
for ax in axes.ravel():ax.grid(alpha=.2)
fig.suptitle('Isaac fixed-path engineering | frozen H75 U77 / LOCAL300 | one initial state\nFive-alpha coverage; no training or task-success claim; endpoint reasons in summary')
fig.savefig(a.run/'comparison.png',dpi=160);(a.run/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
