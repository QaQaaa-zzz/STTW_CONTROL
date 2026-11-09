"""Paired frozen-seed random-terrain assessment; never select on confirmation data."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();a.run=a.run.resolve()
routes=np.load('runs/h75_original_training_portable/routes.npz');sets={k:np.load(a.run/k/'trajectory.npz') for k in ['before','after']}
assert np.array_equal(sets['before']['route_ids'],sets['after']['route_ids'])
assert np.array_equal(sets['before']['terrain'],sets['after']['terrain'])
report=a.run/'analysis';report.mkdir(exist_ok=True);rows=[];links=[]
for lane,route_id in enumerate(sets['before']['route_ids']):
 fig,axs=plt.subplots(1,3,figsize=(14,4));ref=routes['xy'][route_id];goal=routes['goal'][route_id];heading=np.interp(goal,routes['s'],routes['heading'][route_id]);goalxy=np.array([np.interp(goal,routes['s'],ref[:,j]) for j in range(2)]);axs[0].plot(*ref.T,':',color='gray',label='Original reference')
 for name,d in sets.items():
  tr=d['lower_trace'][:,:,lane].reshape(-1,11);valid=tr[:,6].astype(bool);ids=np.flatnonzero(valid);time=ids*.005+tr[ids,7]*.0002;speed=float(routes['speed'][route_id]);failed=bool(d['failure'][:,lane].any());sig=[]
  if failed:
   step=int(np.flatnonzero(d['failure'][:,lane])[0]);sl=slice(step*4,(step+1)*4)
   if d['domain_exit'][step,lane]:sig.append('tracking_domain')
   if d['policy_fault'][step,lane]:sig.append('policy_fault')
   if (tr[sl,4]>.7).any():sig.append('roll_limit')
   if not sig:sig.append('unresolved_contact_or_other')
  ps=np.unique(ids//4);pose=d['pose'][ps,lane];progress=d['progress'][ps,lane];signed=(pose[:,:2]-goalxy)@np.array([np.cos(heading),np.sin(heading)])
  rows.append(dict(method=name,lane=lane,route_id=int(route_id),terrain=int(d['terrain'][lane]),alpha=float(d['alpha'][lane]),observed_s=float(time[-1]),termination_signals=sig,goal_crossed=bool(((signed>=0)&(progress>=goal)).any()),terrain_entered=bool(d['terrain_entered'][lane]),raised_contact_observed=(None if name=='before' and (a.run/'recovery.json').exists() else bool(d['terrain_contacted'][lane])),terrain_cleared=bool(d['terrain_cleared'][lane]),path_peak_m=float(abs(tr[valid,3]).max()),path_rmse_m=float(np.sqrt(np.mean(tr[valid,3]**2))),speed_rmse_m_s=float(np.sqrt(np.mean((tr[valid,2]-speed)**2)))))
  axs[0].plot(tr[valid,0],tr[valid,1],label=name);axs[0].scatter(tr[ids[-1],0],tr[ids[-1],1],marker='x');axs[1].plot(ps,d['reward'][ps,lane],label=name);axs[2].plot(time,tr[valid,2]-speed,label=name)
 axs[0].set_aspect('equal');axs[0].set(xlabel='x (m)',ylabel='y (m)');axs[1].set(xlabel='policy step',ylabel='reward incl. terminal penalty');axs[2].set(xlabel='time (s)',ylabel='speed error (m/s)')
 for ax in axs:ax.legend();ax.grid(alpha=.2)
 fig.suptitle(f'Frozen confirmation lane{lane}, route{route_id}, alpha{sets["before"]["alpha"][lane]:g}; actual endpoints; not training seeds');fig.tight_layout();fig.savefig(report/f'lane_{lane:03d}.png',dpi=120);plt.close(fig);links.append(f'- [lane{lane} paired XY/reward/speed](lane_{lane:03d}.png)')
(report/'metrics.json').write_text(json.dumps(rows,indent=2));(report/'RESULTS.md').write_text('# Random-terrain paired confirmation\n\nOriginal H75 vs final150; no selection on confirmation. Same sampled routes,terrain and RNG seed; Go2 randomization declared in configs. Before contact flags are unknown when recovery.json records the sensor filter error. Contact is20ms sampled and can miss short events. Early termination, domain exit, contact exposure and continuous errors are separate; no new A/B/C gate.\n\n'+'\n'.join(links)+'\n\n[Per-lane metrics](metrics.json)\n')
