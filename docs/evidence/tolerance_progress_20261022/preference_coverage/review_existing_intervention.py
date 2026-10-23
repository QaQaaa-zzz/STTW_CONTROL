from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parent;run=R.parent/'preference_control_intervention_v1/evaluation'
rows=json.loads((run/'results.json').read_text())['cases'];data=[]
key=lambda r:tuple(r[k] for k in ('route','speed','alpha','force_N','onset_s'))
for r in rows:
 if not r['force_N']:continue
 with np.load(run/(r['name']+'.npz')) as z:
  t=z['time'];active=z['active_tick'].astype(bool);m=(t>=4.1)&active
  tail=(t>=9)&active
  data.append(dict(policy=r['policy'],context=key(r),task=r['task_recovered'],domain=r['domain_failure'],late_heading_rms=float(np.sqrt(np.mean(z['path_heading_error'][m]**2))),late_path_rms=float(np.sqrt(np.mean(z['path_cross_track'][m]**2))),tail_heading_rms=float(np.sqrt(np.mean(z['path_heading_error'][tail]**2))) if tail.any() else None,rate_clipped_fraction=float(np.mean(z['correction_rate_clipped'][m])),reference_clip_fraction=np.mean(z['reference_clip_channels'][m],axis=0).tolist()))
idx={(r['policy'],tuple(r['context'])):r for r in data};pairs=[]
for r in data:
 if r['policy']=='control':continue
 b=idx['control',tuple(r['context'])];pairs.append(dict(policy=r['policy'],context=r['context'],outcome='improved' if r['task'] and not b['task'] else 'regressed' if b['task'] and not r['task'] else 'unchanged',domain=r['domain'],heading_rms_change=r['late_heading_rms']-b['late_heading_rms'],path_rms_change=r['late_path_rms']-b['late_path_rms']))
(R/'existing_intervention_diagnostic.json').write_text(json.dumps(dict(source=str(run),windows='late 4.1 to actual endpoint; tail 9 to actual endpoint. Domain exits have shortened windows and are not comparable full-horizon RMS.',cases=data,paired=pairs),indent=2)+'\n')
for a in ('speed_off','steer_off'):
 for o in ('improved','regressed'):
  g=[r for r in pairs if r['policy']==a and r['outcome']==o];print(a,o,len(g),'heading changes',[round(r['heading_rms_change'],4) for r in g],'path changes',[round(r['path_rms_change'],4) for r in g])
g=[r for r in data if r['policy']=='control'];print('control late rate clipped', [round(r['rate_clipped_fraction'],3) for r in g]);print('control reference clips',[r['reference_clip_fraction'] for r in g])
