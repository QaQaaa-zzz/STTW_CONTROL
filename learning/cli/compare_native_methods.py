from pathlib import Path
import json,hashlib,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
parser=argparse.ArgumentParser(description='Matched all-method XY comparison from completed native runs; no simulation.')
parser.add_argument('manifest',type=Path)
parser.add_argument('output',type=Path)
args=parser.parse_args();declaration=json.loads(args.manifest.read_text())
r=Path(declaration['reference_run']);a=r.parent;out=args.output;out.mkdir(parents=True,exist_ok=False)
specs=[(x['label'],x['run'],x.get('alpha')) for x in declaration['methods']]
expected=declaration['conditions_per_method']
base=json.loads((r/'plan.json').read_text());tables=[];provenance=[]
for label,folder,alpha in specs:
 root=a/folder;p=json.loads((root/'plan.json').read_text());receipt=json.loads((root/'analysis/receipt.json').read_text())
 assert receipt['plan_sha256']==hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()
 assert json.loads((root/'evaluation/status.json').read_text())['phase']=='completed'
 for key in ('models','config','initial_source','contexts','replanning','horizon_s'):assert p[key]==base[key],key
 assert p['protocol']['reconnection']==base['protocol']['reconnection']
 rows=json.loads((root/'analysis/results.json').read_text());rows=[x for x in rows if alpha is None or x['alpha']==alpha]
 table={(x['bank_index'],x['route'],x['force_N']):x for x in rows};assert len(rows)==len(table)==expected
 tables.append(table);provenance.append(dict(label=label,root=str(root),alpha_filter=alpha,plan_sha256=receipt['plan_sha256']))
keys=sorted(tables[0]);assert all(set(t)==set(keys) for t in tables)
summary=[];details=[];colors=['#777777','#E69F00','#009E73','#CC79A7','#D55E00','#0072B2'];styles=[':','--','-.',(0,(3,1,1,1)),'--','-'];assert len(specs)<=len(colors)
for k,key in enumerate(keys):
 rows=[t[key] for t in tables];ref=rows[-1]
 for x in rows:
  for field in (('speed','onset_s','duration_s','original_goal_xy') if key[2] else ('speed','original_goal_xy')):assert x[field]==ref[field],field
 with np.load(r/'evaluation'/(ref['path_key']+'_path.npz')) as path:original=path['xy'][path['s']<=path['goal']].copy()
 fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained');axes[0].plot(*original.T,'k:',label='Original reference');axes[0].add_patch(Circle(ref['original_goal_xy'],1.,fill=False,ls=':',ec='black'));axes[0].scatter(*ref['original_goal_xy'],color='black',marker='*')
 for n,(label,_,_),row,color,style in zip(range(len(specs)),specs,rows,colors,styles):
  end=row.get('endpoint',row.get('horizons',{}).get('10'));raw=Path(row['raw_file']);assert hashlib.sha256(raw.read_bytes()).hexdigest()==row['raw_sha256']
  with np.load(raw) as z:
   if not key[2]:assert np.all(z['applied_force_impulse']==0.)
   active=z['active_tick'].astype(bool);xy=z['actual_xy'][active];t=z['time'][active];v=z['actual_forward_speed'][active];alpha=z['alpha_exec'][active] if 'alpha_exec' in z else np.full(len(t),row['alpha'])
   pulse_mask=np.any(z['applied_force_impulse'][active].reshape(len(t),-1)!=0.,axis=1)
  arrival_stop=int(np.clip(round((end['first_arrival_s']-float(t[0]))/.005),0,len(t))) if end['first_arrival_s'] is not None else len(t)
  axes[0].plot(*xy.T,color=color,ls=style,lw=1.7,label=label);axes[0].scatter(*xy[-1],color=color,marker='s' if end['physical_outcome']=='survived' else 'x',s=22)
  axes[1].plot(t,v,color=color,ls=style,lw=1.4);axes[2].step(t,alpha,where='post',color=color,ls=style,lw=1.4)
  details.append(dict(method=label,route=key[1],force_N=key[2],bank=key[0],endpoint=end,raw_file=str(raw),raw_sha256=row['raw_sha256'],requested_speed_m_s=row['speed'],observed_s=len(t)*.005,actual_speed_min_m_s=float(v.min()),actual_speed_max_m_s=float(v.max()),actual_speed_mean_m_s=float(v.mean()),pulse_speed_m_s=dict(min=float(v[pulse_mask].min()),max=float(v[pulse_mask].max())) if key[2] and np.any(pulse_mask) else None,before_goal_user0_fraction=float(np.mean(alpha[:arrival_stop]==0.)) if arrival_stop else None,original_user_three_running_costs=row['original_user_three_running_costs']))
 axes[0].set(xlabel='World X (m)',ylabel='World Y (m)');axes[0].set_aspect('equal',adjustable='datalim');axes[0].legend(fontsize=8)
 axes[1].set(xlabel='Task time (s)',ylabel='True forward speed (m/s)');axes[2].set(xlabel='Task time (s)',ylabel='Policy preference alpha')
 for ax in axes:ax.grid(alpha=.2)
 if key[2]:
  for ax in axes[1:]:ax.axvspan(ref['onset_s'],ref['onset_s']+ref['duration_s'],color='gray',alpha=.15)
 fig.suptitle(f'{key[1]} | bank{key[0]} | worldY {key[2]:+g}N | U170/local200 | original goal disk1m | H10')
 fig.savefig(out/f'case_{k:02d}_xy.png',dpi=150);plt.close(fig)
for label,_,_ in specs:
 ends=[x['endpoint'] for x in details if x['method']==label]
 summary.append(dict(method=label,n=len(ends),arrivals=sum(x['goal_arrived'] for x in ends),survivals=sum(x['physical_outcome']=='survived' for x in ends),joint=sum(x['goal_arrived'] and x['physical_outcome']=='survived' for x in ends)))
(out/'comparison.json').write_text(json.dumps(dict(summary=summary,conditions=details,provenance=provenance,role=declaration['data_role'],task_cost_columns=['speed','path','primary'],window_semantics='Pulse: actual nonzero applied impulse transfers; pre-goal occupancy: integer control transitions through first arrival, excluding subsequent actions.',cost_scope='Per-source user alpha, actual observed window; no terminal replacement included. Do not compare different-alpha weighted costs.'),indent=2)+'\n')
lines=['# Matched physical comparison','', 'All methods share model, physics, initial bank, path, event, H10 and4m/10Hz reconnection. Original1m goal disk and full-window survival are separate. Fixedalpha1 is a preference endpoint comparator. Alpha curves show the configured fixed preference for fixed controllers and the published selector value for online controllers; native permission/warmup may withhold policy action. No smoothed/resampled trajectory; actual failure endpoints retained. Existing per-run reward diagnostics are linked below; different-alpha weighted rewards are not ranked.','','| Method | Arrivals | Survivals | Arrival and survival |','|---|---:|---:|---:|']
for x in summary:lines.append(f"| {x['method']} | {x['arrivals']}/{x['n']} | {x['survivals']}/{x['n']} | {x['joint']}/{x['n']} |")
for k in range(len(keys)):lines+=['',f'![All-method actual/reference XY, speed and alpha](case_{k:02d}_xy.png)']
lines+=['','## Source diagnostics']
for x in provenance:lines+=['',f"[{x['label']}]({x['root']}/analysis/README.md)"]
lines+=['',declaration['data_role'],'','Actual observed speed ranges and durations are saved per case in comparison.json. Requested-speed strata alone do not establish the physical speed regime. Original-user costs use each recorded preference and observation window; do not rank fixed1 against user0 by these costs.']
(out/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
