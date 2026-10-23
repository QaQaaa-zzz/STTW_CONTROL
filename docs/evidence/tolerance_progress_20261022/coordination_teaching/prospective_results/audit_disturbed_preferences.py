from pathlib import Path
import json,hashlib
import numpy as np
R=Path('runs/preference_coordination_prospective_v1');decl=json.loads((R/'declaration.json').read_text());output=[];sources=[]
for panel in decl['panels']:
 folder=R/panel;d=json.loads((folder/'declaration.json').read_text());cfg=d['config']['path'];lo=cfg['straight_prefix_m'];hi=lo+abs(cfg['turn_angle_rad'])*cfg['turn_radius_m']+cfg['curvature_transition_length_m'];rs=json.loads((folder/'evaluation/results.json').read_text())['cases']
 for pol in ('parent','seed918501'):
  rows=[x for x in rs if x['policy']==pol and x['force_N']!=0]
  keys=sorted(set((x['route'],x['speed'],x['force_N'],x['onset_s']) for x in rows))
  for key in keys:
   group=sorted([x for x in rows if (x['route'],x['speed'],x['force_N'],x['onset_s'])==key],key=lambda x:x['alpha']);assert len(group)==4
   vals=[]
   for x in group:
    p=folder/'evaluation'/(x['name']+'.npz')
    with np.load(p) as z:
     mask=z['time']>=x['onset_s']+d['duration_s'];full=bool(len(z['time'])==2000 and not x['domain_failure'] and not x['physical_failure'])
     vals.append(dict(alpha=x['alpha'],task=x['task_recovered'],full_horizon=full,turn_ticks=int(mask.sum()),speed_rmse=float(np.sqrt(np.mean((z['actual_forward_speed'][mask]-x['speed'])**2))) if mask.any() else None,path_rmse=float(np.sqrt(np.mean(z['path_cross_track'][mask]**2))) if mask.any() else None,peak_roll=x['peak_roll_rad']))
    sources.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
   eligible=all(v['full_horizon'] and v['turn_ticks'] for v in vals)
   output.append(dict(panel=panel,policy=pol,context=key,values=vals,eligible_full_horizon=eligible,all_task_success=all(v['task'] for v in vals),speed_monotone=all(a['speed_rmse']>b['speed_rmse'] for a,b in zip(vals,vals[1:])) if eligible else None,speed_endpoint=vals[0]['speed_rmse']>vals[-1]['speed_rmse'] if eligible else None,path_endpoint=vals[0]['path_rmse']<vals[-1]['path_rmse'] if eligible else None))
summary=[]
for pol in ('parent','seed918501'):
 g=[x for x in output if x['policy']==pol];ok=[x for x in g if x['eligible_full_horizon']]
 summary.append(dict(policy=pol,total_contexts=len(g),full_horizon_contexts=len(ok),excluded_early_end=len(g)-len(ok),all_task_success=sum(x['all_task_success'] for x in g),speed_monotone=sum(x['speed_monotone'] for x in ok),speed_endpoint=sum(x['speed_endpoint'] for x in ok),path_endpoint=sum(x['path_endpoint'] for x in ok)))
p=Path('docs/evidence/tolerance_progress_20261022/coordination_teaching/prospective_results/disturbed_preference_audit.json');p.write_text(json.dumps(dict(summary=summary,contexts=output,sources=sources,scope='Exploratory existing pulse panel. Common recovery-time window after pulse end through original deadline; includes bend and outgoing straight. Full-horizon coverage reported; not restricted to task successes. Missing early ends not imputed as preference success.'),indent=2)+'\n');print(json.dumps(summary,indent=2))
# Matched complete-context view: same denominator across policies.
k=lambda r:(r['panel'],tuple(r['context']))
a={k(r):r for r in output if r['policy']=='parent'}
b={k(r):r for r in output if r['policy']=='seed918501'}
ks=[x for x in a if a[x]['eligible_full_horizon'] and b[x]['eligible_full_horizon']]
d=json.loads(p.read_text())
d['matched_full_horizon']=dict(n=len(ks),parent={m:sum(a[x][m] for x in ks) for m in ['speed_monotone','speed_endpoint','path_endpoint']},candidate={m:sum(b[x][m] for x in ks) for m in ['speed_monotone','speed_endpoint','path_endpoint']},contexts=[list(x) for x in ks])
p.write_text(json.dumps(d,indent=2)+'\n')
