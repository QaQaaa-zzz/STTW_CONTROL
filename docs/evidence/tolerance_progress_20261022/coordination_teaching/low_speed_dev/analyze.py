from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;run=R/'evaluation';cache=json.loads((R.parent/'preference_coverage_post_v1/cached_controls.json').read_text())['low_speed']['parent'];base=Path(cache['receipt']['run'])
assert json.loads((run/'status.json').read_text())['phase']=='completed'
rows=json.loads((run/'results.json').read_text())['cases'];parents=cache['rows']
assert len(rows)==len(parents)==60
key=lambda r:tuple(r[k] for k in ('route','speed','alpha','force_N','onset_s'))
idx={key(r):r for r in parents};assert set(idx)=={key(r) for r in rows}
pairs=[];summaries=[]
for name,g in [('parent',parents),('taught',rows)]:
 summaries.append(dict(policy=name,task=sum(r['task_recovered'] for r in g),working=sum(r['performance_success'] for r in g),domain=sum(r['domain_failure'] for r in g),nominal=sum(r['task_recovered'] for r in g if r['force_N']==0),disturbed=sum(r['task_recovered'] for r in g if r['force_N']!=0)))
fig,axes=plt.subplots(12,5,figsize=(25,48),layout='constrained')
for ax,r in zip(axes.flat,sorted(rows,key=key)):
 b=idx[key(r)];pair=dict(context=key(r),baseline_task=b['task_recovered'],task=r['task_recovered'],improved=bool(r['task_recovered'] and not b['task_recovered']),regressed=bool(b['task_recovered'] and not r['task_recovered']),delta_roll=r['peak_roll_rad']-b['peak_roll_rad'],delta_tail_path=r['tail_max_path_m']-b['tail_max_path_m'],delta_return=r['logged_return']-b['logged_return'])
 for name,row,loc in [('parent',b,base),('taught',r,run)]:
  p=loc/(row['name']+'.npz')
  with np.load(p) as z:
   ax.plot(*z['actual_xy'].T,label=name);ax.plot(*z['actual_xy'][-1],marker='x');ref=z['reference_xy'];m=z['time']>=9
   pair[name]=dict(tail_heading=float(np.max(abs(z['path_heading_error'][m]))) if m.any() else None,tail_speed=float(np.max(abs(z['actual_forward_speed'][m]-r['speed']))) if m.any() else None,tail_path=row['tail_max_path_m'],peak_roll=row['peak_roll_rad'],sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 ax.plot(*ref.T,'k--',label='original path');ax.set_aspect('equal',adjustable='datalim');ax.set_title(str(key(r)),fontsize=8);ax.legend(fontsize=6);ax.grid(alpha=.2);pairs.append(pair)
out=R/'analysis';out.mkdir(exist_ok=False);fig.savefig(out/'all_cases_xy.png',dpi=95);plt.close(fig)
(out/'summary.json').write_text(json.dumps(dict(summary=summaries,paired=pairs,role='DEV not used in teaching; no masks; single training seed'),indent=2)+'\n')
lines=['# Fixed-Actor low-speed DEV rollout check','','![All paired actual XY trajectories](all_cases_xy.png)','','| Policy | Task /60 | Disturbed /40 | Nominal /20 | Working /60 | Domain |','|---|---:|---:|---:|---:|---:|']
for g in summaries:lines.append(f"| {g['policy']} | {g['task']} | {g['disturbed']} | {g['nominal']} | {g['working']} | {g['domain']} |")
lines += ['',f"Paired task improvements: {sum(r['improved'] for r in pairs)}; regressions: {sum(r['regressed'] for r in pairs)}.",'','Fixed final400 supervised actor, ordinary closed-loop interface. These DEV contexts were not used as teaching samples; they have been used in prior development diagnosis and are not final TEST. No automatic adoption.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summaries));print('gains/losses',sum(r['improved'] for r in pairs),sum(r['regressed'] for r in pairs))

target=lambda r:r['speed']==2.35 and r['alpha'] in (0.,.125) and r['force_N']!=0
print('target8',[(name,sum(r['task_recovered'] for r in group if target(r))) for name,group in [('parent',parents),('taught',rows)]])
