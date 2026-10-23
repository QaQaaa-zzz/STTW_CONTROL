from pathlib import Path
import json,pickle,math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.tolerance_training import SPEC
root=Path('runs/train_margin_scan_v1/full');assert json.loads((root/'status.json').read_text())['state']=='completed'
manifest=json.loads((root/'manifest.json').read_text());raw=json.loads((root/'results.json').read_text());rows=raw['cases'];features={x['bank_index']:x['features'] for x in manifest['features']}
out=root/'report';out.mkdir(exist_ok=True)
for row in rows:
 x=features[row['bank_index']];lam=x['lambda_u'];K=SPEC['gamma']*(-math.expm1(-lam*.1))/lam
 W=x['b_u']**2*(-math.expm1(-2*lam*min(1.,x['remaining_s'])))/(2*lam);radius=math.sqrt(SPEC['effort_budget']*W)
 risk=abs(x['z_u']+K*row['force_N'])/radius
 row.update(normalized_energy_sqrt=risk,predicted_recoverable=risk<=1.,
  predicted_plus_N=max(0.,(radius-x['z_u'])/K) if abs(x['z_u'])<=radius else 0.,
  predicted_minus_N=max(0.,(radius+x['z_u'])/K) if abs(x['z_u'])<=radius else 0.)
nominal={r['bank_index']:r for r in rows if r['force_N']==0}
def score(rr,label):
 rr=[r for r in rr if r[label] is not None]
 fail=[r['normalized_energy_sqrt'] for r in rr if not r[label]];ok=[r['normalized_energy_sqrt'] for r in rr if r[label]]
 auc=float(np.mean((np.array(fail)[:,None]>np.array(ok)[None,:])+.5*(np.array(fail)[:,None]==np.array(ok)[None,:]))) if fail and ok else None
 return dict(n=len(rr),successes=sum(r[label] for r in rr),predicted_within=sum(r['predicted_recoverable'] for r in rr),false_positive=sum(r['predicted_recoverable'] and not r[label] for r in rr),conservative=sum(not r['predicted_recoverable'] and r[label] for r in rr),risk_rank_auc=auc)
summaries={}
for role in ['calibration','validation']:
 group=[r for r in rows if r['data_role']==role]
 summaries[role]=dict(all_cases={label:score(group,label) for label in ['task_recovered','performance_success','physical_survived_horizon']},
  valid_nominal_nonzero={label:score([r for r in group if r['force_N'] and nominal[r['bank_index']]['task_recovered']],label) for label in ['task_recovered','performance_success','physical_survived_horizon']},
  states=len({r['bank_index'] for r in group}),nominal_task_failures=sum(not r['task_recovered'] for r in group if r['force_N']==0))
with (Path(manifest['declaration']['bank_dir'])/'bank.pkl').open('rb') as f:bank=pickle.load(f)
for idx in features:
 subset=[r for r in rows if r['bank_index']==idx]
 fig,axes=plt.subplots(1,3,figsize=(14,4));ax=axes[0]
 progress=0.
 for row in subset:
  with np.load(root/(row['name']+'.npz')) as d:
   t=d['time'];xy=d['actual_xy'];progress=max(progress,float(d['path_progress'].max()))
   color='black' if not row['force_N'] else plt.cm.coolwarm((row['force_N']+12)/24)
   ax.plot(xy[:,0],xy[:,1],color=color,lw=1.5 if row['force_N']==0 else .7)
   ax.scatter(*xy[-1],c=[color],marker='o' if row['task_recovered'] else 'x',s=14)
   axes[1].plot(t,d['path_cross_track'],color=color,lw=.7)
   axes[2].plot(t,d['phi'],color=color,lw=.7)
 path=bank['routes'].path;mask=np.asarray(path.s[idx])<=progress+1.
 ax.plot(np.asarray(path.xy[idx])[mask,0],np.asarray(path.xy[idx])[mask,1],ls=':',c='gray',lw=1)
 ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='World x (m)',ylabel='World y (m)',title='Actual suffix XY; dotted original path')
 axes[1].set(xlabel='Original task time (s)',ylabel='Path error (m)');axes[2].set(xlabel='Original task time (s)',ylabel='Roll (rad)')
 for a in axes:a.grid(alpha=.2)
 fig.suptitle(f'TRAIN bank {idx}: frozen uniform u27 suffix | black:0N, blue:-12N, red:+12N | x:task failure')
 fig.tight_layout();fig.savefig(out/f'bank{idx:03d}.png',dpi=110);plt.close(fig)
summary=dict(summaries=summaries,rows=rows,boundaries=raw['boundaries'],physical_failures=sum(r['physical_failure'] for r in rows),domain_failures=sum(r['domain_failure'] for r in rows),physical_unknown=sum(r['physical_survived_horizon'] is None for r in rows),scope='Frozen SPEC, no refit. Complete parent TRAIN prefix and fixed uniform suffix; conditional recoverability only. Grouped calibration/validation, descriptive correlated rows; no independent final TEST.')
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# TRAIN同状态方向容扰度扫描','',summary['scope'],'','32状态×13力档；0.1秒脉冲，保留原20秒截止，无训练。每第4个状态预留validation，所有公式参数原样，未拟合。','',f"物理失败{summary['physical_failures']}，路径域失败{summary['domain_failures']}，提前域退出导致的剩余物理存活UNKNOWN={summary['physical_unknown']}。",'', '```json',json.dumps(summaries,indent=2),'```','','## 实际XY与误差','']
lines += [f'- [bank{idx:03d}](bank{idx:03d}.png)' for idx in features]
lines += ['','方向离散边界见summary.json：保留非单调、UNKNOWN、名义失败和右删失。最大通过力档不是连续安全边界；不把父前缀+新后缀宣称为同一Actor全程执行。']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','boundaries']},indent=2))
