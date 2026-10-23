"""Native dual-endpoint outcomes and common first-10s diagnostic."""
from pathlib import Path
import json,hashlib
import numpy as np
from sttw_control.recovery_margin_scan import outcome_labels
R=Path(__file__).resolve().parent
records=[];sources={};summary=[]
for endpoint in ('alpha0','alpha1'):
 folder=R/endpoint/'evaluation'; status=json.loads((folder/'status.json').read_text())
 if status['phase']!='completed': raise RuntimeError(endpoint+' not complete')
 d=json.loads((R/endpoint/'declaration.json').read_text());b=d['config']['stage_B_baseline']
 rows=json.loads((folder/'results.json').read_text())['cases'];assert len(rows)==18
 for row in rows:
  path=folder/(row['name']+'.npz');sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  with np.load(path) as f:data={k:f[k] for k in f.files}
  with np.load(folder/(row['route']+'_path.npz')) as f:goal=float(f['goal'])
  native=outcome_labels(data,speed=row['speed'],goal_s=goal,working_roll=b['working_roll_rad'],horizon=20.,hold_s=b['hold_s'])
  assert native['task_recovered']==row['task_recovered']
  mask=data['time']<10.-1e-5
  clipped={k:v[mask] for k,v in data.items()}
  label=outcome_labels(clipped,speed=row['speed'],goal_s=goal,working_roll=b['working_roll_rad'],horizon=10.,hold_s=b['hold_s'])
  records.append(dict(endpoint=endpoint,route=row['route'],speed=row['speed'],alpha=row['alpha'],force_N=row['force_N'],native20=native,clipped10=label))
 for horizon in ['native20','clipped10']:
  rr=[r for r in records if r['endpoint']==endpoint]
  summary.append(dict(endpoint=endpoint,window=horizon,n=len(rr),task=sum(r[horizon]['task_recovered'] for r in rr),domain=sum(r[horizon]['domain_failure'] for r in rr),nominal=sum(r[horizon]['task_recovered'] for r in rr if r['force_N']==0)))
single=R.parent/'preference_empirical_tolerance_v1/evaluation/results.json'
sources[str(single)]=hashlib.sha256(single.read_bytes()).hexdigest()
sr=json.loads(single.read_text())['cases']
# Report matched external requests, not common-onset state or pure architecture effects.
comparison=[]
for a in (0.,1.):
 for policy in sorted(set(r['policy'] for r in sr)):
  rr=[r for r in sr if r['policy']==policy and r['alpha']==a];assert len(rr)==18
  comparison.append(dict(alpha=a,policy=policy,n=18,task=sum(r['task_recovered'] for r in rr)))
result=dict(summary=summary,single10=comparison,cases=records,sources=sources,limitations='Historical dual native20s vs common clipped10s diagnostic; reward, speed bound and architecture differ. Different policies have different own-prefix states. No architectural causal claim.')
(R/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['summary','single10']},indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
single_dir=single.parent
figdir=R/'figures';figdir.mkdir(exist_ok=True)
for a in (0.,1.):
 endpoint=f'alpha{int(a)}';folder=R/endpoint/'evaluation'
 original=json.loads((folder/'results.json').read_text())['cases']
 for route in sorted(set(r['route'] for r in original)):
  force_list=sorted(r['force_N'] for r in original if r['route']==route)
  for window in (10,20):
   fig,axes=plt.subplots(3,3,figsize=(12,12))
   for ax,force in zip(axes.flat,force_list):
    all_xy=[]
    row=next(r for r in original if r['route']==route and r['force_N']==force)
    with np.load(folder/(row['name']+'.npz')) as z:
     mask=z['time']<window-1e-5;xy=z['actual_xy'][mask]
     all_xy.append(xy);line,=ax.plot(xy[:,0],xy[:,1],label=f'dual a{a:g}');ax.plot(*xy[-1],marker='x',color=line.get_color())
    with np.load(folder/(route+'_path.npz')) as z:ax.plot(z['xy'][:,0],z['xy'][:,1],'k--',lw=.7,label='original path')
    if window==10:
     for policy in sorted(set(r['policy'] for r in sr)):
      other=next(r for r in sr if r['alpha']==a and r['route']==route and r['force_N']==force and r['policy']==policy)
      with np.load(single_dir/(other['name']+'.npz')) as z:xy=z['actual_xy'];all_xy.append(xy);line,=ax.plot(xy[:,0],xy[:,1],label=policy);ax.plot(*xy[-1],marker='x',color=line.get_color())
    points=np.concatenate(all_xy);lo=points.min(0)-1.;hi=points.max(0)+1.;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1])
    ax.set_title(f'{force:+g} N, pulse 2.2-2.3 s');ax.set_aspect('equal',adjustable='box');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
   axes.flat[0].legend(fontsize=7)
   fig.suptitle(f'{endpoint} {route}, v=2.35 m/s, {window}s; x = actual trace end\nHistorical dual reward/bounds differ; matched requests, own-policy prefix states')
   fig.tight_layout();fig.savefig(figdir/f'{endpoint}_{route}_{window}s_xy.png',dpi=110);plt.close(fig)
lines=['# 双网络回访：原生结果与共同窗口诊断','','| 端点 | 窗口 | 成功/18 | 域退出 | 无扰成功/2 |','|---|---|---:|---:|---:|']
for s in summary:lines.append(f"| {s['endpoint']} | {s['window']} | {s['task']} | {s['domain']} | {s['nominal']} |")
lines+=['','| 单网络策略 | α | 原10s成功/18 |','|---|---:|---:|']
for s in comparison:lines.append(f"| {s['policy']} | {s['alpha']} | {s['task']} |")
lines+=['','历史双网原生20s结果完整保留，前10s重新按相同任务恢复标签计算；未把后10s成功记为前10s成功。双网历史reward、最高速度3.0m/s与单网V3/3.5m/s不同，单网与双网的起扰前状态也不同。这是实际方案比较，不是纯网络结构因果消融。18例共享少数任务前缀，不能作为18个独立随机初态。','', '数值不表示单网络达到能力上限，也不否定双网络作为参考或潜在教师的价值。是否作为教师需要同一完整受扰状态下的跨策略接续见证，当前面板尚未执行这种接续。','']
for path in sorted(figdir.glob('*.png')):lines.append(f'[{path.stem}](figures/{path.name})')
(R/'RESULTS.md').write_text('\n'.join(lines)+'\n')
