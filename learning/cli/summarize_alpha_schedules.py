"""Summarize all saved natural alpha schedules, retaining termination causes."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);a=p.parse_args();r=a.run
spec=json.loads((r/'protocol.json').read_text());jobs=json.loads((r/'jobs.json').read_text());rows=[]
for job in jobs:
 d=r/job['name'];metrics=json.loads((d/'condition_metrics.json').read_text());z=np.load(d/'trajectory.npz')
 pulse=np.load(d/'applied_pulse.npz') if (d/'applied_pulse.npz').exists() else None
 for lane,m in enumerate(metrics):
  trace=z['lower_trace'][:,:,lane].reshape(-1,11);valid=trace[:,6].astype(bool);cause='full_window';ending=None
  if m['failed']:
   step=int(np.flatnonzero(z['failure'][:,lane])[0]);block=slice(step*4,(step+1)*4);ending=step*.02
   flags={'domain_exit':bool(z['domain_exit'][block,lane].any()),'roll_limit':bool((trace[block,4]>.7).any()),'policy_fault':bool(z['policy_fault'][block,lane].any())}
   for key in ['contact_failure','numerical_failure','roll_limit_failure']:
    if key in z:
     flags['roll_limit' if key=='roll_limit_failure' else key]=bool(z[key][block,lane].any())
   cause='+'.join(k for k,v in flags.items() if v) or 'other_unresolved'
  switches=z['switch_tick'][:,lane] if 'switch_tick' in z else np.array([]);ticks=switches[switches>=0]
  rows.append(dict(case=job['name'],terrain=job['name'].split('_v')[0],label=spec['case_labels'][lane],cause=cause,termination_policy_step_start_s=ending,switch_time_s=float(ticks[0]*.005) if len(ticks) else None,path_peak_m=float(np.abs(trace[valid,3]).max()),**m))
  if pulse is not None:rows[-1]['applied_pulse_impulse_Ns']=pulse['impulse_Ns'][lane].tolist()
for row in rows:
 row['tasks']={name:bool(not row['failed'] and row['final_hold'] and row['path_peak_m']<=task['W_m'] and row['speed_rmse_m_s']<=task['speed_rmse_m_s']) for name,task in spec.get('tasks',{}).items()}
lines=['# '+spec.get('title','无快照恢复分段alpha结果'),'','[全部实际/参考XY与奖励](RESULTS.md)','',f'全部{len(jobs)}上下文、每组{len(spec["case_labels"])}车道；各车道自然运行，未进行中途状态恢复。以下完整/到达/保持分别统计；误差按各自实际有效窗口计算，提前终止不可凭短窗口误差排名。','','|地形|计划|完整10s|到达|末段保持|','|---|---|---:|---:|---:|']
for terrain in spec['terrains']:
 for label in spec['case_labels']:
  ms=[x for x in rows if x['terrain']==terrain and x['label']==label];n=len(ms)
  lines.append(f"|{terrain}|{label}|{sum(not x['failed'] for x in ms)}/{n}|{sum(x['goal_crossed'] for x in ms)}/{n}|{sum(x['final_hold'] for x in ms)}/{n}|")
if spec.get('tasks'):
 lines+=['','全程任务：满窗且末段保持、路径峰值与速度RMSE分别满足protocol中A/B/C阈值。','','|地形|计划|'+'|'.join(spec['tasks'])+'|','|---|---|'+'|'.join(['---:']*len(spec['tasks']))+'|']
 for terrain in spec['terrains']:
  for label in spec['case_labels']:
   ms=[x for x in rows if x['terrain']==terrain and x['label']==label]
   lines.append('|'+terrain+'|'+label+'|'+'|'.join(f"{sum(x['tasks'][name] for x in ms)}/{len(ms)}" for name in spec['tasks'])+'|')
causes={c:sum(x['cause']==c for x in rows) for c in sorted({x['cause'] for x in rows})}
lines+=['','首终止步可见信号：`'+json.dumps(causes,ensure_ascii=False)+'`。侧倾阈值不等于确认非轮触地。','','|工况|计划|完整|保持|路径RMSE m|速度RMSE m/s|终止原因|','|---|---|---|---|---:|---:|---|']
for x in rows:lines.append(f"|{x['case']}|{x['label']}|{not x['failed']}|{x['final_hold']}|{x['path_rmse_m']:.4f}|{x['speed_rmse_m_s']:.4f}|{x['cause']}|")
pairs=[]
for original,duplicate in spec.get('duplicate_pairs',[['0.5_to_0.5','fixed05_duplicate']]):
 group=[]
 for j in jobs:
  a0=next(x for x in rows if x['case']==j['name'] and x['label']==original);a1=next(x for x in rows if x['case']==j['name'] and x['label']==duplicate)
  group.append(dict(case=j['name'],original=original,duplicate=duplicate,same_full=a0['failed']==a1['failed'],same_hold=a0['final_hold']==a1['final_hold'],path_rmse_difference_m=a1['path_rmse_m']-a0['path_rmse_m']))
 pairs.extend(group)
 lines+=['',f"{original}重复对照：完整标签相同 {sum(x['same_full'] for x in group)}/{len(jobs)}；保持标签相同 {sum(x['same_hold'] for x in group)}/{len(jobs)}。重复车道不增加独立样本分母。"]
lines+=['','[逐条数值、切换时刻和重复差异](analysis_summary.json)']
(r/'ANALYSIS.md').write_text('\n'.join(lines)+'\n');(r/'analysis_summary.json').write_text(json.dumps(dict(rows=rows,termination_counts=causes,duplicate_comparison=pairs),indent=2))
