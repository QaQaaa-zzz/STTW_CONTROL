"""Pre-switch state selector, leave-one-physical-group-out DEV; no outcome leakage."""
import argparse,json,csv
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;spec=json.loads((r/'protocol.json').read_text());metrics=json.loads((r/'analysis_summary.json').read_text())['rows'];bank=[];audit=[];alphas=spec['target_alphas']
for job in json.loads((r/'jobs.json').read_text()):
 z=np.load(r/job['name']/'trajectory.npz');obs=z['observation'];ticks=z['switch_tick'];m=z['measurement'];xy=z['lower_trace'].reshape(-1,28,11)[:,:,:2]
 for pre in spec['prefix_alphas']:
  lanes=[spec['case_labels'].index(f'{pre:g}_to_{alpha:g}') for alpha in alphas];starts=[ticks[:,l][ticks[:,l]>=0] for l in lanes]
  if any(len(x)==0 for x in starts):audit.append(dict(case=job['name'],prefix=pre,eligible=False,reason='some lanes did not reach switch'));continue
  k=[int(x[0]) for x in starts];i=max(0,min(k)-1);spread=float(np.max(abs(m[:i+1,lanes]-m[:i+1,lanes[0],None])));xyspread=float(np.max(abs(xy[:i+1,lanes]-xy[:i+1,lanes[0],None])));ok=len(set(k))==1 and spread<=.001 and xyspread<=.001
  audit.append(dict(case=job['name'],prefix=pre,eligible=ok,switch_ticks=k,max_measurement_spread=spread,max_xy_spread_m=xyspread))
  if not ok:continue
  l=lanes[0];features=np.r_[obs[i,l,300:305],obs[i,l,337:340],obs[i,l,346:350],obs[i,l,345]]
  outcomes=[next(x for x in metrics if x['case']==job['name'] and x['label']==f'{pre:g}_to_{alpha:g}') for alpha in alphas]
  bank.append(dict(case=job['name'],terrain=job['name'].split('_v')[0],seed=job['physical_seed'],prefix=pre,features=features,outcomes=outcomes))
results=[]
for b in bank:
 train=[t for t in bank if t['seed']!=b['seed'] and t['terrain']==b['terrain']]
 if len(train)<3:continue
 nearest=sorted(train,key=lambda t:float(np.sum((t['features']-b['features'])**2)))[:3]
 for task,limit in spec['tasks'].items():
  def rank(pool):
   scores=[]
   for j,alpha in enumerate(alphas):
    rows=[x['outcomes'][j] for x in pool];success=np.mean([x['tasks'][task] for x in rows]);cost=np.mean([10*x['failed']+x['path_rmse_m']/limit['W_m']+x['speed_rmse_m_s']/limit['speed_rmse_m_s'] for x in rows]);scores.append((-success,cost,alpha,j))
   return min(scores)[3]
  selected=rank(nearest);context=rank(train)
  result=dict(case=b['case'],prefix=b['prefix'],task=task,selected_alpha=alphas[selected],selected_pass=b['outcomes'][selected]['tasks'][task],context_only_alpha=alphas[context],context_only_pass=b['outcomes'][context]['tasks'][task],oracle_pass=any(x['tasks'][task] for x in b['outcomes']),neighbors=[f"{x['case']}/{x['prefix']}" for x in nearest])
  result.update({f'fixed_continuation_{alpha:g}':b['outcomes'][j]['tasks'][task] for j,alpha in enumerate(alphas)});results.append(result)
(r/'state_selection.json').write_text(json.dumps(dict(audit=audit,results=results,role='physical-group held-out DEV; familiar terrain; retrospective paired outcome lookup, not online selector rollout'),indent=2))
lines=['# 状态相关选档：物理组留出DEV','', '[全覆盖实际/参考XY](RESULTS.md)','',f'共{len(audit)}前段组，{len(bank)}组通过共同前段核对；{len(results)}项任务决策。全部回合仍保留于ANALYSIS.md。','', '|任务|决策数|状态近邻|仅地形|固定0|固定.25|固定.5|固定.75|固定1|事后覆盖|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for task in spec['tasks']:
 rows=[x for x in results if x['task']==task];keys=['selected_pass','context_only_pass']+[f'fixed_continuation_{v:g}' for v in alphas]+['oracle_pass'];lines.append('|'+task+'|'+str(len(rows))+'|'+'|'.join(str(sum(x[k] for x in rows)) for k in keys)+'|')
lines+=['','固定档列表示同一前段之后固定使用该档；全程固定档为ANALYSIS.md中的对角线。事后覆盖使用结果，只是上界。选档只读切换前观测；整个待测物理组的所有前段及速度不参与拟合。此处复用已执行配对分支作回顾性评价，尚非新布局在线选档验证。','', '源码：learning/cli/analyze_state_alpha_scan.py；全部组前段偏差、失败/缺失原因、决策与邻居见state_selection.json。']
(r/'state_selection.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
