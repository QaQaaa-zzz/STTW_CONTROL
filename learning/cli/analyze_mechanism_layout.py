"""Paired frozen S4 selector audit; no refitting or outcome-based reselection."""
import argparse, collections, csv, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;o=r/'selection_analysis';o.mkdir(exist_ok=True)
rows=json.loads((r/'analysis_summary.json').read_text())['rows'];jobs=json.loads((r/'jobs.json').read_text());spec=json.loads((r/'protocol.json').read_text());decisions=json.loads((r/'frozen_decisions.json').read_text());idx={(x['case'],x['label']):x for x in rows}
assert len(rows)==2112 and len(idx)==2112 and len(jobs)==176
model=r.parent/'S2/mechanism_forecast/mechanism_forecast.json'
assert hashlib.sha256(model.read_bytes()).hexdigest()==spec['prediction_sha256']==decisions['model_sha256']
fixed=['fixed0','fixed0.25','fixed0.5','fixed0.75','fixed1'];tasks={};pairs=[];cells=[];same_alpha=[];identities=set()
for j in jobs:
 c=json.loads((r/j['name']/'contract.json').read_text());cfg=json.loads(Path(j['config']).read_text());identities.add(c['identity']['source_actor']['source_sha256']);assert c['settings']['alphas']==cfg['alphas'];assert (c['physics_dt'],c['lower_dt'],c['policy_dt'])==(.0002,.005,.02)
 d=next(x for x in decisions['decisions'] if (x['family'],x['height_m'],x['speed_m_s'])==(j['family'],j['height_m'],j['speed_m_s']))
 for task in 'ABC':
  for method in ['mechanism','simple']:
   label=method+'_'+task;assert cfg['alphas'][cfg['case_labels'].index(label)]==d[method][task]
  s=idx[j['name'],'mechanism_'+task]
  for base in fixed+['Zero','simple_'+task]:
   b=idx[j['name'],base];sa=d['mechanism'][task];ba=cfg['alphas'][cfg['case_labels'].index(base)]
   rec=dict(case=j['name'],family=j['family'],height_m=j['height_m'],speed_m_s=j['speed_m_s'],layout=j['layout'],group=j['physical_seed'],task=task,baseline=base,selected_alpha=sa,baseline_alpha=ba,selected_success=int(s['tasks'][task]),baseline_success=int(b['tasks'][task]),gain=int(s['tasks'][task] and not b['tasks'][task]),loss=int(b['tasks'][task] and not s['tasks'][task]))
   pairs.append(rec)
   if base!='Zero' and sa==ba:same_alpha.append(rec)
assert len(identities)==1
rng=np.random.default_rng(401404);gs=rng.integers(0,4,(10000,4));ls=rng.integers(0,2,(10000,2))
for task in 'ABC':
 methods=fixed+['Zero','mechanism_'+task,'simple_'+task];counts={m:sum(idx[j['name'],m]['tasks'][task] for j in jobs) for m in methods};comparisons={}
 for base in fixed+['Zero','simple_'+task]:
  pp=[x for x in pairs if x['task']==task and x['baseline']==base];mat=np.zeros((2,4))
  for layout in [0,1]:
   for g in range(401,405):
    block=[x for x in pp if x['layout']==layout and x['group']==g];assert len(block)==22;mat[layout,g-401]=sum(x['gain']-x['loss'] for x in block)/22
  boot=mat[ls[:,:,None],gs[:,None,:]].mean(axis=(1,2))*100
  same=[x for x in pp if base!='Zero' and x['selected_alpha']==x['baseline_alpha']]
  comparisons[base]=dict(gain=sum(x['gain'] for x in pp),loss=sum(x['loss'] for x in pp),delta_percentage_points=float(mat.mean()*100),crossed_bootstrap95_pp=np.quantile(boot,[.025,.975]).tolist(),same_alpha_pairs=len(same),same_alpha_disagreements=sum(x['gain']+x['loss'] for x in same),different_alpha_net=sum(x['gain']-x['loss'] for x in pp if base=='Zero' or x['selected_alpha']!=x['baseline_alpha']))
 for layout in [0,1]:
  for g in range(401,405):
   jj=[j for j in jobs if j['layout']==layout and j['physical_seed']==g]
   cells.append(dict(task=task,layout=layout,group=g,n=len(jj),**{m:sum(idx[j['name'],m]['tasks'][task] for j in jj) for m in methods}))
 tasks[task]=dict(n=176,counts=counts,paired=comparisons)
result=dict(episodes=len(rows),cases=len(jobs),identity_source_sha256=next(iter(identities)),prediction_sha256=spec['prediction_sha256'],termination_counts=dict(collections.Counter(x['cause'] for x in rows)),tasks=tasks,uncertainty='Descriptive 95% crossed percentile bootstrap; resample 2 layouts and 4 physical groups, retain all 22 contexts within a cell; 10000 draws seed401404. Few layout/group levels; not 176 independent draws.')
result['by_family']={family:{t:{m:sum(x['tasks'][t] for x in rows if x['case'].startswith(family) and x['label']==m) for m in fixed+['Zero','mechanism_'+t,'simple_'+t]} for t in 'ABC'} for family in ['ridge','gravel']}
result['physical_outcomes']={m:dict(n=sum(x['label']==m for x in rows),full=sum(not x['failed'] for x in rows if x['label']==m),hold=sum(x['final_hold'] for x in rows if x['label']==m)) for m in spec['case_labels']}
flat=[{**{k:v for k,v in x.items() if k!='tasks'},**{'task_'+t:x['tasks'][t] for t in 'ABC'}} for x in rows]
with (o/'episode_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(flat[0]),lineterminator="\n");w.writeheader();w.writerows(flat)
(o/'summary.json').write_text(json.dumps(result,indent=2))
for name,data in [('paired_choices.csv',pairs),('group_layout_counts.csv',cells),('same_alpha_pairs.csv',same_alpha)]:
 fields=list(dict.fromkeys(k for x in data for k in x))
 with (o/name).open('w') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(data)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
colors=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#777777','#332288','#CC79A7'];fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
for ax,t in zip(axes,'ABC'):
 vals=list(tasks[t]['counts'].values());x=np.arange(8);ax.bar(x,vals,color=colors);ax.set(xticks=x,xticklabels=['0','.25','.5','.75','1','Zero','Model','Rule'],ylim=(0,100),title='Task '+t,ylabel='Completed / 176');ax.tick_params(axis='x',labelrotation=45);ax.grid(axis='y',alpha=.2)
 for i,v in enumerate(vals):ax.text(i,v+1,str(v),ha='center',fontsize=10)
fig.savefig(o/'task_comparison.png');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,6),layout='constrained')
for ax,t in zip(axes,'ABC'):
 data=[x for x in cells if x['task']==t];keys=fixed+['mechanism_'+t,'simple_'+t];mat=np.array([[x[k] for k in keys] for x in data]);im=ax.imshow(mat,cmap='viridis',vmin=0,vmax=22,aspect='auto');ax.set(xticks=range(7),xticklabels=['0','.25','.5','.75','1','Model','Rule'],yticks=range(8),yticklabels=[f"L{x['layout']} / G{x['group']}" for x in data],title='Task '+t);ax.tick_params(axis='x',labelrotation=45)
 for i in range(8):
  for k in range(7):ax.text(k,i,str(mat[i,k]),ha='center',va='center',color='white' if mat[i,k]<11 else 'black')
fig.colorbar(im,ax=axes,label='Completed / 22',shrink=.65);fig.savefig(o/'layout_groups.png');plt.close(fig)
lines=['# S4：新布局上的冻结选档验证','','[全部176工况实际/原参考XY](../RESULTS.md)','','![任务完成数](task_comparison.png)','','**机制选档在任务C为75/176，优于最佳固定档70/176及简单规则66/176；任务A/B未超过最佳固定档。选档收益依赖任务，当前规则还不是全面优胜者。**','','22工况×2新布局×4新物理组×12车道=2112条；原H75/Local300、共同起点、10s，0训练。每个任务分别比较对应选择器，五固定档和Zero均用同任务重评分。运行前冻结决策，未按结果调档。','','|任务|固定0|固定.25|固定.5|固定.75|固定1|Zero|机制选档|简单规则|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for t in 'ABC':lines.append('|'+t+'|'+'|'.join(str(v) for v in tasks[t]['counts'].values())+'|')
lines+=['','任务通过要求：完整运行、原目标最终保持，且全过程原路径峰值和速度RMSE满足A(.3m,.2m/s)、B(.6m,.1m/s)、C(.4m,.15m/s)。','','## 配对得失与组别变化','','|任务|比较基线|救回|丢失|净增百分点|描述性95%区间|同alpha不一致/同alpha配对|','|---|---|---:|---:|---:|---|---|']
for t in 'ABC':
 for b,v in tasks[t]['paired'].items():
  lo,hi=v['crossed_bootstrap95_pp'];lines.append(f"|{t}|{b}|{v['gain']}|{v['loss']}|{v['delta_percentage_points']:.2f}|[{lo:.2f}, {hi:.2f}]|{v['same_alpha_disagreements']}/{v['same_alpha_pairs']}|")
lines+=['','区间按布局和物理组交叉重采样，保留组内22工况。只有2布局、4物理组，区间用于描述该测试范围的变化。相同alpha的重复车道也有结果差异，单次配对得失不全部归因于选档动作。','','![分布局物理组](layout_groups.png)','','## 终止原因','','`'+json.dumps(result['termination_counts'],ensure_ascii=False)+'`','','共1620条完整、492条提前终止；225条包含侧倾超限，57条包含非轮接触，213条包含运行域退出。组合原因重复计入对应分项；不能将492条统称摔倒。直接终止标志已记录，无未解析分类。','','## 当前结论','','- S3的中间alpha可运行区得到细扫描支持，S4检验了冻结选择在新布局的实际任务效果。','- 任务C有正向增益，A/B仍由固定档或简单规则占优；后续论文应讲地形与任务下的协调及条件收益。','- 不按本轮结果修改预测或扩展训练。先把S0–S4合并为完整证据，再进入写作。','','[逐条配对](paired_choices.csv) · [布局与物理组](group_layout_counts.csv) · [同alpha重复车道](same_alpha_pairs.csv) · [机器汇总](summary.json)','','复算：`python learning/cli/analyze_mechanism_layout.py --run '+str(r)+'`。源analysis_summary.json、jobs.json、各contract及冻结决策；无平滑、无结果剔除。']
lines+=['','## 地形拆分与完整运行','','单脊128工况，碎石48工况；下面表格按相应分母读取。','','|地形|任务|固定0|固定.25|固定.5|固定.75|固定1|Zero|机制|简单规则|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for family,tt in result['by_family'].items():
 for t,counts in tt.items():lines.append('|'+family+'|'+t+'|'+'|'.join(str(v) for v in counts.values())+'|')
lines+=['','任务C单脊机制66/128，最佳固定63/128、简单规则53/128；碎石机制9/48，固定.5为17/48、简单规则13/48。总体收益不等于每类地形均有收益。C相对最佳固定档的描述性区间跨零，当前是小幅观察收益，尚未建立稳定的总体优势。','','|车道|完整10s|最终保持|分母|','|---|---:|---:|---:|']
for m,v in result['physical_outcomes'].items():lines.append(f"|{m}|{v['full']}|{v['hold']}|{v['n']}|")
lines+=['','[全部2112条指标和终止原因](episode_metrics.csv)']
(o/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
