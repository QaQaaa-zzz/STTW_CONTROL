"""Evaluate frozen S3 choices without fitting to the new parameter groups."""
import argparse,csv,json,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();r=a.run;out=r/'prediction_analysis';out.mkdir(exist_ok=True)
rows=json.loads((r/'analysis_summary.json').read_text())['rows'];jobs=json.loads((r/'jobs.json').read_text());frozen=json.loads((r/'frozen_predictions.json').read_text());forecasts=frozen['contexts'];lookup={(x['case'],x['label']):x for x in rows};fixed=[0,.25,.5,.75,1];pairs=[];calibration=[];context_rows=[]
for j in jobs:
    f=next(x for x in forecasts if x['family']==j['family'] and x['height_m']==j['height_m'] and x['speed_m_s']==j['speed_m_s'])
    for pr in f['predictions']:
        actual=lookup[j['name'],f"full_{pr['alpha']:g}"]
        calibration.append(dict(case=j['name'],family=j['family'],height_m=j['height_m'],speed_m_s=j['speed_m_s'],group=j['physical_seed'],repeat=j['repeat'],alpha=pr['alpha'],physical_vote=pr['physical_feasibility_vote'],actual_full=not actual['failed'],actual_hold=actual['final_hold'],cause=actual['cause']))
    for task in ['A','B','C']:
        choice=f['selected_alpha_tasks'][task];selected=lookup[j['name'],f"full_{choice['alpha']:g}"]
        for base in fixed:
            baseline=lookup[j['name'],f'full_{base:g}'];pairs.append(dict(case=j['name'],family=j['family'],group=j['physical_seed'],repeat=j['repeat'],task=task,selected_alpha=choice['alpha'],selection_mode=choice['mode'],fixed_alpha=base,selected_full=not selected['failed'],selected_hold=selected['final_hold'],selected_success=selected['tasks'][task],fixed_success=baseline['tasks'][task],gain=int(selected['tasks'][task] and not baseline['tasks'][task]),loss=int(baseline['tasks'][task] and not selected['tasks'][task])))
for f in forecasts:
    groupjobs=[j for j in jobs if (j['family'],j['height_m'],j['speed_m_s'])==(f['family'],f['height_m'],f['speed_m_s'])]
    for pr in f['predictions']:
        actual=[lookup[j['name'],f"full_{pr['alpha']:g}"] for j in groupjobs]
        context_rows.append(dict(family=f['family'],height_m=f['height_m'],speed_m_s=f['speed_m_s'],alpha=pr['alpha'],n=len(actual),full=sum(not x['failed'] for x in actual),hold=sum(x['final_hold'] for x in actual),**{t:sum(x['tasks'][t] for x in actual) for t in ['A','B','C']}))
for name,data in [('paired_choices.csv',pairs),('forecast_reliability.csv',calibration),('context_alpha_counts.csv',context_rows)]:
    with (out/name).open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
contact_checks=[]
for case in sorted({x['case'] for x in rows if x['cause']=='other_unresolved'}):
    z=np.load(r/case/'trajectory.npz');c=json.loads((r/case/'contract.json').read_text());nonwheel=[i for i,name in enumerate(c['bodies']) if name not in ['frontwheel','rearwheel']]
    for item in [x for x in rows if x['case']==case and x['cause']=='other_unresolved']:
        lane=item['lane'];step=int(np.flatnonzero(z['failure'][:,lane])[0]);force=np.linalg.norm(z['contact_force'][step*4:(step+1)*4,lane][:,nonwheel],axis=-1);peak=float(force.max())
        contact_checks.append(dict(case=case,label=item['label'],max_nonwheel_force_N=peak,nonwheel_contact_recorded=peak>0))
(out/'unresolved_contact_check.json').write_text(json.dumps(contact_checks,indent=2))
summary={}
for task in ['A','B','C']:
    counts={};comparisons={}
    for base in fixed:
        pr=[x for x in pairs if x['task']==task and x['fixed_alpha']==base];counts[str(base)]=sum(x['fixed_success'] for x in pr);comparisons[str(base)]={key:sum(x[key] for x in pr) for key in ['gain','loss']}
    selected=[x for x in pairs if x['task']==task and x['fixed_alpha']==0];counts['selected']=sum(x['selected_success'] for x in selected)
    group_values=[]
    for g in sorted({j['physical_seed'] for j in jobs}):
        gp=[x for x in pairs if x['task']==task and x['group']==g];group_values.append(dict(group=g,selected=sum(x['selected_success'] for x in gp if x['fixed_alpha']==0),**{str(b):sum(x['fixed_success'] for x in gp if x['fixed_alpha']==b) for b in fixed}))
    summary[task]=dict(n=len(selected),counts=counts,paired=comparisons,by_physical_group=group_values)
reliability=[]
for vote in [0,.5,1]:
    c=[x for x in calibration if x['physical_vote']==vote];reliability.append(dict(vote=vote,n=len(c),full=sum(x['actual_full'] for x in c),observed_full_rate=float(np.mean([x['actual_full'] for x in c]))))
result=dict(episodes=len(rows),contexts=len(jobs),termination_counts=dict(collections.Counter(x['cause'] for x in rows)),tasks=summary,reliability=reliability,brier=float(np.mean([(x['physical_vote']-x['actual_full'])**2 for x in calibration])),interpretation='Four physical parameter groups; two nested technical repetitions; no test tuning or alpha reselection')
(out/'summary.json').write_text(json.dumps(result,indent=2))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained');ridge=[x for x in context_rows if x['family']=='ridge' and x['height_m']==.1 and x['speed_m_s']==2.6]
axes[0].plot([x['alpha'] for x in ridge],[x['full'] for x in ridge],'o-',label='Full 10 s');axes[0].plot([x['alpha'] for x in ridge],[x['C'] for x in ridge],'s--',label='Task C');axes[0].set(xlabel='Alpha',ylabel='Count / 8',title='10 cm ridge, 2.6 m/s',ylim=(-.3,8.5));axes[0].legend();axes[0].grid(alpha=.2)
x=np.arange(3);labels=['0','.25','.5','.75','1','selected'];colors=['#0072B2','#56B4E9','#009E73','#E69F00','#D55E00','#332288']
for k,(key,label,color) in enumerate(zip([str(b) for b in fixed]+['selected'],labels,colors)):
    axes[1].bar(x+(k-2.5)*.13,[summary[t]['counts'][key] for t in ['A','B','C']],width=.12,color=color,label=label)
axes[1].set(xticks=x,xticklabels=['Task A','Task B','Task C'],ylabel='Task completions / 176',title='Frozen selector versus fixed alpha',ylim=(0,110));axes[1].legend(ncol=3,fontsize=9);axes[1].grid(axis='y',alpha=.2);fig.savefig(out/'main_results.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,9),layout='constrained')
for ax,metric,title in zip(axes,['full','C'],['Complete 10 seconds','Task C']):
    matrix=np.array([[x[metric] for x in context_rows if (x['family'],x['height_m'],x['speed_m_s'])==(f['family'],f['height_m'],f['speed_m_s'])] for f in forecasts])
    im=ax.imshow(matrix,vmin=0,vmax=8,cmap='viridis',aspect='auto');ax.set(xticks=np.arange(13),xticklabels=[f"{pr['alpha']:g}" for pr in forecasts[0]['predictions']],yticks=np.arange(22),yticklabels=[f"{f['family']} {f['height_m']*1000:g}mm / {f['speed_m_s']:g}" for f in forecasts],xlabel='Alpha',title=title)
    ax.tick_params(axis='x',labelrotation=90)
    for i in range(22):
        for j in range(13):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',fontsize=8,color='white' if matrix[i,j]<4 else 'black')
fig.colorbar(im,ax=axes,label='Count / 8',shrink=.6);fig.savefig(out/'all_contexts.png');plt.close(fig)
lines=['# S3：冻结预测的细alpha验证','','[全部176工况实际/原参考XY](../RESULTS.md)','','![主结果](main_results.png)','','**单脊中间档优势在新物理组复现；任务C冻结选档93/176，高于五个固定档的最高77/176。任务A/B增益较小。**','','2464条：22工况×4新物理组×2重复×(13alpha+Zero)。完整10s、最终保持和A/B/C任务分开计数；各组两次重复不当作独立物理组。使用运行前冻结预测，不根据本轮重新选档。','','## 全部工况\n\n![22工况13档](all_contexts.png)\n\n## 任务结果','','|任务|固定0|固定.25|固定.5|固定.75|固定1|冻结选档|','|---|---:|---:|---:|---:|---:|---:|']
for t in ['A','B','C']:
    counts=summary[t]['counts'];lines.append('|'+t+'|'+'|'.join(str(counts[str(b)]) for b in fixed)+'|'+str(counts['selected'])+'|')
lines+=['','每项分母176。任务C对固定0/.25/.5分别均救回23、丢失7，净增16；不把净增说成所有工况改善。','','## 四个物理组分别统计','','|任务|物理组|固定0|固定.25|固定.5|固定.75|固定1|选档|','|---|---|---:|---:|---:|---:|---:|---:|']
for t in ['A','B','C']:
    for g in summary[t]['by_physical_group']:lines.append('|'+t+'|'+str(g['group'])+'|'+'|'.join(str(g[str(b)]) for b in fixed)+'|'+str(g['selected'])+'|')
lines+=['','每组分母44（22工况×2技术重复）。四个参数组跨工况复用，不能把176行当176独立物理参数组。','','## 单脊10cm、2.6m/s','','|alpha|完整/8|任务C/8|','|---|---:|---:|']
for c in ridge:lines.append(f"|{c['alpha']:g}|{c['full']}|{c['C']}|")
lines+=['','0.2–0.6之间各档完整8/8，而两端0和1均0/8；中间可运行区域得到细扫描支持。任务C在.4为8/8、.5为7/8；.4是本次观察到的结果，冻结选择仍为.5，不回写成事前预测。','','## 终止与预测可靠性','',str(result['termination_counts']),'','原汇总other_unresolved共61条，其中22条在终止20ms步内保存了非轮部件接触力；另39条200Hz端点未保存到该接触信号，继续保留未解析。不把这些记录统称侧倾超限，原汇总文件不覆盖。详见unresolved_contact_check.json。','','|预测可行票|条数|实际完整|实际比例|','|---|---:|---:|---:|']
for c in reliability:lines.append(f"|{c['vote']:g}|{c['n']}|{c['full']}|{c['observed_full_rate']:.3f}|")
lines+=['','可行票不等同于已校准任务成功概率；本表直接检验其可靠性。','', '[全部22工况13档数据](context_alpha_counts.csv) · [所有任务配对得失](paired_choices.csv) · [逐条预测与实际](forecast_reliability.csv) · [汇总与组别数据](summary.json)','','下一步：完成连续余量预测复算与未解析终止分类，再用冻结选档在新布局上做S4，与所有五个固定档及已有简单规则配对。当前结果支持推进，不改变网络或重训。']
(out/'README.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
