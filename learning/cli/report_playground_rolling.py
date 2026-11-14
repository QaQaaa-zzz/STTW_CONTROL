"""Summarize physical and planner outcomes separately for rolling execution."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
manifest=json.loads((a.root/'manifest.json').read_text());static_plans=json.loads((Path(manifest['source'])/'plans.json').read_text());rows=[];events=[]
fig,axes=plt.subplots(6,1,figsize=(15,13),layout='constrained')
for entry in manifest['entries']:
    layout=entry['layout'];e=Path(entry['evaluation']);status=json.loads((e/'status.json').read_text());ee=json.loads((e/'rb_run3_rolling_events.json').read_text());events.extend(dict(layout=layout,**r) for r in ee);ax=axes[layout]
    static=next(r for r in static_plans if r['layout']==layout and r['method']=='joint');se=Path(static['evaluation']);ss=json.loads((se/'status.json').read_text())
    for lane,result in enumerate(status['results']):
        root=result['variant']['root_variant'];paired=next(r for r in ss['results'] if r['variant']['root_variant']==root)
        row=dict(layout=layout,root=root,goal=result['goal'],planning_failed=result['planning_failed'],termination=result['termination'],duration_s=result['duration_s'],static_duration_s=paired['duration_s'],final_candidate=result['final_candidate'],final_route=result['final_route'],handoff_s=result['handoff_s'],goal_hold_s=result['goal_hold_s'],trajectory=str(e/result['case_dir']/'trajectory.npz'))
        queries=[r for r in ee if r['event']=='query'];row['query_count']=len(queries);row['first_planning_failure_s']=next((r['time_s'] for r in queries if r['planning_failed'][lane]),None)
        row['publications']=[r['time_s'] for r in ee if r['event']=='publication' and r['changed'][lane]];rows.append(row)
        for label,path,color in [('rolling',Path(row['trajectory']),'#0072b2'),('static',se/paired['case_dir']/'trajectory.npz','#d55e00')]:
            with np.load(path) as d:
                xy=d['qpos'][:,:2];ax.plot(xy[:,0],xy[:,1],color=color,ls=['-','--'][lane],lw=1.3,label=f'{label} root{root}');ax.plot(*xy[-1],marker='o',color=color,ms=4)
                if label=='rolling' and lane==0:ref=d['published_reference_xy'];ax.plot(ref[:,0],ref[:,1],':',color=color,alpha=.5)
    world=json.loads(Path(json.loads((e/'resolved_config.json').read_text())['world_map']).read_text());x0,x1,y0,y1=world['terrain']['forbidden_rectangles'][0];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='crimson',alpha=.6));ax.axvspan(10.3,10.6,color='gray',alpha=.25)
    ax.set(xlim=(1.2,18.6),ylim=(-.95,.95),xlabel='World x (m)',ylabel='World y (m)',title=f'Layout{layout}: actual XY, dotted final reference; RB225/H75');ax.set_aspect('equal');ax.legend(loc='upper left',ncol=4,fontsize=7);ax.grid(alpha=.2)
fig.savefig(a.output/'paired_xy.png',dpi=160);plt.close(fig)
queries=[r for r in events if r['event']=='query'];pub=[r for r in events if r['event']=='publication'];timing=lambda key,rs:{k:float(np.percentile([r[key] for r in rs],p)) for k,p in [('median',50),('p95',95),('max',100)]}
receipts=[r for r in pub if any(r['changed'])]
summary=dict(physical_goals=sum(r['goal'] for r in rows),episodes=len(rows),goals_without_planning_failure=sum(r['goal'] and not r['planning_failed'] for r in rows),planning_failures=sum(r['planning_failed'] for r in rows),query_batches=len(queries),publication_attempts=len(pub),changed_publications=len(receipts),all_changed_history_preserved=all(r['state_preserved'] for r in receipts),all_delays_covered=all(r['latency_covered'] for r in pub),query_ms=timing('query_ms',queries),publication_ms=timing('publication_ms',pub),mean_time_difference_s=float(np.mean([r['duration_s']-r['static_duration_s'] for r in rows])),charged_delay_s=sorted({r['charged_delay_s'] for r in pub}))
(a.output/'summary.json').write_text(json.dumps(dict(summary=summary,cases=rows,events=events),indent=2)+'\n')
lines=['# P2滚动有限库执行','', '![全部实际/参考XY，与静态联合配对](paired_xy.png)','',f"物理任务完成{summary['physical_goals']}/{len(rows)}，全程未出现规划覆盖失效且到达{summary['goals_without_planning_failure']}/{len(rows)}；两者分开报告。相对静态同地图/根，平均物理完成时间差{summary['mean_time_difference_s']:.4f}秒。",'',f"每200ms在地面阶段查询，完整RB执行期间保持技能归属。共{len(queries)}次批量查询、{len(receipts)}次改变计划的发布，历史保持均为{summary['all_changed_history_preserved']}。每次计入物理执行的延迟{summary['charged_delay_s']}秒，查询加发布耗时均被覆盖：{summary['all_delays_covered']}。计时含读取状态特征、GPU筛选、同步和发布；固定地图净空及运动库离线预计算，不含新地图构建或连续CEM。",'', '|布局|根|物理结果|规划失效|首次失效s|发布s|时长s|','|---|---|---|---|---|---|---|']
for r in rows:lines.append(f"|{r['layout']}|{r['root']}|{r['termination']}|{r['planning_failed']}|{r['first_planning_failure_s']}|{r['publications']}|{r['duration_s']:.3f}|")
lines+=['', '初始只承诺左上游/直下游到x8.3m的前缀，不冒称完整可行计划。筛选使用实测物理状态和381维控制历史的经验覆盖，状态/路径/激活条件不满足时不发布新候选；失去覆盖仍保留失败标志，实际继续控制不解释为已认证安全回退。已承诺路径前缀不变，原物理、地图、技能、目标及沟两侧2m平地不变。', '', '六张既有DEV图和两个复用DEV根；不称独立TEST、连续规划完备性、任意速度域或全任务完成。固定图中主要发生一次有效改路，后续周期查询多为验证当前候选；不把查询次数当多次不同路线优化收益。', '',f'原始记录：`{a.root}`；完整事件、逐例、时延分位数见summary.json。']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
