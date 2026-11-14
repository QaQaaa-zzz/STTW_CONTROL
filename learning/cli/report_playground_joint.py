"""Report every frozen finite-motion comparison, including no-plan successes."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--backtracking',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
manifest=json.loads((a.root/'manifest.json').read_text());plans=json.loads((a.root/'plans.json').read_text())
if a.backtracking:plans+=json.loads((a.backtracking/'plans.json').read_text())
rows=[];fig,axes=plt.subplots(6,1,figsize=(15,13),layout='constrained')
colors={'segmented':'#d55e00','joint':'#0072b2'}
if a.backtracking:colors['backtracking']='#009e73'
for plan in plans:
    evaluation=Path(plan['evaluation']);status=json.loads((evaluation/'status.json').read_text())
    ax=axes[plan['layout']]
    for index,case in enumerate(plan['case_dirs']):
        result=next(r for r in status['results'] if r['case_dir']==case)
        row={k:plan[k] for k in ['layout','method','feasible','candidate','route','activation_x_m','shared_planning_ms','clearance_m']}
        row.update(root=result['variant']['root_variant'],goal=result['goal'],termination=result['termination'],duration_s=result['duration_s'],penalized_time_s=result['duration_s'] if result['goal'] else manifest['seconds'],handoff_s=result['handoff_s'],forbidden_contact_steps=result['forbidden_contact_steps'],trajectory=str(evaluation/case/'trajectory.npz'))
        rows.append(row)
        with np.load(row['trajectory']) as data:
            xy=data['qpos'][:,:2];ref=data['reference_xy']
            phase=data['trace'][:,-1];drift=[]
            for value in [0,2]:
                selected=data['qpos'][phase==value,10:12]
                if len(selected):drift.append(float(np.max(np.abs(selected-selected[0]))))
            row['locked_joint_max_drift_rad']=max(drift)
            row['goal_hold_s']=result['goal_hold_s']
            row['min_world_forward_m_s']=float(data['qvel'][:,0].min())
            ax.plot(xy[:,0],xy[:,1],color=colors[plan['method']],ls=['-','--'][index],lw=1.4,label=f"{plan['method']} root{row['root']} {row['termination']}")
            ax.plot(*xy[-1],marker='o' if row['goal'] else 'x',color=colors[plan['method']],ms=5)
            if index==0:ax.plot(ref[:,0],ref[:,1],':',color=colors[plan['method']],alpha=.5,lw=1)
for layout,ax in enumerate(axes):
    spec=manifest['layouts'][layout];x0,x1,y0,y1=spec['rectangle']
    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,color='crimson',alpha=.6))
    ax.axvspan(10.3,10.6,color='gray',alpha=.25)
    for lo,hi in [(4.5,5.5),(13,15)]:ax.axvspan(lo,hi,color='sandybrown',alpha=.12)
    ax.set(xlim=(1.2,18.6),ylim=(-.95,.95),xlabel='World x (m)',ylabel='World y (m)',title=f'Layout {layout}: actual trajectories; dotted = selected reference; RB checkpoint225 / H75')
    ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(loc='upper left',fontsize=7,ncol=4)
fig.savefig(a.output/'paired_xy.png',dpi=160);plt.close(fig)
summary={}
for method in colors:
    selected=[r for r in rows if r['method']==method]
    summary[method]=dict(episodes=len(selected),physical_goals=sum(r['goal'] for r in selected),planned_goals=sum(r['goal'] and r['feasible'] for r in selected),no_plan_goals=sum(r['goal'] and not r['feasible'] for r in selected),mean_penalized_time_s=float(np.mean([r['penalized_time_s'] for r in selected])))
pairs=[]
for baseline in [r for r in rows if r['method']=='segmented']:
    joint=next(r for r in rows if r['method']=='joint' and (r['layout'],r['root'])==(baseline['layout'],baseline['root']))
    pairs.append(dict(layout=baseline['layout'],root=baseline['root'],gained=joint['goal'] and not baseline['goal'],lost=baseline['goal'] and not joint['goal'],time_difference_s=joint['penalized_time_s']-baseline['penalized_time_s']))
summary.update(paired_gains=sum(r['gained'] for r in pairs),paired_losses=sum(r['lost'] for r in pairs),paired_mean_time_difference_s=float(np.mean([r['time_difference_s'] for r in pairs])))
(a.output/'summary.json').write_text(json.dumps(dict(summary=summary,cases=rows,pairs=pairs),indent=2)+'\n')
lines=['# E2有限运动库联合规划 DEV 对照','', '![全部配对实际/参考XY](paired_xy.png)','',f"实际全程成功：分段 {summary['segmented']['physical_goals']}/12，联合 {summary['joint']['physical_goals']}/12；新增 {summary['paired_gains']}，丢失 {summary['paired_losses']}。失败按10秒计，配对平均时间差（联合减分段）{summary['paired_mean_time_difference_s']:.4f}秒。",'', '无可行计划时执行原直线候选，实际成功仍计入主结果。规划可行且成功与物理成功分别统计。','','|布局|根|方法|计划可行|路线|启动x m|实际结果|时长s|交回s|','|---|---|---|---|---|---|---|---|---|']
for r in sorted(rows,key=lambda r:(r['layout'],r['root'],r['method'])):
    lines.append(f"|{r['layout']}|{r['root']}|{r['method']}|{r['feasible']}|{r['route']}|{r['activation_x_m']:.2f}|{r['termination']}|{r['duration_s']:.3f}|{r['handoff_s']}|")
if a.backtracking:
    back=summary['backtracking'];lines+=['',f"加强的回溯分段实际成功{back['physical_goals']}/12，平均惩罚时间{back['mean_penalized_time_s']:.4f}秒。按相同完整可行性筛选，依前段代价顺序回溯启动/上游组合，在首个有可行后续的组合中选最低全程代价；与联合法同36候选。已完成且配置相同的条件复用，新增物理10条。"]
if a.backtracking:
    lines[4]=f"加强基线后：回溯分段{summary['backtracking']['physical_goals']}/12，联合{summary['joint']['physical_goals']}/12；平均完成时间差（联合减回溯）{summary['joint']['mean_penalized_time_s']-summary['backtracking']['mean_penalized_time_s']:.4f}秒。未证明独立成功率收益。原贪心分段3/12的九条失败均保留；三方法36个条件来自34条物理运行，另两条同身份复用。"
lines+=['','两方法同36个实测完整候选、同原粗糙地图/沟两侧2m平地/终点保持/冻结RB225和H75。H0先选直下游的最低代价前缀，再选同启动/上游下的后续；H1联合选择全链。H0无可行后续时保留原始候选，不能代表所有允许回溯的分段算法。','', '候选来自72条TRAIN轨迹，空间支持包络用既有18条DEV校准。六个新增障碍布局×两个复用DEV初态，非独立TEST。最初各向同性0.4565m裕量令两方法全部无计划；该负结果保存于e2_primitive_pilot且未启动物理评测。本轮按固定物理区域/坐标轴最大偏差加0.02m扩张，不是安全概率或连续碰撞证明。','', '静态启动前选计划；不是200ms滚动发布或完整P2。计时只覆盖共享GPU候选评分，不包含数据上传/完整在线循环。全部实际轨迹在真实终止处截止。','',f'原始证据：`{a.root}`。逐例路径和完整数值见summary.json；未重跑物理。']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(summary,indent=2))
