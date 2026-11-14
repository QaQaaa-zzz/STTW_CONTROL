"""Report every static M0/M1/M2 DEV plan and its actual full-chain execution."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
m=json.loads((a.root/'manifest.json').read_text());plans=json.loads((a.root/'plans.json').read_text());rows=[];colors={'M0':'#009e73','M1':'#0072b2','M2':'#d55e00'};figure,axes=plt.subplots(3,2,figsize=(17,11),layout='constrained')
for ax,(map_id,layout) in zip(axes.flat,sorted({(r['map_id'],r['layout']) for r in plans})):
    selected=[r for r in plans if (r['map_id'],r['layout'])==(map_id,layout)]
    for plan in selected:
        source=Path(plan['evaluation']);status=json.loads((source/'status.json').read_text());cfg=json.loads((source/'resolved_config.json').read_text());bank=np.load(cfg['ground_handoff']['reference_path'])
        for case in plan['case_dirs']:
            actual=next(r for r in status['results'] if r['case_dir']==case);root=actual['variant']['root_variant'];style='-' if root==6 else '--'
            with np.load(source/case/'trajectory.npz') as d:xy=d['qpos'][:,:2].copy()
            ax.plot(xy[:,0],xy[:,1],style,color=colors[plan['model']],alpha=.85,label=f"{plan['model']} root{root}");ax.plot(*xy[-1],marker='o' if actual['goal'] else 'x',color=colors[plan['model']]);ref=bank['xy'][actual['route']];use=(ref[:,0]>=xy[0,0])&(ref[:,0]<=xy[-1,0]);ax.plot(ref[use,0],ref[use,1],':',color=colors[plan['model']],alpha=.5)
            rows.append(dict(map_id=map_id,layout=layout,model=plan['model'],root=root,route=plan['route'],speed=plan['speed'],feasible_plan=plan['feasible'],physical_goal=actual['goal'],feasible_plan_goal=plan['feasible'] and actual['goal'],termination=actual['termination'],duration_s=actual['duration_s'],penalized_s=actual['duration_s'] if actual['goal'] else m['seconds'],clearance_m=plan['all_clearance_m'][plan['candidate']],evaluation=str(source),case=case))
    world=cfg['world_map_spec'];ax.axvspan(10.3,10.6,color='gray',alpha=.3)
    for rect in world['terrain']['forbidden_rectangles']:ax.add_patch(plt.Rectangle((rect[0],rect[2]),rect[1]-rect[0],rect[3]-rect[2],color='#cc3311',alpha=.6))
    ax.set(title=f'map{map_id:02d} layout{layout:02d}',xlabel='World x (m)',ylabel='World y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=3)
figure.savefig(a.output/'paired_xy.png',dpi=160);plt.close(figure)
summary=[]
for name in colors:
    sub=[r for r in rows if r['model']==name];summary.append(dict(model=name,conditions=len(sub),physical_goals=sum(r['physical_goal'] for r in sub),feasible_plan_goals=sum(r['feasible_plan_goal'] for r in sub),no_plan=sum(not r['feasible_plan'] for r in sub),predicted_feasible_execution_failures=sum(r['feasible_plan'] and not r['physical_goal'] for r in sub),mean_penalized_s=float(np.mean([r['penalized_s'] for r in sub]))))
for name,data in [('cases',rows),('models',summary)]:
    with (a.output/f'{name}.csv').open('w') as f:writer=csv.DictWriter(f,list(data[0]),lineterminator="\n");writer.writeheader();writer.writerows(data)
lines=['# P3同候选M0/M1/M2实际选路开发比较','','![全部模型、初态、实际和原始参考XY](paired_xy.png)','','两张既有DEV地图×三禁行盒布局×两初态。三个方法共享18条完整路线/速度候选、TRAIN入口历史、成本、底层/RB与物理；分别预测离场并按同一规则校准空间包络。地图/轨迹已用于DEV校准，不是P4独立TEST。静态启动前选路，不是在线神经重规划。','','|模型|条件数|物理目标|空间准入且目标|无计划|失败计10秒的均值s|','|---|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"|{r['model']}|{r['conditions']}|{r['physical_goals']}|{r['feasible_plan_goals']}|{r['no_plan']}|{r['mean_penalized_s']:.5f}|")
unique={(r['evaluation'],r['case']) for r in rows};lines+=['', '空间预测准入但实际失败：'+', '.join(f"{r['model']} {r['predicted_feasible_execution_failures']}/{r['conditions']-r['no_plan']}" for r in summary)+'。分母为实际发布的空间包络准入条件；无计划不归入假可行。'];lines+=['',f'{len(rows)}条方法条件记录对应{len(unique)}条实际轨迹；同布局同配置的重复选择复用同一物理轨迹，不作为独立重复。无计划保留原直线/1.8候选，其物理成功也不消除规划失败。','', '预测初始化来自map00两个TRAIN样本，未使用未来真实DEV交接状态。地形输入使用当前已知图；区域/坐标扩张是既有DEV最大支持误差+2cm，不是概率界或安全保证。模型和校准包络一起构成被比较系统，不能把差异全归因网络结构。预测使用来源观察时域，未实现新到达时间或目标保持预测证明；实际目标仍按原物理门槛判定。','','全部失败如下：']
for r in rows:
    if not r['feasible_plan_goal']:lines.append(f"- map{r['map_id']} layout{r['layout']} {r['model']} root{r['root']}: {r['termination']}, {r['duration_s']:.3f}s, feasible={r['feasible_plan']}.")
lines+=['', 'M1/M2选中同一配置时不能宣称结构模型胜过直接模型；物理目标改善也要区分减少假可行与避免无计划回退。','',f'原始冻结合同/选择/校准/逐步状态：`{a.root}`。所有条件见cases.csv。']
if m.get('predicted_terminal'):
    lines=[line.replace('P3同候选M0/M1/M2实际选路开发比较','P3同候选模型：加入预测动态域和目标保持门槛').replace('原规划只筛空间包络','原空间筛选已扩展').replace('空间准入','预测约束准入').replace('预测使用来源观察时域，未实现新到达时间或目标保持预测证明；实际目标仍按原物理门槛判定。','预测按共同最长来源时域展开，在原目标条件连续保持0.5秒处终止；到该时刻检查预测姿态/跟踪域/前视/前进，且前缀耗时加预测保持时刻不超过10秒。成本为该预测总时长。个别来源真实终点之后的预测属于未观察部分，不能冒充新增精度证据或安全证明；实际目标仍按原物理门槛判定。') for line in lines]
    lines+=['',f"本轮仅新增{m['new_physical_episodes']}个物理回合，其余按相同世界/控制合同/路线/请求/初态复用旧轨迹。该规则是在已有DEV结果之后加入，仍属开发验证。"]
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');(a.output/'summary.json').write_text(json.dumps(dict(models=summary,method_conditions=len(rows),physical_episodes=len(unique),raw_root=str(a.root)),indent=2)+'\n');print(json.dumps(summary,indent=2))
