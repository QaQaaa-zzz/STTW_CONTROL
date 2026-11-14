"""Report every frozen terrain-map collection condition, including true failed endpoints."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sttw_control.playground_arx import Reference,states
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
m=json.loads((a.root/'manifest.json').read_text());rows=[];summary=[];diagnostics=[];colors={1.8:'#0072b2',1.9:'#d55e00'}
for entry in m['entries']:
    e=Path(entry['evaluation']);cfg=json.loads((e/'resolved_config.json').read_text());goal=cfg['ground_handoff']['goal'];bank=np.load(cfg['ground_handoff']['reference_path']);follower=json.loads((Path(cfg['ground_handoff']['bundle'])/'config.json').read_text())['follower'];status=json.loads((e/'status.json').read_text());schema=json.loads((e/'model_state_schema.json').read_text())
    joints=[schema['qpos_names'].index(name) for name in ['hip_joint','knee_joint']]
    roots=sorted({r['variant']['root_variant'] for r in status['results']});fig,axes=plt.subplots(9,1,figsize=(15,18),layout='constrained');map_rows=[];extent=[1.2,18.7,-1.,1.]
    for result in status['results']:
        v=result['variant'];root=v['root_variant'];speed=v['requested_speed_m_s'];route=result['route'];path=e/result['case_dir']/'trajectory.npz'
        with np.load(path) as d:
            q=d['qpos'];extent=[min(extent[0],float(q[:,0].min())-.1),max(extent[1],float(q[:,0].max())+.1),min(extent[2],float(q[:,1].min())-.1),max(extent[3],float(q[:,1].max())+.1)];tr=d['trace'];phase=tr[:,-1];xy=q[:,:2];ground=np.isin(phase,[0,2]);drift=[]
            for stage in [0,2]:
                angles=q[phase==stage][:,joints]
                if len(angles):drift.append(float(np.abs(angles-angles[0]).max()))
            row=dict(map=entry['layout'],map_role=entry['map_role'],route=route,root=root,speed_m_s=speed,goal=result['goal'],termination=result['termination'],duration_s=result['duration_s'],rb_entry_s=result['rb_entry_s'],handoff_s=result['handoff_s'],goal_hold_s=result['goal_hold_s'],final_x_m=float(xy[-1,0]),final_y_m=float(xy[-1,1]),max_roll_rad=float(abs(tr[:,4]).max()),locked_joint_drift_rad=max(drift),requested_speed_max_error_m_s=float(abs(tr[ground,18]-np.float32(speed)).max()),trajectory=str(path));map_rows.append(row)
            if not row['goal']:
                pose,xi=states(q,d['qvel'],schema);eta=Reference(bank,route,follower).features(pose,xi)
                inside=(phase==2)&(q[:,0]>=goal['x_min_m'])&(q[:,0]<=goal['x_max_m'])&(abs(q[:,1])<=goal['abs_y_max_m'])&(d['qvel'][:,0]>=goal['min_vx_m_s'])
                heading_ok=abs(eta[:,10])<=goal['heading_tolerance_rad'];path_ok=tr[:,11]<=goal['path_tolerance_m'];speed_ok=abs(xi[:,0]-tr[:,18])<=goal['speed_tolerance_m_s'];streak=longest=0
                for ok in inside&heading_ok&path_ok&speed_ok:streak=streak+1 if ok else 0;longest=max(longest,streak)
                diagnostics.append(dict(map=entry['layout'],case=result['case_dir'],inside_goal_s=float(inside.sum()*.005),max_reconstructed_hold_s=longest*.005,heading_rejected_s=float((inside&~heading_ok).sum()*.005),path_rejected_s=float((inside&~path_ok).sum()*.005),speed_rejected_s=float((inside&~speed_ok).sum()*.005)))
            ax=axes[route];ax.plot(xy[:,0],xy[:,1],color=colors[speed],ls='-' if root==roots[0] else '--',lw=1.2,label=f'{speed}m/s root{root}');ax.plot(*xy[-1],marker='o' if row['goal'] else 'x',color=colors[speed] if row['goal'] else 'red')
            if speed==1.8 and root==roots[0]:ref=d['reference_xy'];ax.plot(ref[:,0],ref[:,1],':',color='black',lw=.8,label='original reference')
    for route,ax in enumerate(axes):
        selected=[r for r in map_rows if r['route']==route];ax.set(title=f"{entry['layout']} {entry['map_role']}: route{route}; goals {sum(r['goal'] for r in selected)}/{len(selected)}",xlabel='World x (m)',ylabel='World y (m)',xlim=extent[:2],ylim=extent[2:]);ax.set_aspect('equal');ax.add_patch(Rectangle((goal['x_min_m'],-goal['abs_y_max_m']),goal['x_max_m']-goal['x_min_m'],2*goal['abs_y_max_m'],facecolor='green',edgecolor='green',alpha=.08,label='goal XY region'));ax.axvspan(10.3,10.6,color='gray',alpha=.2);ax.grid(alpha=.2);ax.legend(ncol=5,fontsize=7,loc='upper left')
    fig.savefig(a.output/(entry['layout']+'_xy.png'),dpi=150);plt.close(fig);rows.extend(map_rows)
    for speed in [1.8,1.9]:
        selected=[r for r in map_rows if r['speed_m_s']==speed]
        summary.append(dict(map=entry['layout'],map_role=entry['map_role'],speed_m_s=speed,goals=sum(r['goal'] for r in selected),episodes=len(selected),termination_counts={term:sum(r['termination']==term for r in selected) for term in sorted({r['termination'] for r in selected})}))
for name,data in [('episodes',rows),('summary',summary)]:
    with (a.output/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
(a.output/'summary.json').write_text(json.dumps(dict(summary=summary,cases=rows,source=str(a.root)),indent=2)+'\n')
lines=['# P3多地图双速度数据采集','', '所有地图先冻结分组：六TRAIN、两DEV_MAP，初态根已有历史使用，非独立P4 TEST。每图九曲线×两初态×两速度。地形仅改变声明碎石种子和2/4cm脊坡x位置，保留原沟、平地缓冲、技能、物理与目标。没有用失败结果替换地图。','']
for entry in m['entries']:lines+=['',f"![{entry['layout']}全部实际/原始参考XY]({entry['layout']}_xy.png)"]
lines+=['','|地图/角色|请求m/s|完整任务|终止分类|','|---|---|---|---|']
for r in summary:lines.append(f"|{r['map']} {r['map_role']}|{r['speed_m_s']}|{r['goals']}/{r['episodes']}|{r['termination_counts']}|")
lines+=['','|失败地图|路线|根|请求|终止|时间s|终点XY m|','|---|---|---|---|---|---|---|']
for r in rows:
    if not r['goal']:lines.append(f"|{r['map']}|{r['route']}|{r['root']}|{r['speed_m_s']}|{r['termination']}|{r['duration_s']:.3f}|{r['final_x_m']:.3f}/{r['final_y_m']:.3f}|")
lines+=['',f"实际请求最大误差{max(r['requested_speed_max_error_m_s'] for r in rows):.3g}m/s；锁定阶段髋膝最大漂移{max(r['locked_joint_drift_rad'] for r in rows):.6f}rad。所有条件见episodes.csv，失败轨迹截止真实终点。此处是固定控制器的数据采集，不是新预测器/规划器的收益。",'',f'原始配置/物理轨迹/阶段状态：`{a.root}`。']
(a.output/'timeout_diagnosis.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
lines+=['','失败目标条件独立重算见timeout_diagnosis.json：航向/路径/速度拒绝时段可重叠，不相加；连续保持为几何重算诊断，不覆盖原在线终止。']
for d in diagnostics:lines.append(f"{d['map']}/{d['case']}：目标XY区内{d['inside_goal_s']:.3f}s，最大连续合格约{d['max_reconstructed_hold_s']:.3f}s（要求0.5s）；航向/路径/速度分别拒绝{d['heading_rejected_s']:.3f}/{d['path_rejected_s']:.3f}/{d['speed_rejected_s']:.3f}s。")
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
