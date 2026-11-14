"""Full paired speed-domain outcomes, with preserved failure endpoints."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
m=json.loads((a.root/'manifest.json').read_text());rows=[];fig,axes=plt.subplots(9,1,figsize=(15,18),layout='constrained');colors={1.7:'#009e73',1.8:'#0072b2',1.9:'#d55e00'}
switches='phase_pairs' in m
labels={e['speed_m_s']:f"{e['speed_m_s']}→{e.get('continuation_speed_m_s',e['speed_m_s'])}" if switches else str(e['speed_m_s']) for e in m['entries']}
sources=[(e['speed_m_s'],e.get('continuation_speed_m_s',e['speed_m_s']),Path(e['evaluation']),False) for e in m['entries']]
if m.get('baseline_reuse'):sources.append((1.8,1.8,Path(m['baseline_reuse']['evaluation']),True))
for speed,after,e,reused in sources:
    status=json.loads((e/'status.json').read_text())
    for result in status['results']:
        root=result['variant']['root_variant'];path=e/result['case_dir']/'trajectory.npz'
        with np.load(path) as d:
            tr=d['trace'];phase=tr[:,-1];q=d['qpos'];v=d['qvel'];xy=q[:,:2];ground=np.isin(phase,[0,2]);request_error=float(np.max(np.abs(tr[ground,18]-np.where(phase[ground]==0,np.float32(speed),np.float32(after)))))
            w,x,y,z=q[:,3:7].T;body_x=np.stack((1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)),1);forward=(body_x*v[:,:3]).sum(1)
            drift=[]
            for stage in [0,2]:
                joints=q[phase==stage,10:12]
                if len(joints):drift.append(float(np.max(np.abs(joints-joints[0]))))
            row=dict(speed_m_s=speed,continuation_speed_m_s=after,route=result['route'],root=root,role=result['variant']['data_split'],reused=reused,goal=result['goal'],termination=result['termination'],duration_s=result['duration_s'],rb_entry_s=result['rb_entry_s'],handoff_s=result['handoff_s'],goal_hold_s=result['goal_hold_s'],final_x_m=float(xy[-1,0]),final_y_m=float(xy[-1,1]),max_abs_roll_rad=float(np.max(abs(tr[:,4]))),min_body_forward_m_s=float(forward.min()),locked_joint_drift_rad=max(drift),requested_speed_max_error_m_s=request_error,trajectory=str(path));rows.append(row)
            ax=axes[result['route']];ax.plot(xy[:,0],xy[:,1],color=colors[speed],alpha=.65,lw=1,ls='-' if root%2==0 else '--',label=f'{labels[speed]}m/s' if root==0 else None);ax.plot(*xy[-1],marker='o' if result['goal'] else 'x',color=colors[speed],ms=4)
            if speed==1.8 and root==0:ref=d['reference_xy'];ax.plot(ref[:,0],ref[:,1],':',color='black',lw=1,label='reference')
for route,ax in enumerate(axes):
    ax.set(xlim=(1.2,18.7),ylim=(-.95,.95),xlabel='World x (m)',ylabel='World y (m)',title=f'Route{route}: upstream{route//3}, downstream{route%3};8 roots per speed; crosses=true failure endpoints');ax.set_aspect('equal');ax.axvspan(10.3,10.6,color='gray',alpha=.2);ax.grid(alpha=.2);ax.legend(ncol=4,loc='upper left',fontsize=8)
fig.savefig(a.output/'paired_xy.png',dpi=150);plt.close(fig)
summary=[]
for speed in m['speeds_m_s']:
    selected=[r for r in rows if r['speed_m_s']==speed];goals=sum(r['goal'] for r in selected)
    summary.append(dict(speed_m_s=speed,continuation_speed_m_s=selected[0]['continuation_speed_m_s'],goals=goals,episodes=len(selected),validated_discrete=goals==72,mean_penalized_time_s=float(np.mean([r['duration_s'] if r['goal'] else 10 for r in selected])),by_route=[dict(route=i,goals=sum(r['goal'] for r in selected if r['route']==i),episodes=sum(r['route']==i for r in selected)) for i in range(9)],failure_counts={term:sum(r['termination']==term for r in selected) for term in sorted({r['termination'] for r in selected if not r['goal']})}))
(a.output/'summary.json').write_text(json.dumps(dict(summary=summary,cases=rows),indent=2)+'\n')
with (a.output/'episodes.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(sorted(rows,key=lambda r:(r['speed_m_s'],r['route'],r['root'])))
domain=dict(requested_ground_speeds_m_s=[r['speed_m_s'] for r in summary if r['validated_discrete']],alpha=0.,activation_x_m=8.55,scope='Discrete full-chain DEV qualification only: nine fixed curves/eight initial states/original terrain/RB225/H75/10s; no continuous interval or independent TEST',interpolation_validated=False,source=str(a.root),results=summary)
if switches:domain.update(validated_phase_pairs=[[r['speed_m_s'],r['continuation_speed_m_s']] for r in summary if r['validated_discrete']],arbitrary_ground_switches_validated=False)
(a.root/'validated_command_domain.json').write_text(json.dumps(domain,indent=2)+'\n')
lines=['# P3相邻速度请求的完整链验证','', '![全部九路线、三个速度、八根的实际/原参考XY](paired_xy.png)','', '|请求m/s|全程成功|失败|平均惩罚时间s|离散命令域准入|','|---|---|---|---|---|']
for r in summary:lines.append(f"|{labels[r['speed_m_s']]}|{r['goals']}/{r['episodes']}|{r['failure_counts']}|{r['mean_penalized_time_s']:.4f}|{r['validated_discrete']}|")
lines+=['','每个速度54TRAIN来源初态条件＋18复用DEV根条件，不称独立TEST。共216条件，198新增物理回合＋18同身份1.8m/s轨迹复用；初始水平速度幅值仍1.8m/s，按既有初始航向旋转，包含到请求速度的过渡。只改变两个STTW阶段的速度请求；RB原生策略、锁定/释放/回交、原粗糙地形、沟两侧2m平地、激活x8.55及10秒/0.5秒保持标准不变。','','|路线|1.7m/s|1.8m/s|1.9m/s|','|---|---|---|---|']
for route in range(9):lines.append('|'+str(route)+'|'+'|'.join(f"{r['by_route'][route]['goals']}/8" for r in summary)+'|')
lines+=['','## 真实失败条件','','|请求|路线|根|终止|时间s|终点x/y m|','|---|---|---|---|---|---|']
for r in rows:
    if not r['goal']:lines.append(f"|{r['speed_m_s']}|{r['route']}|{r['root']}|{r['termination']}|{r['duration_s']:.3f}|{r['final_x_m']:.3f}/{r['final_y_m']:.3f}|")
lines+=['',f"实际逐步请求误差最大{max(r['requested_speed_max_error_m_s'] for r in rows):.3g}m/s，STTW锁定阶段关节漂移最大{max(r['locked_joint_drift_rad'] for r in rows):.6f}rad；全部逐例见episodes.csv。只有72/72完成的离散值进入本实验限定命令域，不由邻点成功推断区间可用。当前规划器仍固定1.8m/s，尚未接入速度节点优化。",'',f'原始记录：`{a.root}`；命令域：同目录validated_command_domain.json。']
if switches:
    replacements={
        '# P3相邻速度请求的完整链验证':'# P3跳前与回交后不同速度请求验证',
        '全部九路线、三个速度、八根':'全部九路线、两组阶段速度、八根',
        '共216条件，198新增物理回合＋18同身份1.8m/s轨迹复用':'共144新增物理回合，两组阶段请求各72条，没有物理轨迹复用',
        '只改变两个STTW阶段的速度请求':'跳前与落地回交后STTW请求不同，在真实RB回交事件改变',
        '|路线|1.7m/s|1.8m/s|1.9m/s|':'|路线|1.8→1.9m/s|1.9→1.8m/s|',
        '|---|---|---|---|':'|---|---|---|'}
    for old,new in replacements.items():lines=[line.replace(old,new) for line in lines]
    lines+=['','准入只适用于两个指定阶段的有序速度对，不代表STTW阶段任意时刻切换，也不代表新增速度的预测模型已验证。']
(a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))
