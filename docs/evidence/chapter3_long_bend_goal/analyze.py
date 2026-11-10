"""Evaluate the predeclared common goal and corridor, retaining post-goal outcomes."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
b=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_long_bend_goal');o=Path(__file__).parent
protocol=json.loads((b/'protocol.json').read_text());goal=protocol['goal'];rows=[];arrays={}
fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for row,name in enumerate(protocol['cases']):
 p=b/name;tr=np.load(p/'heading125/trajectory.npz');phys=np.load(p/'heading125/physics_contact.npz');r=phys['root_state_wxyz'][:,0];t=phys['time'];reward=tr['reward'][:,0];ref=np.load(b/(name+'.npz'))['xy'];s=json.loads((p/'summary.json').read_text())['cases'][0]
 h=json.loads((p/'handoff.json').read_text()) if (p/'handoff.json').exists() else None
 v=np.diff(ref,axis=0);d=r[:,None,:2]-ref[None,:-1];u=np.clip((d*v).sum(-1)/(v*v).sum(-1),0,1);distance=np.linalg.norm(d-u[:,:,None]*v,axis=-1).min(1)
 inside=(r[:,0]>=goal['x_min_m'])&(r[:,0]<=goal['x_max_m'])&(abs(r[:,1])<=goal['abs_y_max_m'])&(r[:,7]>=goal['min_vx_m_s'])
 if h:inside&=t>=h['time_s']
 else:inside[:]=False
 ii=np.flatnonzero(inside);arrival=int(ii[0]) if len(ii) else None
 diag=np.load(p/'heading125/verified_diagnostics.npz');before=diag['time']<=(t[arrival] if arrival is not None else t[-1]);physical_failed=bool(diag['failed'][before].any())
 corridor_ok=bool((distance[:arrival+1] <= protocol['road_half_width_m']).all()) if arrival is not None else None
 valid=arrival is not None and corridor_ok and not physical_failed
 segment=(r[:,0]>=1.5)&(r[:,0]<=7.5)
 entry=np.load(p/'approach_entry.npz') if (p/'approach_entry.npz').exists() else None
 entry_features=entry['observation'][0,-4:] if entry is not None else None
 result=dict(case=name,entry_yaw_deg=float(np.degrees(np.arctan2(entry_features[1],1+entry_features[2]))) if entry_features is not None else None,entry_world_vy_m_s=float(entry_features[3]) if entry_features is not None else None,handoff_time_s=h['time_s'] if h else None,first_goal_entry_s=float(t[arrival]) if arrival is not None else None,goal_completed=valid,corridor_until_arrival=corridor_ok,physical_failure_before_end_or_arrival=physical_failed,first_corridor_exit_s=float(t[np.flatnonzero(distance>protocol['road_half_width_m'])[0]]) if (distance>protocol['road_half_width_m']).any() else None,max_centerline_distance_m=float(distance.max()),actual_bend_max_abs_y_m=float(abs(r[segment,1]).max()),end_code=s['terminal_code'],duration_s=s['duration_s'],final_xyz=s['final_xyz']);rows.append(result)
 ax=axes[row,0];ax.plot(*ref.T,'k:',label='Fixed reference');ax.plot(r[:,0],r[:,1],label='Actual');ax.plot(*r[-1,:2],'x');ax.axvspan(10.3,10.6,color='0.7');ax.add_patch(plt.Rectangle((16.5,-.5),2,1,fill=False,edgecolor='green',label='Common goal'));ax.set(xlim=(1,20),ylim=(-2,2),xlabel='World x (m)',ylabel='World y (m)',title=name+' actual/reference');ax.set_aspect('equal',adjustable='box')
 for ax,y,title in [(axes[row,1],reward,'Reward / step (diagnostic)'),(axes[row,2],np.cumsum(reward),'Cumulative reward (diagnostic)')]:ax.plot(tr['time'],y);ax.set(xlabel='Time (s)',ylabel='Reward',title=title)
 for ax in axes[row]:ax.grid(alpha=.2)
 axes[row,0].legend(fontsize=8)
 arrays.update({name+'_time':t,name+'_root':r,name+'_reward':reward,name+'_reward_time':tr['time'],name+'_reference':ref,name+'_corridor_distance':distance})
fig.suptitle('Isaac heading125/H75 | left/right 0.8m offset-return over6m | nominal DEV,0training\nFirst-far-contact handoff; microstones before/after flat30cm jump; fixed common goal')
fig.savefig(o/'xy_and_rewards.png',dpi=150);np.savez_compressed(o/'plot_data.npz',**arrays);(o/'summary.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
report='# 长弯道接近与共同目标区\n\n![全部XY与奖励](xy_and_rewards.png)\n\n运行前合同固定左右参考峰值±0.8m、偏移返回段x1.5～7.5m长6m；触发线x9，平面开口沟10.3～10.6m；0.5～1mm微凸面位于x2～7和12～22。两条共同起点，heading125/H75，接近2m/s α.5，首次远岸触地交接，续行1.5m/s α.5，名义DEV、0训练。\n\n共同目标：x∈[16.5,18.5]m、|y|≤0.5m、vx≥0.5m/s且已交回STTW。到达前不得物理失败，车体中心距原参考折线≤0.6m；以2.5ms物理记录检查。此为中心线道路合同，不冒称完整车体扫掠碰撞保证。无需支撑保持，继续到10秒仅作后续诊断，不用后续表现重写到达时判定。\n\n|候选|实际接近最大|y|(m)|回交时间(s)|首次进目标(s)|满足完整目标合同|终止码/时间|\n|---|---:|---:|---:|---|---|\n'
# Escape vertical bar in the prose header, keeping Markdown columns unambiguous.
report=report.replace('实际接近最大|y|(m)','实际接近最大绝对y(m)')
for r in rows:report+=f"|{r['case']}|{r['actual_bend_max_abs_y_m']:.3f}|{r['handoff_time_s']}|{r['first_goal_entry_s']}|{r['goal_completed']}|{r['end_code']} / {r['duration_s']:.2f}s|\n"
report+='\n全部实际轨迹保留至真实终点。进入目标区与满足道路/物理条件分别存于summary，不把存活或参考弯道存在当作任务成功。原接触奖励仅作诊断，无滤波，累计自首转移求和。两条不是独立训练种子，不构成泛化或规划增益结果；未使用世界模型在线选档。\n\n原始记录：`'+str(b)+'`。协议与目标在运行前冻结，图和逐点距离由本目录analyze.py从原物理记录计算。\n'
(o/'RESULTS.md').write_text(report)
