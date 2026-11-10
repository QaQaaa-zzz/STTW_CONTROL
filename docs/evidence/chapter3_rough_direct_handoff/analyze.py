"""All paired outcomes; no smoothing or post-hoc exclusions."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
b=Path('/media/qy/6A7E3DB07E3D75C3/STTW_runs/chapter3_rough_direct_handoff');o=Path(__file__).parent
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');rows=[];data={}
for name,color,style in [('capture','#0072B2','-'),('first_contact','#D55E00','--')]:
 p=b/name;d=np.load(p/'heading125/trajectory.npz');s=json.loads((p/'summary.json').read_text())['cases'][0];h=json.loads((p/'handoff.json').read_text()) if (p/'handoff.json').exists() else None
 t=d['time'];r=d['root'][:,0];reward=d['reward'][:,0]
 for ax,x,y in [(axes[0,0],r[:,0],r[:,1]),(axes[0,1],t,r[:,7]),(axes[1,0],t,reward),(axes[1,1],t,np.cumsum(reward))]:
  ax.plot(x,y,color=color,ls=style,label=name);ax.plot(x[-1],y[-1],'o' if s['ground_continued_to_horizon'] else 'x',color=color)
 rows.append(dict(case=name,handoff=h,summary=s));data.update({name+'_root':r,name+'_time':t,name+'_reward':reward})
 hist=np.load(p/'heading125/ground_controller_history.npz');cmd=np.load(p/'heading125/ground_commands.npz')
 np.testing.assert_array_equal(hist['time'],cmd['policy_start_time_s']);np.testing.assert_array_equal(hist['phase'],cmd['phase'])
 assert len(hist['controller_eso'])==len(cmd['commands'])
 print(name,'history rows',len(hist['time']),'duration',s['duration_s'],'end',s['terminal_code'],'handoff',h['time_s'] if h else None)
axes[0,0].axhline(0,color='k',ls=':',label='Fixed reference')
for lo,hi in [(2,4),(8,18)]:axes[0,0].axvspan(lo,hi,color='#009E73',alpha=.1)
axes[0,0].axvspan(6.3,6.6,color='0.6',alpha=.5);axes[0,0].set_aspect('equal',adjustable='box');axes[0,0].set(xlim=(1,18),ylim=(-2,2))
for ax,title,x,y in [(axes[0,0],'XY: green=rough, gray=ditch','World x (m)','World y (m)'),(axes[0,1],'Actual forward velocity','Time (s)','World vx (m/s)'),(axes[1,0],'Original contact-score diagnostic','Time (s)','Reward / step'),(axes[1,1],'Cumulative diagnostic reward','Time (s)','Reward')]:
 ax.set(title=title,xlabel=x,ylabel=y);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Isaac heading125/H75 | 0.5-1mm microstones outside flat jump region | same reset, nominal DEV\nApproach1.8m/s alpha0.5; continuation1.5m/s alpha0.5; seed20261109; o=horizon x=failure')
fig.savefig(o/'xy_and_rewards.png',dpi=150);np.savez_compressed(o/'plot_data.npz',**data);(o/'summary.json').write_text(json.dumps(rows,indent=2))
report='# 碎石接近/续行与直接触地交接\n\n![全部配对XY及奖励](xy_and_rewards.png)\n\n固定微凸石面高0.5～1mm，仅x2～4m和8～18m；起跳/落地x4～8m平坦。同高30cm沟6.3～6.6m，直道共同起点x1.5/vx2，无快照。heading125/H75，名义条件，0训练。地面前进请求1.8m/s、α.5，回交后1.5m/s、α.5。两条DEV，原物理终止限制保留。\n\n|交接判据|实际交接时间(s)|实际阶段|终止时间(s)|终止码|续行至8s|终点y(m)|\n|---|---:|---:|---:|---:|---|---:|\n'
for row in rows:
 s=row['summary'];h=row['handoff'];report+=f"|{row['case']}|{h['time_s'] if h else '未交接'}|{h['stage'] if h else '—'}|{s['duration_s']:.2f}|{s['terminal_code']}|{s['ground_continued_to_horizon']}|{s['final_xyz'][1]:.3f}|\n"
report+='\n首次远岸触地使用原已确认腾空尝试的远岸轮地接触，在下一个20ms策略边界切回STTW，不等待双轮捕获或1秒支撑；不声称物理采样瞬间零延迟切换。绿色区域为碎石，灰色为开口沟；圆点为观察窗结束，叉号为真实失败。全部曲线无平滑、累计自首实际转移求和、真实终点截断。原接触评分仍是诊断，不替代后续任务结果。没有独立ECBC基线或泛化测试。\n\n控制器历史与执行命令批次时间/phase逐值核验；记录包含ESO、过滤状态和观测历史，尚不包含PhysX求解器缓存。数据盘原始轨迹：`'+str(b)+'`。脚本、摘要及绘图原数组随报告保留。\n'
(o/'RESULTS.md').write_text(report)
