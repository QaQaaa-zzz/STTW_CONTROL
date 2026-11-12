"""Summarize paired contact OAT sweeps, preserving cause, group and baseline identity."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from analyze_roll_contacts import roll_state

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--stage');a=p.parse_args();root=a.root;plan=json.loads((root/'protocol.json').read_text());cells=json.loads((root/'cells.json').read_text());vehicle={x['scale']:x for x in json.loads((root/'vehicle_geometry.json').read_text())};rows=[];out=root/'analysis';out.mkdir(exist_ok=True)
for cell in cells:
 folder=root/cell['stage'];jobs=json.loads((folder/'jobs.json').read_text())
 for job in [j for j in jobs if j['cell']==cell['id']]:
  run=folder/job['name']
  if not (run/'condition_metrics.json').exists():continue
  c=json.loads((run/'contract.json').read_text());z=np.load(run/'trajectory.npz');tr=z['lower_trace'].reshape(-1,z['reward'].shape[1],11);metrics=json.loads((run/'condition_metrics.json').read_text());front_index=c['bodies'].index('frontwheel');chasis=c['bodies'].index('chasis')
  for lane,m in enumerate(metrics[:len(plan['case_labels'])]):
   live=np.flatnonzero(tr[:,lane,6]);t=live*.005+tr[live,lane,7]*c['physics_dt'];f=np.linalg.norm(z['contact_force'][live,lane,front_index],axis=-1);phi,omega=roll_state(z['body_state'][live,lane,chasis]);off=np.flatnonzero(f<.1);segments=np.split(off,np.flatnonzero(np.diff(off)>1)+1);long=[s for s in segments if len(s)>=20]
   j=long[0][0] if long else None;reasons=[k for k in ('roll_limit_failure','contact_failure','domain_exit','numerical_failure','policy_fault') if z[k][live,lane].any()]
   alpha=plan['alphas'][lane];peak=float(np.max(abs(tr[live,lane,3])));tasks={name:bool(not m['failed'] and m['final_hold'] and peak<=criterion['W_m'] and m['speed_rmse_m_s']<=criterion['speed_rmse_m_s']) for name,criterion in json.loads((folder/'protocol.json').read_text()).get('tasks',{}).items()}
   row=dict(cell=cell['id'],stage=cell['stage'],factor=cell['factor'],value=cell['value'],center_s_m=cell['center_s_m'],length_m=cell['length_m'],height_m=cell['height_m'],wheelbase_scale=cell['wheelbase_scale'],oblique_s_per_n=cell['oblique_s_per_n'],reference_heading_deg=cell['reference_heading_deg'],physical_group=job['physical_seed'],lane=lane,label=plan['case_labels'][lane],alpha=alpha,full_window=int(not m['failed']),goal_crossed=int(m['goal_crossed']),final_hold=int(m['final_hold']),physical_termination=int(any(k in reasons for k in ('roll_limit_failure','contact_failure'))),roll_limit=int('roll_limit_failure' in reasons),nonwheel_contact=int('contact_failure' in reasons),domain_exit=int('domain_exit' in reasons),numerical=int('numerical_failure' in reasons),cause='+'.join(reasons) or 'full_window',observed_seconds=float(t[-1]),path_rmse_m=m['path_rmse_m'],speed_rmse_m_s=m['speed_rmse_m_s'],peak_roll_deg=float(np.degrees(max(abs(phi)))),front_unload_max_s=max(len(s) for s in segments)*.005,front_unload_fraction=float((f<.1).mean()),first_long_unload_s=float(t[j]) if j is not None else None,entry_speed_m_s=float(tr[live[j],lane,2]) if j is not None else None,entry_roll_deg=float(np.degrees(phi[j])) if j is not None else None,entry_euler_rate_deg_s=float(np.degrees(omega[j])) if j is not None else None,steer_command_at_limit_fraction=float((abs(z['final_command'][live,lane,0])>=2.999).mean()),**{f'task_{key}':int(value) for key,value in tasks.items()})
   axle=vehicle[cell['wheelbase_scale']]['zero_pose_projected_axle_distance_m'];row.update(axle_distance_m=axle,height_over_wheel_radius=cell['height_m']/.1,terrain_length_over_axle=cell['length_m']/axle,nominal_peak_grade=cell['height_m']*np.pi/cell['length_m'],axle_delay_at_request_s=axle/plan['speed_m_s'])
   rows.append(row)
if not rows:raise SystemExit('No completed case metrics available')
with (out/'episodes.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for cell in cells:
 for lane in range(6):
  group=[r for r in rows if r['cell']==cell['id'] and r['lane']==lane]
  if not group:continue
  n=len(group);count=sum(r['full_window'] for r in group);z=1.96;phat=count/n;mid=(phat+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n)
  summary.append(dict(**cell,lane=lane,label=group[0]['label'],n=n,full_window=count,full_rate=phat,descriptive_wilson95=[mid-half,mid+half],final_hold=sum(r['final_hold'] for r in group),physical_termination=sum(r['physical_termination'] for r in group),roll_limit=sum(r['roll_limit'] for r in group),nonwheel_contact=sum(r['nonwheel_contact'] for r in group),domain_exit=sum(r['domain_exit'] for r in group),front_unload_max_mean_s=float(np.mean([r['front_unload_max_s'] for r in group])),**{key:sum(r[key] for r in group) for key in group[0] if key.startswith('task_')}))
# Reuse the same original-placement baseline, without adding physical episodes.
def baseline_id(cell):
 if cell['factor'] in ('length','wheelbase'):return f'position_s{cell["center_s_m"]:g}'
 if cell['factor']=='flat_wheelbase':return 'flat_original'
 if cell['factor'] in ('obliquity','world_rotation'):return 'position_s4.2'
 return None
paired=[]
for cell in cells:
 base=baseline_id(cell)
 if base is None:continue
 for lane in range(6):
  changed={r['physical_group']:r for r in rows if r['cell']==cell['id'] and r['lane']==lane};original={r['physical_group']:r for r in rows if r['cell']==base and r['lane']==lane};groups=sorted(changed.keys()&original.keys())
  if not groups:continue
  paired.append(dict(cell=cell['id'],baseline_cell=base,label=plan['case_labels'][lane],n=len(groups),full_gains=sum(changed[g]['full_window']>original[g]['full_window'] for g in groups),full_losses=sum(changed[g]['full_window']<original[g]['full_window'] for g in groups),physical_failures_change=sum(changed[g]['physical_termination']-original[g]['physical_termination'] for g in groups),front_unload_max_change_s=float(np.mean([changed[g]['front_unload_max_s']-original[g]['front_unload_max_s'] for g in groups]))))
(out/'summary.json').write_text(json.dumps(dict(planned_episodes=plan['total_episodes'],recorded_episodes=len(rows),physical_blocks=plan['physical_groups'],cells=summary,paired_contrasts=paired,scope='Within-block one-factor contrasts; flat transfer controls and world-rotation symmetry controls separate. Observed contact statistics stop at actual termination. Wilson intervals are descriptive across fixed physical blocks, not training-seed replication.'),indent=2)+'\n')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160});colors=['#0072b2','#d55e00','#009e73','#cc79a7','#8a6500','#333333']
panels=[('position',None,'Position: reference heading (deg)'),('length',3.9,'Slope length at s=3.9 (m)'),('length',4.5,'Slope length at s=4.5 (m)'),('wheelbase',3.9,'Axle-distance scale at s=3.9'),('wheelbase',4.5,'Axle-distance scale at s=4.5'),('flat_wheelbase',None,'Axle-distance scale: flat control'),('obliquity',None,'Obstacle obliquity ds/dn'),('world_rotation',None,'Whole-scene rotation (deg)')]
trends=[]
for outcome in ('full_window','physical_termination'):
 fig,axs=plt.subplots(3,3,figsize=(16,12),layout='constrained')
 for ax,(factor,center,title) in zip(axs.ravel(),panels):
  for lane in range(6):
   points=[s for s in summary if s['factor']==factor and s['lane']==lane and (center is None or s['center_s_m']==center)]
   reference=next((s for s in summary if s['id']==('flat_original' if factor=='flat_wheelbase' else f'position_s{center:g}' if factor in ('length','wheelbase') else 'position_s4.2') and s['lane']==lane),None)
   if reference is not None and factor!='position':
    reference=dict(reference,value={'length':.8,'wheelbase':1,'flat_wheelbase':1,'obliquity':.25,'world_rotation':0}[factor]);points.append(reference)
   points.sort(key=lambda x:x['reference_heading_deg'] if factor=='position' else x['value'])
   x=[v['reference_heading_deg'] if factor=='position' else v['value'] for v in points];y=[v[outcome]/v['n'] for v in points]
   ax.plot(x,y,marker='o',color=colors[lane],ls='--' if lane==5 else '-',ms=4,label=plan['case_labels'][lane])
   if outcome=='full_window' and len(points)>2:
    rho=float(spearmanr(x,y).statistic) if np.std(y)>0 else None
    trends.append(dict(factor=factor,center_s_m=center,label=plan['case_labels'][lane],number_of_levels=len(points),descriptive_spearman_rho=rho))
  ax.set(title=title,ylim=(-.04,1.04),ylabel='Full 10 s fraction' if outcome=='full_window' else 'Physical-termination fraction');ax.grid(alpha=.2)
 axs.ravel()[-1].axis('off');handles,labels=axs.ravel()[0].get_legend_handles_labels();axs.ravel()[-1].legend(handles,labels,loc='center',fontsize=12)
 fig.suptitle('Paired one-factor sweep | 8 fixed physics blocks per level | shared baselines reused\n'+('Full window includes remaining within the tracking domain' if outcome=='full_window' else 'Roll limit or nonwheel contact; domain exit reported separately'),fontsize=14)
 fig.savefig(out/(outcome+'_trends.png'));plt.close(fig)
(out/'trends.json').write_text(json.dumps(trends,indent=2)+'\n')
lines=['# 单因素接触实验统计','','先看各阶段完整实际/参考XY，再看因素趋势。','']
for stage in json.loads((root/'queue/plan.json').read_text())['stages']:lines.append(f'- [{stage["name"]}全部XY与奖励](../{stage["name"]}/RESULTS.md)')
lines+=['',f'已读取{len(rows)}/{plan["total_episodes"]}条；每个参数单元8个固定物理组配对。完整10s、最终保持、侧倾阈值、非轮触地、跟踪域退出分别统计；域退出之后没有继续运行，不能据此宣称已验证余下时域不摔倒。','','![完整运行趋势](full_window_trends.png)','','![物理终止趋势](physical_termination_trends.png)','','|参数单元|档位|完整10s|最终保持|物理终止|域退出|最长卸载均值s|','|---|---|---:|---:|---:|---:|---:|']
for s in summary:lines.append(f'|{s["id"]}|{s["label"]}|{s["full_window"]}/{s["n"]}|{s["final_hold"]}/{s["n"]}|{s["physical_termination"]}/{s["n"]}|{s["domain_exit"]}/{s["n"]}|{s["front_unload_max_mean_s"]:.3f}|')
lines+=['','[逐回合时序摘要](episodes.csv) · [配对增益/损失与分母](summary.json) · [各档随单因素变化的秩趋势](trends.json)','','秩相关仅描述本次曲线，不预设单调方向，也不以整体相关替代逐档曲线。位置图横轴参考航向与弯道进度共变；整体旋转图单独检验坐标变换。车长试验保持原控制器，先查看对应平地控制能力再解释越野变化。各参数所致实际坡度、整体质心/惯量和轴距见上层geometry文件。']
(out/'README.md').write_text('\n'.join(lines)+'\n');print(f'Analyzed {len(rows)} episodes, {len(summary)} condition-alpha groups',flush=True)
