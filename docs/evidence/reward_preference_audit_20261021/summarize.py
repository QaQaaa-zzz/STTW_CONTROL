"""Render reviewed offline reward scores; read-only with respect to original runs."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).parent
DATA=OUT.parents[2]/'runs/reward_preference_audit_20261021'
r=json.loads((OUT/'results.json').read_text());rec=r['records'];pairs=r['pairs']
OLD=Path(r['source_manifests'][0]).parent;NEW=Path(r['source_manifests'][1]).parent
sets=[('coverage_dual_u200',{'policy0':OLD/'alpha0/final_best','policy1':OLD/'alpha1/final_best','B0':OLD/'zero_upper'},['left90_R2_2.6','right90_R2_2.6']),('high_demand_u150',{**{f'policy{a:g}':NEW/'shared/final_best'/k for a,k in zip([0,.25,.5,.75,1],['alpha_0','alpha_0p25','alpha_0p5','alpha_0p75','alpha_1'])},'B0':NEW/'zero_upper'},['left90_3.2','right90_3.2']),('high_demand_u200',{**{f'policy{a:g}':NEW/'shared/validation/update_0200'/k for a,k in zip([0,.25,.5,.75,1],['alpha_0','alpha_0p25','alpha_0p5','alpha_0p75','alpha_1'])},'B0':NEW/'zero_upper'},['left90_3.2','right90_3.2'])]
fig,axes=plt.subplots(3,2,figsize=(12,13),layout='constrained')
for row,(label,dirs,cases) in enumerate(sets):
 for ax,case in zip(axes[row],cases):
  p=np.load(dirs['policy0']/(case+'_path.npz'));ix=np.flatnonzero(abs(p['curvature'])>1e-8);start=float(p['s'][ix[0]]);end=float(p['turn_end']);mask=(p['s']>=start-2)&(p['s']<=end+7);xy=p['xy'][mask]
  ax.plot(xy[:,0],xy[:,1],'k--',label='original reference')
  for method,folder in dirs.items():
   d=np.load(folder/(case+'.npz'));ax.plot(d['actual_xy'][:,0],d['actual_xy'][:,1],label=method,lw=1.3)
   if np.any(d['physical_failure']) or np.any(d.get('tracking_domain_failure',d.get('domain_exit',False))):ax.plot(*d['actual_xy'][-1],'x',color='red')
  lo=xy.min(axis=0)-1.5;hi=xy.max(axis=0)+1.5;ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect('equal');ax.grid(alpha=.2);ax.set_xlabel('world X (m)');ax.set_ylabel('world Y (m)');ax.set_title(label+' | '+case);ax.legend(fontsize=7)
fig.suptitle('Actual XY and fixed reference | bend/exit zoom | frozen local300 | no new simulation')
fig.savefig(OUT/'XY_TURN_EXIT.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
for row,(label,dirs,cases) in enumerate(sets):
 for ax,case in zip(axes[row],cases):
  ps=[p for p in pairs if p['dataset']==label and p['case']==case]
  ax.plot([p['score_alpha'] for p in ps],[p['delta_return_policy1_minus_policy0'] for p in ps],'o-',label='whole 20 s')
  ax.plot([p['score_alpha'] for p in ps],[p['window_deltas']['turn'] for p in ps],'o-',label='spatial turn only')
  ax.axhline(0,color='k',lw=.8);ax.set_title(label+' | '+case);ax.set_xlabel('scoring alpha (fixed behavior)');ax.set_ylabel('return(policy1) - return(policy0)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Positive = prefers policy1; negative = prefers policy0 | CURRENT reward')
fig.savefig(OUT/'PREFERENCE_GAPS.png',dpi=150);plt.close(fig)
# Complete per-scene component/physical diagnostics. All declared behavior alphas,
# both endpoint scores; every intermediate score has raw components in the NPZ.
components=np.load(DATA/'reward_components.npz')
for label,dirs,_ in sets:
 cases=sorted({x['case'] for x in rec if x['dataset']==label})
 for case in cases:
  fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
  for method,folder in dirs.items():
   d=np.load(folder/(case+'.npz'));t=d['time']+.005
   for ax,key in zip(axes[0],['actual_forward_speed','path_cross_track']):ax.plot(t,d[key],label=method,lw=.9)
   axes[1,0].plot(t,d['peak_roll'],label=method,lw=.9)
   for a in (0,1):
    pref=f'{label}/{case}/{method}/score{a}/';ls='-' if a==0 else '--'
    for ax,names in [(axes[1,1],['speed']), (axes[2,0],['path','primary']), (axes[2,1],['roll','working_roll'])]:
     y=sum(components[pref+k] for k in names);ax.plot(t,y,ls=ls,lw=.8,label=f'{method}:score{a}')
  for ax,title in zip(axes.flat,['actual speed (m/s)','signed lateral error (m)','peak roll (rad)','speed reward / lower tick','path + primary reward / lower tick','roll + working reward / lower tick']):
   ax.set_title(title);ax.set_xlabel('time (s)');ax.grid(alpha=.2)
  axes[1,0].axhline(.30,color='red',ls=':');axes[0,0].legend(fontsize=6,ncol=2);axes[1,1].legend(fontsize=5,ncol=2)
  fig.suptitle(label+' | '+case+' | solid score alpha0, dashed score alpha1 | no external disturbance')
  fig.savefig(OUT/(label+'_'+case+'_diagnostics.png'),dpi=110);plt.close(fig)
lines=['# 当前奖励函数离线偏好验证','', '![原始路径与实际轨迹：过弯/出弯局部放大](XY_TURN_EXIT.png)','',
'**结论：当前奖励存在局部的速度—路径偏好，但对已知历史保速轨迹，整回合并未形成两端相反的排序。主要反例是出弯恢复代价抵消弯内保速收益；不能把问题一概归因于侧倾惩罚，也不能据此断言奖励无效。**','',
'## 范围与核验','',
'仅CPU重评分，无训练、无新仿真、无Actor反事实闭环。历史Coverage双网u200、当前HighDemand best150和last200分开；每组6场景，分别3/6/6个方法（含匹配local300 B0），共90条实际轨迹，五个评分α，共450次重评分。历史数据角色为已用开发诊断，不是独立holdout。路径逐方法核对一致；冻结底层Actor manifest身份与时序/terminal_contract_v2相同，未重新扫描SHA。不同数据组不直接比较回报。','',
'当前奖励为实际manifest中的原Coverage公式（reward_version字段缺省），roll/work权重80/240、cap1800/6000、failure bound8980.2；与旧配置其余奖励参数相同。每条固定行为分别代入0/.25/.5/.75/1，先计算含α代价再限顶，不混合端点最终奖励。变化率按完整20ms边界重建，5ms积分，保留失败区间替换尾罚。','',
'以各原配置/原α核对90条日志：正常步最大误差'+f"{max(a['normal_max_error'] for a in r['audits']):.3g}"+'。失败尾罚float64重算对日志最大绝对差'+f"{r['native_audit_max_abs_tick_error']:.5f}"+'，来自γ幂的float32表示；按运行时精度独立复算最大差'+f"{max(a['failure_float32_error'] or 0 for a in r['audits']):.6f}"+'。不修改原日志，不把精度差冒称奖励定义错误。主要策略比较均完整20s、无失败，不受尾罚精度影响。','',
'报告主分数为实际20s累计奖励；另保存按PPO γ=.997每20ms、从reset开始的折扣回报，两者所有端点排序方向一致。弯内窗口是各自投影位于同一原始几何弯道范围的实际样本，观察时间可能不同；窗口分数仅描述局部收益，不替代整回合目标或等空间误差。','',
'## 1. 中等难度的关键反例','',
'下表均为R2m、请求2.6m/s。A=历史α0策略实际轨迹；B=历史α1策略实际轨迹。差值为B减A，正值偏爱B；所有项均重新使用当前加强侧倾后的奖励。','',
'| 场景 | 评分α | 弯内奖励差 B−A | 出弯后奖励差 B−A | 全20s奖励差 B−A | reset折扣差 B−A |','|---|---:|---:|---:|---:|---:|']
for p in pairs:
 if p['dataset']=='coverage_dual_u200' and p['case'] in ('left90_R2_2.6','right90_R2_2.6') and p['score_alpha'] in (0,1):lines.append(f"| {p['case']} | {p['score_alpha']:g} | {p['window_deltas']['turn']:+.2f} | {p['window_deltas']['exit']:+.2f} | {p['delta_return_policy1_minus_policy0']:+.2f} | {p['delta_discounted_policy1_minus_policy0']:+.2f} |")
lines+=['','在α0评分下，弯内偏爱A；α1评分下，弯内偏爱B，**局部取舍方向成立**。但α1评分的整回合仍偏爱A：左弯弯内收益+11.90被出弯后−35.53抵消；右弯+4.84被−12.88抵消。出弯后χ回落时两端均要求路径恢复，历史B恢复较慢且偏离较大。此处不能通过删除出弯路径要求来人为制造偏好。','',
'| 场景 | 行为 | 弯内速度RMSE m/s | 弯内路径RMSE m | 出弯后路径RMSE m | 峰值侧倾 rad | 出弯至连续1s共同误差入带 s |','|---|---|---:|---:|---:|---:|---:|']
for x in rec:
 if x['dataset']=='coverage_dual_u200' and x['case'] in ('left90_R2_2.6','right90_R2_2.6') and x['method'] in ('policy0','policy1') and x['score_alpha']==1:
  v=x['physical'];lines.append(f"| {x['case']} | {x['method']} | {v['turn']['speed_rmse']:.3f} | {v['turn']['path_rmse']:.3f} | {v['exit']['path_rmse']:.3f} | {v['whole']['peak_roll']:.3f} | {x['recovery_after_turn_s']:.3f} |")
lines+=['','共同误差入带为速度≤.1m/s、横向≤.1m、航向≤.05rad连续1s的描述性恢复时长，从几何出弯开始；不替换既有goal/资格判断。以上四条均无物理/域失败、完整20s、最终保持通过、峰值低于.30rad。左弯A/B均未触发.26侧倾代价，右弯B仅比A多3.316的整回合侧倾损失；因此并非新增侧倾惩罚普遍压掉历史偏好。','',
'四个历史转弯（含2.0m/s）在整回合及reset折扣评分下，五个评分α全部更偏爱A。2.6m/s两弯弯内在α0到α1间发生排序翻转；2.0m/s两弯没有翻转。直线χ=0时，同一物理轨迹改变评分α的分数完全不变。','',
'## 2. 当前高难度结果','',
'u150所有6场景、五个评分α均更偏爱其policy0而不是policy1。u200在2.8m/s左右弯和两条直线均更偏爱policy0；3.2m/s左右弯五个评分α均更偏爱policy1，但理由是更小的侧倾损失，并非目标速度偏好已经学成。','',
'α1评分下，u200左3.2弯policy1相对policy0的侧倾两项收益+15.72，速度/路径/primary合计−7.70，净收益+8.02；右弯对应+14.24、−5.16、+9.08。两个评分端点均选同一行为，仍无目标排序翻转。两个高速场景的源策略均有工作侧倾超限，不作为共同合格的保速/保路径方案。','',
'![五个评分α下端点行为的奖励差](PREFERENCE_GAPS.png)','',
'## 3. 下一步判断','',
'本次证据不足以给出“当前奖励已验证能引导完整目标，直接课程即可”的结论。它支持：弯内偏好机制存在，已知保速轨迹的出弯恢复不足，且高难度当前候选没有体现目标差异。也不能由有限候选证明不存在更优的保速且快速回线路径。','',
'建议保留当前侧倾要求，先把中等难度的“保速过弯+及时回线”作为下一次短课程的诊断目标，α从头参与；升级同时检查侧倾、弯内速度/路径和出弯恢复。若能找到满足这些要求的匹配轨迹而α1依然不偏爱它，再根据分项调整奖励。不要因本次B的总分低就删除出弯跟踪要求，也不要只增加训练轮数或继续加重侧倾惩罚。新课程尚未执行。','',
'## 全覆盖证据','',
'[450次重评分/原始身份/90条重建核验](results.json) · [端点差值CSV](pair_scores.csv) · [本地：全部五评分α的逐步有符号奖励分项](/home/qy/STTW_CONTROL/runs/worktrees/path-conflict-exposure-v1/runs/reward_preference_audit_20261021/reward_components.npz) · [重评分脚本](analyze.py) · [报告绘图脚本](summarize.py)','']
for label,dirs,_ in sets:
 lines += [f'### {label}','',f'[六场景XY]({label}_XY.png) · [每步总奖励]({label}_step_rewards.png) · [累计总奖励]({label}_cumulative_rewards.png)','']
 for case in sorted({x['case'] for x in rec if x['dataset']==label}):lines.append(f'- [{case}：速度/路径/侧倾/跟踪及侧倾分项]({label}_{case}_diagnostics.png)')
 lines.append('')
(OUT/'INDEX.md').write_text('\n'.join(lines)+'\n')
print('report complete; recovery:')
for x in rec:
 if x['dataset']=='coverage_dual_u200' and x['score_alpha']==1 and x['method'] in ('policy0','policy1') and '2.6' in x['case'] and 'straight' not in x['case']:print(x['case'],x['method'],x['recovery_after_turn_s'])
