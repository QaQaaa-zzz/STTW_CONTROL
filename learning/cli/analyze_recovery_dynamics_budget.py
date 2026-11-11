"""No-fit roll-rate change budgets and actual steering execution from archived contact cases."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from analyze_roll_contacts import nominal_coeff
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def integrate(x,t):return float(np.sum(.5*(x[1:]+x[:-1])*np.diff(t)))

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--controller',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=json.loads(a.controller.read_text());rows=[];links=[]
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
 for path in sorted(a.source.glob('*_seed*.npz')):
  z=np.load(path);fields=list(z['contact_fields']);labels=list(z['labels']);show=['full_0','full_0.25','full_0.5','full_0.75','full_1','Zero'];fig,axs=plt.subplots(6,4,figsize=(19,15),layout='constrained')
  for lane,label in enumerate(labels):
   valid=z['valid'][:,lane];t=z['time'][valid,lane];phi=z['phi'][valid,lane];w=z['euler_roll_rate'][valid,lane];delta=z['delta'][valid,lane];dr=z['delta_dot'][valid,lane];cmd=z['command'][valid,lane];contact=z['contact_mean'][valid,lane];v=z['speed'][valid,lane];c,b,gravity=nominal_coeff(v,cfg);terms=np.stack([gravity*phi,b*delta,c*dr],-1)
   windows={'last100ms':(t[-1]-.1,t[-1])}
   events={}
   for wheel in ['front','rear']:
    hit=np.flatnonzero(contact[:,fields.index(wheel+'_raised')]>0)
    if len(hit):
     events[wheel]=float(t[hit[0]]);windows[wheel+'_contact100ms']=(events[wheel],events[wheel]+.1)
   for start in [1.5,1.8,1.9,2.1,2.3]:windows[f'common_{start:g}']=(start,start+.1)
   for name,(lo,hi) in windows.items():
    ids=np.flatnonzero((t>=lo-1e-8)&(t<=hi+1e-8))
    if len(ids)<2:continue
    tt=t[ids];parts=np.array([integrate(terms[ids,k],tt) for k in range(3)]);actual=float(w[ids[-1]]-w[ids[0]]);residual=actual-parts.sum();cm=contact[ids];moment=cm[:,4]+cm[:,5]+cm[:,10]+cm[:,11]
    rows.append(dict(case=path.stem,label=str(label),window=name,start_s=float(tt[0]),end_s=float(tt[-1]),complete_window=bool(tt[-1]-tt[0]>=.099),observation_end_s=float(t[-1]),phi_start_deg=float(np.degrees(phi[ids[0]])),phi_end_deg=float(np.degrees(phi[ids[-1]])),omega_start_deg_s=float(np.degrees(w[ids[0]])),omega_end_deg_s=float(np.degrees(w[ids[-1]])),actual_change_deg_s=float(np.degrees(actual)),gravity_change_deg_s=float(np.degrees(parts[0])),steer_angle_change_deg_s=float(np.degrees(parts[1])),steer_rate_change_deg_s=float(np.degrees(parts[2])),unmodeled_change_deg_s=float(np.degrees(residual)),speed_mean_m_s=float(v[ids].mean()),command_mean_rad_s=float(cmd[ids].mean()),actual_steer_rate_mean_rad_s=float(dr[ids].mean()),rate_tracking_MAE_rad_s=float(abs(cmd[ids]-dr[ids]).mean()),rate_limit_fraction=float((abs(cmd[ids])>=2.999).mean()),delta_start_rad=float(delta[ids[0]]),delta_end_rad=float(delta[ids[-1]]),contact_roll_impulse_Nms=integrate(moment,tt) if np.isfinite(moment).all() else None,front_contact_s=events.get('front'),rear_contact_s=events.get('rear')))
   if label not in show:continue
   row=show.index(label);mask=(t>=1.1)&(t<=3.3);tt=t[mask]
   axs[row,0].plot(tt,np.degrees(phi[mask]),color='#0072B2',label='Roll angle (deg)');axs[row,0].set_ylabel(str(label));axs[row,0].axhline(40.107,color='.5',ls=':',lw=.7);axs[row,0].axhline(-40.107,color='.5',ls=':',lw=.7)
   axs[row,1].plot(tt,np.degrees(w[mask]),color='#0072B2',label='Euler roll rate (deg/s)')
   axs[row,2].plot(tt,cmd[mask],label='Command',color='#D55E00');axs[row,2].plot(tt,dr[mask],label='Actual',ls='--',color='#0072B2');axs[row,2].axhline(3,color='.5',ls=':',lw=.7);axs[row,2].axhline(-3,color='.5',ls=':',lw=.7)
   axs[row,3].plot(tt,contact[mask,4]+contact[mask,5],label='Front',color='#E69F00');axs[row,3].plot(tt,contact[mask,10]+contact[mask,11],label='Rear',color='#009E73')
   for ax in axs[row]:
    ax.set_xlim(1.1,3.3)
    for wheel,event in events.items():ax.axvline(event,color='.6',ls=':' if wheel=='front' else '--',lw=.6)
    if t[-1]<3.3:ax.axvline(t[-1],color='#CC3311',lw=.8)
    if row==5:ax.set_xlabel('Time (s)')
  for ax,title in zip(axs[0],['Roll (deg)','Euler roll rate (deg/s)','Steering rate (rad/s)','Contact roll moment (Nm)']):ax.set_title(title);ax.legend(fontsize=7)
  fig.suptitle(path.stem+' | all five alpha + Zero | same original start\nGray: first raised-wheel contacts; red: actual terminal time; contact moment about total COM');name=path.stem+'_dynamics.png';fig.savefig(a.output/name,dpi=140);plt.close(fig);links.append((path.stem,name))
 with (a.output/'budgets.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
 (a.output/'definition.json').write_text(json.dumps(dict(formula='Delta Euler omega = integral(a4 phi + a2(v) delta + a1(v) actual_delta_dot)dt + unmodeled residual',integration='trapezoid over actual saved timestamps; no smoothing or fitted coefficients',residual='model discrepancy including contact, variable-speed/coupling/nonlinear effects and sampling; NOT identified pure terrain impulse',windows='first front/rear raised contact+100ms; fixed1.5/1.8/1.9/2.1/2.3+100ms; last100ms. Truncated windows retained and flagged.',sources=str(a.source),controller=str(a.controller),rows=len(rows),new_simulations=0),indent=2))
 lines=['# S2：实际转向执行与侧倾变化预算','','先看[原始全部XY]('+str(a.source/'INDEX.md')+')。本次读取全部216条，不拟合系数，不新增仿真。','','名义预算：Δω=∫a4φdt+∫a2(v)δdt+∫a1(v)δ̇dt+R。ω使用Euler侧倾导数，δ̇使用实际关节速度。R是完整运动相对名义方程的差额，包含接触和模型遗漏，不能直接命名为地形冲量。','','## 单脊2.6m/s：失败前100ms','','|组|α|终点s|实际Δω °/s|重力项|转角项|转速项|差额R|命令/实际转速 rad/s|','|---|---|---:|---:|---:|---:|---:|---:|---|']
 for x in rows:
  if x['case'].startswith('single10_v2.6') and x['label'] in ['full_0','full_1'] and x['window']=='last100ms':
   lines.append(f'|{x["case"].split("seed")[-1]}|{x["label"]}|{x["end_s"]:.3f}|{x["actual_change_deg_s"]:.1f}|{x["gravity_change_deg_s"]:.1f}|{x["steer_angle_change_deg_s"]:.1f}|{x["steer_rate_change_deg_s"]:.1f}|{x["unmodeled_change_deg_s"]:.1f}|{x["command_mean_rad_s"]:.3f}/{x["actual_steer_rate_mean_rad_s"]:.3f}|')
 lines+=['','α0三组和α1的101/102组终止前负向倾倒，命令−3rad/s时实际转向基本跟随；不能归因为转向命令没有执行。α1组103为跟踪域退出，最后窗口已经向正侧回转，单列而不解释为负向倾倒。','','以101为例：α0最后100ms，转角项与转速项合计正向约164°/s，重力项负向约104°/s；实际却负向改变约96°/s。α1对应转向两项合计约41°/s，重力负向约85°/s，实际负向改变383°/s。两者名义修正能力和实际净变化明显不同，后续必须沿接触/多体动力学解释R，不能仅用命令饱和宣称必败。','','这些末窗用于定位各自失败过程，不作为成功/失败同状态因果对照。budgets.csv另外保留所有档的共同固定窗口和按轮接触对齐窗口。','','## 全12工况五档/Zero时序']
 for case,name in links:lines += ['',f'### {case}',f'[XY]({a.source}/{case}_xy.png)',f'![接触、侧倾与实际转向]({name})']
 (a.output/'README.md').write_text('\n'.join(lines)+'\n');print('budget rows',len(rows))
if __name__=='__main__':main()
