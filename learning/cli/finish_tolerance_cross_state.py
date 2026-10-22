"""Collect finite-grid labels and frozen linear-probe diagnostics, separated by speed role."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sttw_matplotlib')
import sys,json,time,fcntl,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.preference_best import atomic_json
from sttw_control.empirical_tolerance import summarize_direction
from sttw_control.tolerance_batch_layout import contexts,case_name,ONSETS,FORCES
root=Path(sys.argv[1]).resolve()
def status(phase,**kw):atomic_json(root/'analysis_status.json',dict(phase=phase,checked_unix=time.time(),**kw))
def signals(d,speed):
 tail=d['time']+.005>19.+1e-5
 return [(d['phi'],.302),((d['actual_forward_speed']-speed)[tail],.1),(d['path_cross_track'][tail],.1),(d['path_heading_error'][tail],.05)]
status('waiting_for_execution')
try:
 while True:
  s=json.loads((root/'status.json').read_text())
  if s['phase'] in ['completed','error','failed']:break
  time.sleep(10)
 if s['phase']!='completed':raise RuntimeError('execution failed')
 status('running',stage='probe_predictions');predictions=[]
 for c in contexts():
  for onset in ONSETS:
   d={f:dict(np.load(root/(case_name(dict(**c,force_N=f),onset)+'.npz'))) for f in [0,.5,-.5]}
   key=dict(**c,onset_s=onset)
   if any(len(a['time'])!=4000 for a in d.values()):predictions.append(dict(**key,status='UNKNOWN_INCOMPLETE_PROBE'));continue
   nom=signals(d[0],c['speed']);plus=signals(d[.5],c['speed']);minus=signals(d[-.5],c['speed']);slope=[p[0]-m[0] for p,m in zip(plus,minus)];valid=all(np.all(abs(x)<=limit) for x,limit in nom)
   for sign in [-1,1]:
    limits=[]
    for (x,limit),q in zip(nom,slope):
     for side in [-1,1]:
      den=side*sign*q;mask=den>0
      if mask.any():limits.append(float(np.min((limit-side*x[mask])/den[mask])))
    predictions.append(dict(**key,sign=sign,status='PREDICTED' if valid else 'INVALID_NOMINAL',bound_N=min(limits) if valid and limits else None))
 atomic_json(root/'probe_predictions.json',dict(predictions=predictions,note='Fixed +/-0.5N central difference; no fitted correction. Written before reading larger-force rows. Values beyond 0.5N are unverified extrapolation.'))
 rows=json.loads((root/'results.json').read_text())['cases'];assert len(rows)==264;groups=[];validation=[]
 for c in contexts():
  for onset in ONSETS:
   matching=[r for r in rows if r['route']==c['route'] and r['speed']==c['speed'] and r['alpha']==c['alpha'] and r['onset_s']==onset];assert len(matching)==11
   nominal=next(r for r in matching if r['force_N']==0)
   for sign in [-1,1]:
    trial=[r for r in matching if r['force_N']==0 or r['force_N']*sign>0];groups.append(dict(**c,onset_s=onset,sign=sign,**summarize_direction([(abs(r['force_N']),r['success']) for r in trial])))
   for row in matching:
    if abs(row['force_N'])<=.5:continue
    pr=next((p for p in predictions if all(p[k]==c[k] for k in ['route','speed','alpha']) and p['onset_s']==onset and p.get('sign')==int(np.sign(row['force_N']))),None)
    pred=None if not nominal['success'] or pr is None or pr['status']!='PREDICTED' or pr['bound_N'] is None else abs(row['force_N'])<=pr['bound_N']
    validation.append(dict(**row,predicted_pass=pred))
 stats={}
 for role in ['calibration','speed_transfer']:
  rr=[r for r in rows if r['data_role']==role];vv=[r for r in validation if r['data_role']==role and r['predicted_pass'] is not None]
  stats[role]=dict(cases=len(rr),successes=sum(r['success'] for r in rr),physical_failures=sum(r['physical_failure'] for r in rr),working_failures=sum(r['working_failure'] for r in rr),domain_failures=sum(r['domain_failure'] for r in rr),nominal_failures=sum(not r['success'] for r in rr if r['force_N']==0),scored=len(vv),correct=sum(r['predicted_pass']==r['success'] for r in vv),false_safe=sum(r['predicted_pass'] and not r['success'] for r in vv),conservative=sum(not r['predicted_pass'] and r['success'] for r in vv))
 atomic_json(root/'summary.json',dict(stats=stats,groups=groups,validation=validation,scope='DEV calibration and speed-transfer diagnostic; no continuous safety claim or final-policy improvement'))
 images=[]
 for speed in [2.3,2.8]:
  for route in ['left90_R2','right90_R2']:
   fig,axes=plt.subplots(3,4,figsize=(20,12));path=np.load(root/f'{route}_path.npz')
   for oi,onset in enumerate(ONSETS):
    for alpha in [0,1]:
     ax,ar=axes[oi,alpha*2:alpha*2+2]
     for row in rows:
      if row['route']!=route or row['speed']!=speed or row['alpha']!=alpha or row['onset_s']!=onset:continue
      d=np.load(root/(row['name']+'.npz'));sel=d['time']<7;xy=d['actual_xy'][sel];label=f"{row['force_N']:+g} {'P' if row['success'] else 'F'}";ax.plot(xy[:,0],xy[:,1],lw=.8,label=label);ar.plot(d['time'],np.rad2deg(d['peak_roll']),lw=.8,label=label)
      if row['physical_failure'] or row['domain_failure']:ax.scatter(*d['actual_xy'][-1],marker='x',s=18)
     mask=path['s']<18;ax.plot(path['xy'][mask,0],path['xy'][mask,1],'k--',lw=1,label='original path');ax.set_aspect('equal',adjustable='datalim');ax.set(xlabel='X [m]',ylabel='Y [m]',title=f'alpha{alpha} onset{onset:g}s XY (first7s)');ar.axhline(np.rad2deg(.302),color='k',ls='--');ar.set(xlabel='episode [s]',ylabel='peak roll [deg]',xlim=(0,7),title='P=pass F=fail')
     for axis in [ax,ar]:axis.legend(fontsize=5,ncol=3);axis.grid(alpha=.2)
   fig.suptitle(f'Frozen Coverage u200 seed910101/local300 | {route}, {speed:g} m/s | 0.1s pulse');fig.tight_layout();name=f'{route}_v{speed:g}_xy_roll.png';fig.savefig(root/name,dpi=130);plt.close(fig);images.append(name)
 lines=['# Cross-state directional tolerance calibration','',*[f'![XY and roll]({im})' for im in images],'', 'All original frozen policies, physics and success criteria retained; no training.', '', '```json',json.dumps(stats,indent=2),'```','', '2.3m/s is calibration; 2.8m/s is speed-transfer DEV. Predictions are local extrapolations from +/-0.5N, not certified bounds. Report nominal failures separately. Do not fit on speed-transfer outcomes and continue calling them untouched validation. See summary.json for all finite-grid statuses and predictions.']
 (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
 hub=Path('/home/qy/STTW_CONTROL/research-hub')
 with (hub/'.write.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);(hub/'AGENTS.md').read_text();p=hub/'PROJECT_STATE.md';text=p.read_text();note=f"\n\n### 2026-10-22 跨状态容扰度批量标定自动回收\n\nsource_type=repo_artifact; {root}，264条完成，实际lower ticks{s['active_lower_ticks']}，preflight704，训练0。按速度分组统计{json.dumps(stats,ensure_ascii=False)}。XY和完整标签见REPORT.md/summary.json，图待视觉检查。2.3标定/2.8速度迁移DEV，不是最终TEST、连续安全认证或训练效果。下一步核验假安全/名义失败和分组误差，决定训练代理，不自动采用线性外推。\n";text=text.replace('## 2. STTW_CONTROL','## 2. STTW_CONTROL'+note,1);text=re.sub(r'state_revision: (\d+)',lambda m:f'state_revision: {int(m[1])+1}',text,count=1);tmp=p.with_suffix('.cross-result.tmp');tmp.write_text(text);tmp.replace(p)
 status('completed',report=str(root/'REPORT.md'))
except BaseException as e:status('error',error=f'{type(e).__name__}: {e}');raise
