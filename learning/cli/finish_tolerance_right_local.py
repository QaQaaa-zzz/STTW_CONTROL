"""Frozen predictor, right-turn local validity. No fitting on validation amplitudes."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sttw_matplotlib')
import json,time,sys,fcntl,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]).resolve()
def write(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def status(phase,**kw):write(root/'analysis_status.json',dict(phase=phase,checked_unix=time.time(),**kw))
def signals(d):
 tail=d['time']+.005>19.+1e-5
 return [(d['phi'],.302,'roll'),((d['actual_forward_speed']-2.6)[tail],.1,'speed_hold'),(d['path_cross_track'][tail],.1,'cross_track_hold'),(d['path_heading_error'][tail],.05,'heading_hold')]
status('waiting_for_execution')
try:
 while True:
  s=json.loads((root/'status.json').read_text())
  if s['phase'] in ('completed','error','failed'):break
  time.sleep(10)
 if s['phase']!='completed':raise RuntimeError('execution did not complete')
 status('running',stage='probe_predictions')
 predictions=json.loads((root/'frozen_predictions_before_execution.json').read_text())['predictions']
 status('running',stage='validation')
 rows=json.loads((root/'results.json').read_text())['cases'];assert len(rows)==14
 validation=[]
 for r in rows:
  if r['force_N']==0:continue
  pr=next((p for p in predictions if p['alpha']==r['alpha'] and p.get('sign')==int(np.sign(r['force_N']))),None)
  nominal_ok=next(x for x in rows if x['alpha']==r['alpha'] and x['force_N']==0)['success']
  pred=None if not nominal_ok or pr is None or pr['status']!='PREDICTED' else abs(r['force_N'])<=pr['bound_N']
  validation.append(dict(name=r['name'],alpha=r['alpha'],force_N=r['force_N'],predicted_pass=pred,actual_pass=r['success'],peak_roll=r['peak_roll'],working_failure=r['working_failure'],physical_failure=r['physical_failure'],domain_failure=r['domain_failure'],final_hold=r['final_hold']))
 known=[r for r in validation if r['predicted_pass'] is not None]
 replay_errors={}
 for alpha in [0,1]:
  a=np.load(root/f'alpha{alpha}_t1_F+0.npz');b=np.load(root.parent/'right_transfer_v1'/f'alpha{alpha}_t1_F+0.npz');replay_errors[str(alpha)]=float(np.max(abs(a['phi']-b['phi'])))
  assert replay_errors[str(alpha)]==0.,('zero replay mismatch',alpha,replay_errors[str(alpha)])
 result=dict(baseline_replay_phi_errors=replay_errors,cases=14,successes=sum(r['success'] for r in rows),predictions=predictions,validation=validation,scored=len(known),correct=sum(r['predicted_pass']==r['actual_pass'] for r in known),unsafe_predictions=sum(r['predicted_pass'] and not r['actual_pass'] for r in known),conservative_predictions=sum(not r['predicted_pass'] and r['actual_pass'] for r in known),scope='new route for tolerance calibration, not necessarily unseen by pretrained policies; two deterministic DEV states; no continuous certification')
 write(root/'summary.json',result)
 fig,axes=plt.subplots(2,2,figsize=(13,9));path=np.load(root/'path.npz')
 for alpha in [0,1]:
  for r in rows:
   if r['alpha']!=alpha:continue
   d=np.load(root/(r['name']+'.npz'));sel=d['time']<6
   axes[alpha,0].plot(d['actual_xy'][sel,0],d['actual_xy'][sel,1],label=f"{r['force_N']:+g} N {'pass' if r['success'] else 'fail'}")
   axes[alpha,1].plot(d['time'],np.rad2deg(d['peak_roll']),label=f"{r['force_N']:+g} N")
  mask=path['s']<15;axes[alpha,0].plot(path['xy'][mask,0],path['xy'][mask,1],'k--',label='original reference');axes[alpha,0].set_aspect('equal',adjustable='datalim');axes[alpha,0].set(xlabel='world X [m]',ylabel='world Y [m]',title=f'alpha{alpha} | right90_R2 | first 6s')
  axes[alpha,1].axhline(np.rad2deg(.302),color='k',ls='--');axes[alpha,1].set(xlabel='episode time [s]',ylabel='substep peak absolute roll [deg]',xlim=(.8,4),title=f'alpha{alpha} | 0.1s force from t=1s')
 for ax in axes.flat:ax.legend(fontsize=7);ax.grid(alpha=.2)
 fig.suptitle('Frozen Coverage u200 seed910101 + local300 | right-turn local validity');fig.tight_layout();fig.savefig(root/'xy_and_roll.png',dpi=160);plt.close(fig)
 lines=['# Right-turn local validity','', '![All cases XY and roll](xy_and_roll.png)','',f"Completed 14 cases, {result['successes']} passes. No training. Scored {len(known)} validation amplitudes: {result['correct']} correct, {result['unsafe_predictions']} false-safe, {result['conservative_predictions']} conservative rejections.",'','| case | predicted pass | actual pass | peak roll rad |','|---|---|---|---|']
 for r in validation:lines.append(f"| {r['name']} | {r['predicted_pass']} | {r['actual_pass']} | {r['peak_roll']:.6f} |")
 lines+=['','Predictions: '+json.dumps(predictions),'','Predictions copied from the prior right_transfer_v1 +/-0.5 N probes before this execution. No refitting; this panel tests intermediate amplitudes at the same DEV states. Zero-force continuation roll must reproduce the saved baseline exactly. This is a route-transfer development test, not final TEST. Roll predictor samples control ticks while actual success uses physical-substep peak; report any resulting mismatch. No upper bound inferred from all-pass results, and no automatic force expansion. Theory force map and effort budget remain uncalibrated.']
 (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
 ledger=Path('/home/qy/STTW_CONTROL/research-hub/PROJECT_STATE.md')
 with (ledger.parent/'.write.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);(ledger.parent/'AGENTS.md').read_text();txt=ledger.read_text();entry=f"\n\n### 2026-10-22 容扰度右转局部有效性自动回收\n\nsource_type=repo_artifact; {root}，14条完成，lower ticks {s['active_lower_ticks']}/53200，另8preflight，训练0。通过{result['successes']}/14；大幅验证{len(known)}条，判对{result['correct']}，假安全{result['unsafe_predictions']}，保守误拒{result['conservative_predictions']}。使用前轮冻结预测，本轮无拟合；零扰动侧倾复现通过；原始结果和XY见REPORT.md，图待视觉检查。仅新容扰度路线DEV状态，非独立最终TEST或连续安全证明。下一步回收逐项失败、核验预测是否可迁移，再决定模型标定；未训练/部署。\n";txt=txt.replace('## 2. STTW_CONTROL','## 2. STTW_CONTROL'+entry,1);txt=re.sub(r'state_revision: (\d+)',lambda m:f'state_revision: {int(m[1])+1}',txt,count=1);tmp=ledger.with_suffix('.right.tmp');tmp.write_text(txt);tmp.replace(ledger)
 status('completed',report=str(root/'REPORT.md'))
except BaseException as e:status('error',error=f'{type(e).__name__}: {e}');raise
