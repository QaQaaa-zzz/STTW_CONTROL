import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sttw_matplotlib')
import json,time,subprocess,sys,fcntl,re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]).resolve();parent=root.parent/'pulse_pilot_v1_attempt2'
def status(phase,**kw):
 p=root/'analysis_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(phase=phase,checked_unix=time.time(),**kw),indent=2));tmp.replace(p)
status('waiting_for_execution')
while True:
 s=json.loads((root/'status.json').read_text())
 if s['phase'] in ('completed','error','failed'):break
 time.sleep(15)
if s['phase']!='completed':status('blocked_by_execution');sys.exit(1)
try:
 status('running');rows=json.loads((root/'results.json').read_text())['cases'];assert len(rows)==7
 base=dict(np.load(root/'alpha0_t1_F+0.npz'));prior=dict(np.load(parent/'alpha0_t1_F+0.npz'))
 assert next(r for r in rows if r['force_N']==0)['success'],'zero force continuation fails original standard'
 def g(d):
  tail=d['time']+.005>19.+1e-5
  return np.concatenate([d['peak_roll']-.302,abs(d['actual_forward_speed'][tail]-2.6)-.1,abs(d['path_cross_track'][tail])-.1,abs(d['path_heading_error'][tail])-.05])
 slopes=[];g0=g(base)
 for a in [.25,.5]:
  d=dict(np.load(root/f'alpha0_t1_F{-a:+g}.npz'));q=(g(d)-g0)/a;mask=q>0;slopes.append({'probe_N':a,'secant_bound_N':float((-g0[mask]/q[mask]).min())})
 check=[dict(force_N=r['force_N'],predicted_success=abs(r['force_N'])<=4.721686363220215,actual_success=r['success'],peak_roll=r['peak_roll']) for r in rows if abs(r['force_N'])>=4]
 result=dict(cases=rows,validation=check,smaller_probe_predictions=slopes,baseline_replay_max_phi_difference=float(abs(base['phi']-prior['phi']).max()),scope='one development state; local test, not independent generalization')
 (root/'summary.json').write_text(json.dumps(result,indent=2))
 fig,axs=plt.subplots(1,2,figsize=(12,5))
 for r in rows:
  d=np.load(root/(r['name']+'.npz'));sel=d['time']<6
  axs[0].plot(d['actual_xy'][sel,0],d['actual_xy'][sel,1],label=f"{r['force_N']:g} N")
  axs[1].plot(d['time']+.005,np.rad2deg(d['peak_roll']),label=f"{r['force_N']:g} N")
 p=np.load(parent/'path.npz');mask=p['s']<15;axs[0].plot(p['xy'][mask,0],p['xy'][mask,1],'k--',label='original path');axs[0].set_xlim(7,16);axs[0].set_ylim(-1,10);axs[0].set_aspect('equal');axs[0].set_xlabel('X [m]');axs[0].set_ylabel('Y [m]')
 axs[1].axhline(np.rad2deg(.302),ls='--',color='k');axs[1].set_xlim(.8,3);axs[1].set_xlabel('episode time [s]');axs[1].set_ylabel('absolute roll peak [deg]')
 for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(root/'xy_and_roll.png',dpi=160)
 lines=['# Local boundary verification','', 'Original prediction: 4.721686 N. No training. Same complete state, held-out amplitudes only.', '', '| force N | predicted pass | actual pass | peak roll rad |','|---|---|---|---|']
 for r in check:lines.append(f"| {r['force_N']} | {r['predicted_success']} | {r['actual_success']} | {r['peak_roll']:.6f} |")
 lines+=['','Small probe predictions: '+json.dumps(slopes),'','![XY and roll](xy_and_roll.png)','','No continuous boundary or cross-scene guarantee. Review summary.json for baseline replay differences.']
 (root/'REPORT.md').write_text('\n'.join(lines))
 ledger=Path('/home/qy/STTW_CONTROL/research-hub/PROJECT_STATE.md')
 with open(ledger.parent/'.write.lock','a') as lk:
  fcntl.flock(lk,fcntl.LOCK_EX);txt=ledger.read_text();entry=f"### 2026-10-22 定向局部边界验证自动回收\n\n{root}，7条完成，{s['active_lower_ticks']}/26600新面板ticks，额外preflight8；无训练。4.721686N先验预测对新幅值结果：{json.dumps(check,ensure_ascii=False)}。小幅割线：{json.dumps(slopes)}。结果/XY见REPORT.md及summary.json；仅单个开发状态，未认证泛化或连续边界。图已生成待视觉审阅。\n\n";txt=txt.replace('## 2. STTW_CONTROL\n','## 2. STTW_CONTROL\n\n'+entry,1);txt=re.sub(r'state_revision: (\d+)',lambda m:'state_revision: '+str(int(m[1])+1),txt,count=1);ledger.write_text(txt)
 status('completed',report=str(root/'REPORT.md'))
except BaseException as e:status('error',error=f'{type(e).__name__}: {e}');raise
