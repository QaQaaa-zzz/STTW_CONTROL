#!/usr/bin/env python3
"""Plot saved fixed best/R196 traces; no simulation or policy selection."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sttw_control.lower_command import reward_terms
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
cfg=json.loads((root/'config.json').read_text());best=json.loads((root/'training/best_model.json').read_text());dest=root/'analysis';dest.mkdir(parents=True,exist_ok=True)
lines=['# Local no-alpha lower — fixed best review','',f"Checkpoint update{best['source_update']}; SHA {best['checkpoint_sha256']}. Local qualification={best['qualified']}; parent-state gate remains separate.",'','XY first, then exact issued/actual tracking, step reward and cumulative reward. R196 receives the identical published reference.','']
rows=[]
for case in cfg['validation']['case_ids']:
 ds={'local best':dict(np.load(root/'training/final_best_validation'/f'{case}.npz')),'R196':dict(np.load(root/'training/R196_reference'/f'{case}.npz'))}
 fig,axs=plt.subplots(3,3,figsize=(16,12));ax=axs.flat;colors={'local best':'#1f77b4','R196':'#e09020'}
 for name,d in ds.items():
  t=d['time'];cmd=d['limited_command'];v=d['actual_forward_speed'];delta=d['actual_delta'];xy=d['actual_xy'];co=colors[name]
  ax[0].plot(xy[:,0],xy[:,1],label=name,color=co);ax[0].scatter(xy[-1,0],xy[-1,1],s=25,color=co,marker='x' if d['physical_failure'][-1] else 'o')
  for i,y in [(1,v),(2,delta),(3,v-cmd[:,0]),(4,delta-cmd[:,1]),(5,d['phi']),(6,d['actual_delta_rate'])]:ax[i].plot(t,y,label=name,color=co)
  if name=='local best':
   ax[1].plot(t,cmd[:,0],'k--',label='issued reference');ax[2].plot(t,cmd[:,1],'k--',label='issued reference')
   if 'reference_xy' in d:ax[0].plot(d['reference_xy'][:,0],d['reference_xy'][:,1],'k--',label='original integrated reference')
  # Reconstruct the same local reward for baseline; never compare upper reward.
  action=d['action'] if 'action' in d else d['lower_action'];prev=np.vstack([np.zeros(2),action[:-1]])
  import jax,jax.numpy as jp
  fn=jax.jit(jax.vmap(lambda ev,ed,roll,u,previous,failed,rr,remaining,peak:reward_terms(ev,ed,roll,u,previous,failed,cfg,roll_rate=rr,remaining_ticks=remaining,peak_roll=peak)))
  parts={k:np.asarray(x) for k,x in fn(jp.asarray(v-cmd[:,0]),jp.asarray(delta-cmd[:,1]),jp.asarray(d['phi']),jp.asarray(action),jp.asarray(prev),jp.asarray(d['physical_failure']),jp.asarray(d['phi_dot']),jp.asarray(2000-np.arange(len(t))),jp.asarray(d['peak_roll'])).items()}
  if name=='local best':np.testing.assert_allclose(parts['reward'],d['reward'],atol=2e-6,rtol=2e-5)
  ax[7].plot(np.arange(len(t)),parts['reward'],label=name,color=co);ax[8].plot(np.arange(len(t)),np.cumsum(parts['reward']),label=name,color=co)
  np.savez_compressed(dest/(case+'_'+name.replace(' ','_')+'_scored.npz'),**parts,time=t,error_speed=v-cmd[:,0],error_steer=delta-cmd[:,1])
 for i,title in enumerate(['XY [m]','forward speed [m/s]','steer [rad]','speed error [m/s]','steer error [rad]','roll [rad]','actual steering rate [rad/s]','step reward','cumulative reward']):
  ax[i].set_title(title);ax[i].grid(alpha=.2);ax[i].legend(fontsize=8)
  if i:ax[i].set_xlabel('control steps' if i>=7 else 'time [s]')
  if 0<i<7:
   for time in (.5,2.,4.5,6.,7.5):ax[i].axvline(time,color='.6',lw=.5)
 ax[0].set_aspect('equal',adjustable='datalim');ax[0].set_xlabel('world X [m]');ax[0].set_ylabel('world Y [m]')
 for bound in (-cfg['validation']['speed_tolerance'],cfg['validation']['speed_tolerance']):ax[3].axhline(bound,color='.5',ls='--',lw=.6)
 tol=cfg['validation']['small_steer_tolerance'] if case.startswith('post_return_small') else cfg['validation']['steer_tolerance']
 for bound in (-tol,tol):ax[4].axhline(bound,color='.5',ls='--',lw=.6)
 fig.suptitle(f"{case} | no-alpha best update{best['source_update']} / fixed R196 | bank3 | 200Hz | command changes marked; true endpoint")
 fig.tight_layout();fig.savefig(dest/(case+'.png'),dpi=150);plt.close(fig)
 lines+=['## '+case,'',f'![{case}]({case}.png)','']
 m=best['metrics']['cases'][case];rows.append((case,m['physical_failure'],m['working_limit_failure'],m['primary_failure'],m['joint_final_hold_failure'],m['peak_roll'],m['windows']['steady']['rmse'],m['windows']['post_return_small']['rmse']))
lines+=['|case|physical fail|working fail|tracking fail|final hold fail|peak roll|steady RMSE v/delta|small RMSE v/delta|','|---|---|---|---|---|---|---|---|']
lines += ['|'+'|'.join(map(str,row))+'|' for row in rows]
(dest/'INDEX.md').write_text('\n'.join(lines)+'\n')
print(dest/'INDEX.md')
