#!/usr/bin/env python3
"""Validate and summarize one completed diagnostic replay, without physics."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
s=json.loads((r/'status.json').read_text());assert s['state']=='completed'
idn=json.loads((r/'identity.json').read_text());d=dict(np.load(r/'trace.npz'));t=d['time'];n=len(t);pre='lower_diagnostic_';diag=lambda k:d[pre+k]
checks={}
checks['pre_post_clock_max_error']=float(np.max(abs(diag('post_time')-diag('pre_time')-.005)))
checks['u_nom_goal_max_error']=float(np.max(abs(d['u_nom']-d['u_goal'])))
checks['ecbc_logged_output_max_error']=float(np.max(abs(diag('ecbc_output')-d['u_nom'][:,0])))
checks['composition_unclipped_max_error']=float(np.max(abs(d['final_command'][~d['final_command_clipped']]-(d['u_nom']+d['applied_residual'])[~d['final_command_clipped']])))
mean=np.asarray(idn['normalizer']['mean'],np.float32);std=np.asarray(idn['normalizer']['std'],np.float32)
checks['normalization_max_error']=float(np.max(abs((diag('lower_input_raw')-mean)/std-diag('lower_input_normalized'))))
checks['history_first_ten_valid_counts']=diag('lower_mask')[:10].sum(1).tolist()
checks['fixed_lower_alpha_all_frames']=bool(np.all(diag('lower_frame')[:,18]==1.))
checks['eso_commit_continuity_max_error']=float(np.max(abs(diag('eso_pre')[1:]-diag('eso_post')[:-1])))
return_fields=['pending','elapsed','hold','credited','ever_left','deadline_missed','previous_action']
checks['return_commit_continuity_all_fields']=all(np.array_equal(diag('return_pre_'+k)[1:],diag('return_post_'+k)[:-1]) for k in return_fields)
armed=diag('return_pre_pending')&diag('return_post_pending')&(t>=idn['source_task']['tracking']['start_seconds'])
checks['return_one_tick_elapsed_max_error']=float(np.max(abs(diag('return_post_elapsed')[armed]-diag('return_pre_elapsed')[armed]-.005)))
assert checks['return_one_tick_elapsed_max_error']<2e-6
checks['path_frame_features_max_error']=float(np.max(abs(diag('lower_frame')[:,15:18]-d['lower_path_features'])))
checks['finite_required_arrays']=all(np.isfinite(d[k]).all() for k in ['actual_delta','actual_delta_rate','actuator_force_min','actuator_force_max',pre+'lower_input_raw',pre+'lower_input_normalized',pre+'eso_pre'])
checks['physical_failure']=bool(np.any(d['physical_failure']));checks['lower_fault']=bool(np.any(d['lower_fault']))
assert checks['finite_required_arrays'] and checks['u_nom_goal_max_error']==0 and checks['ecbc_logged_output_max_error']==0
assert checks['return_commit_continuity_all_fields'] and checks['eso_commit_continuity_max_error']==0
assert checks['history_first_ten_valid_counts']==list(range(1,11))
force={};windows=[('full',0.,12.),('B0_first',5.445,6.205),('B0_second',6.315,8.745)]
for name,start,end in windows:
 mask=(t>=start-1e-5)&(t<end-1e-5);vals=[]
 for i,actuator in enumerate(idn['actuator_names']):
  vals.append(dict(index=i,name=actuator,limited=idn['actuator_forcelimited'][i],forcerange=idn['actuator_forcerange'][i],
    minimum=float(d['actuator_force_min'][mask,i].min()),maximum=float(d['actuator_force_max'][mask,i].max()),
    at_limit_substeps=int(d['actuator_force_limit_substeps'][mask,i].sum()),observed_substeps=int(mask.sum()*25)))
 force[name]=vals
progress=diag('projection_progress');index=diag('projection_segment_index');path=d['lower_path_features']
jumps=np.flatnonzero((abs(np.diff(progress))>.25)|(abs(np.diff(path[:,1]))>.3))+1
events=[dict(time=float(t[i]),index_before=int(index[i-1]),index_after=int(index[i]),progress_before=float(progress[i-1]),progress_after=float(progress[i]),heading_before=float(path[i-1,1]),heading_after=float(path[i,1])) for i in jumps]
summary=dict(status=s,identity=dict(method=idn['method'],dtype=idn['dtype'],prepared_sha256=idn['prepared_sha256'],lower=idn['lower']),checks=checks,
    original_overlap=json.loads((r/'original_overlap.json').read_text()),force=force,native_projection_discontinuities=events,
    scope='One B0 replay only; force evidence pertains to this replay. Alpha0 force was not measured; no second physics replay.',
    interpretation='Observed path/return and network activations do not isolate causal neural field contributions.')
(out/'replay_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
fig,axs=plt.subplots(4,1,figsize=(13,12),sharex=True)
for i,name in enumerate(idn['actuator_names']):
 axs[0].plot(t,d['actuator_force_max'][:,i],label=f'{i} {name} max');axs[0].plot(t,d['actuator_force_min'][:,i],':',linewidth=.6)
axs[0].set_ylabel('Actuator force [N m]\nsubstep extrema');axs[0].legend(fontsize=7,ncol=2)
axs[1].plot(t,index,label='chosen segment (ray uses endpoint index)');axs[1].plot(t,diag('projection_path_count'),label='published points',linestyle='--');axs[1].set_ylabel('Path index');axs[1].legend(fontsize=8)
axs[2].plot(t,d['lower_path_features'][:,1],label='path heading error [rad]');axs[2].plot(t,diag('return_pre_elapsed'),label='return elapsed [s]');axs[2].plot(t,diag('return_pre_deadline_missed'),label='deadline missed');axs[2].set_ylabel('Path / return input');axs[2].legend(fontsize=8)
axs[3].plot(t,d['actual_delta'],label='actual steer');axs[3].plot(t,d['governed'][:,1],label='governed steer');axs[3].plot(t,d['phi'],label='actual roll');axs[3].set_ylabel('Angle [rad]');axs[3].set_xlabel('Interval start [s]; post state at t+0.005s');axs[3].legend(fontsize=8)
for ax in axs:
 for _,start,end in windows[1:]:ax.axvspan(start,end,color='gold',alpha=.12)
 ax.grid(alpha=.2)
fig.suptitle('Single original B0 fast_turn12s replay / seed77001 / frozen R196\nInternal and actuator measurements; comparison with historical run recorded separately',fontsize=11);fig.tight_layout();fig.savefig(out/'B0_replay_internals.png',dpi=150);plt.close(fig)
print(json.dumps(summary,indent=2))
