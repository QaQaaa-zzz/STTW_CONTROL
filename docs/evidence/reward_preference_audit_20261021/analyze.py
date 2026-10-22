"""CPU-only fixed-trajectory reward audit; never imports or steps a simulator."""
from pathlib import Path
import sys,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'learning/src'))
from sttw_control.path_command_reporting import reward_series, reconstruct_costs, _interval_context
OUT=Path(__file__).parent
DATA=ROOT/'runs/reward_preference_audit_20261021'
DATA.mkdir(exist_ok=True)
OLD=Path('/home/qy/STTW_CONTROL/runs/worktrees/path-feedback-v1/runs/path_coverage_v2_resume_20261011')
NEW=ROOT/'runs/path_high_demand_shared_200_20261020/training'
om=json.loads((OLD/'manifest.json').read_text());nm=json.loads((NEW/'manifest.json').read_text())
cfg=nm['config'];oldcfg=om['config'];dt=cfg['timing']['lower_dt_s'];gamma=cfg['training_future_phase_C']['gamma']
assert om['source']['candidate']['actor_sha256']==nm['source']['candidate']['actor_sha256']
assert oldcfg['timing']==cfg['timing'] and om['terminal_contract']==nm['terminal_contract']=='terminal_contract_v2'
changed={k:[oldcfg['path_reward'].get(k),v] for k,v in cfg['path_reward'].items() if oldcfg['path_reward'].get(k)!=v}
assert set(changed)=={'roll_weight','working_roll_weight','component_caps','failure_cost_rate_bound'}
ALPHAS=[0.,.25,.5,.75,1.]
keys=['alpha_0','alpha_0p25','alpha_0p5','alpha_0p75','alpha_1']
sets=[('coverage_dual_u200',OLD,{'policy0':OLD/'alpha0/final_best','policy1':OLD/'alpha1/final_best','B0':OLD/'zero_upper'},oldcfg),
      ('high_demand_u150',NEW,{**{f'policy{a:g}':NEW/'shared/final_best'/k for a,k in zip(ALPHAS,keys)},'B0':NEW/'zero_upper'},cfg),
      ('high_demand_u200',NEW,{**{f'policy{a:g}':NEW/'shared/validation/update_0200'/k for a,k in zip(ALPHAS,keys)},'B0':NEW/'zero_upper'},cfg)]
records=[];pairs=[];audit=[];arrays={};snapshots=[];component_series={}
for label,base,dirs,originalcfg in sets:
 cases=sorted(p.stem for p in dirs['policy0'].glob('*.npz') if not p.stem.endswith('_path'))
 assert len(cases)==6
 fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
 for ax,case in zip(axes.flat,cases):
  path=dict(np.load(dirs['policy0']/(case+'_path.npz')))
  curves=np.flatnonzero(abs(path['curvature'])>1e-8)
  start=float(path['s'][curves[0]]) if len(curves) else None
  end=float(path['turn_end'])
  ax.plot(path['xy'][:,0],path['xy'][:,1],color='0.6',ls='--',label='fixed reference')
  trajectories={}
  for method,folder in dirs.items():
   file=folder/(case+'.npz');d=dict(np.load(file));n=len(d['time']);assert n>0 and np.isfinite(d['actual_xy']).all()
   if method!='B0':
    matching=dict(np.load(folder/(case+'_path.npz')))
    for k in ('s','xy','heading','curvature','goal','turn_end'):assert np.array_equal(path[k],matching[k]),(label,case,k)
   failed=bool(np.any(d['physical_failure']) or np.any(d.get('tracking_domain_failure',d.get('domain_exit',False))))
   # Complete endpoint trajectories are the matched alternatives; B0 failures remain visible.
   if method!='B0':assert n==4000 and not failed
   alpha=0. if method=='B0' else float(method.removeprefix('policy'))
   native,native_comp=reward_series(d,alpha,originalcfg)
   err=float(np.max(abs(native-np.asarray(d['scored_tick_reward']))))
   normal=native_comp['failure_tail']==0
   normal_error=float(np.max(abs(native[normal]-np.asarray(d['scored_tick_reward'])[normal])))
   assert normal_error<3e-5,(file,normal_error)
   terminal_error=None
   if not normal.all():
    ctx=_interval_context(d,originalcfg);ti=np.flatnonzero(~normal);remain=(1000-ctx['interval_start_tick'][ti]//4).astype(np.float32)
    rc=originalcfg['path_reward'];g=originalcfg['training_future_phase_C']['gamma']
    runtime=-np.float32(rc['failure_extra'])-np.float32(rc['scale']*.02*rc['failure_cost_rate_bound']/(1-g))*(np.float32(1)-np.float32(g)**remain)
    terminal_error=float(np.max(abs(runtime-np.asarray(d['scored_tick_reward'])[ti])))
    assert terminal_error<.002,(file,terminal_error)
   audit.append(dict(file=str(file),native_alpha=alpha,max_abs_tick_error=err,normal_max_error=normal_error,failure_float32_error=terminal_error))
   inter=_interval_context(d,cfg);discount=gamma**(inter['ticks']//4)
   ev=d['actual_forward_speed']-d['v_user'];ey=d['path_cross_track'];eh=d['path_heading_error'];progress=d['path_progress']
   tail=np.asarray(d['time'])+dt>19+1e-5
   hold=bool(n==4000 and not failed and tail.sum()==200 and np.all(abs(ev[tail])<=.1) and np.all(abs(ey[tail])<=.1) and np.all(abs(eh[tail])<=.05))
   turn=np.zeros(n,bool) if start is None else (progress>=start)&(progress<=end)
   windows={'whole':np.ones(n,bool),'turn':turn,'exit':np.zeros(n,bool) if start is None else progress>end}
   phy={}
   for wn,mask in windows.items():
    if not mask.any():continue
    phy[wn]=dict(seconds=float(mask.sum()*dt),speed_rmse=float(np.sqrt(np.mean(ev[mask]**2))),path_rmse=float(np.sqrt(np.mean(ey[mask]**2))),mean_speed=float(np.mean(d['actual_forward_speed'][mask])),peak_roll=float(max(d['peak_roll'][mask])),roll_over_030_s=float(np.sum(d['peak_roll'][mask]>.30)*dt),roll_over_0302_s=float(np.sum(d['peak_roll'][mask]>.302)*dt))
   recovery=None
   if start is not None:
    good=(abs(ev)<=.1)&(abs(ey)<=.1)&(abs(eh)<=.05)
    valid=np.convolve(good.astype(int),np.ones(200,dtype=int),mode='valid')==200
    hits=np.flatnonzero(valid & (progress[:len(valid)]>end))
    after=np.flatnonzero(progress>end)
    if len(hits) and len(after):recovery=float((hits[0]-after[0])*dt)
   results=[]
   for score_alpha in ALPHAS:
    r,comp=reward_series(d,score_alpha,cfg)
    item=dict(dataset=label,case=case,method=method,behavior_alpha=alpha,score_alpha=score_alpha,samples=n,failed=failed,final_hold=hold,recovery_after_turn_s=recovery,physical=phy,
      total_return=float(sum(r)),discounted_return=float(sum(r*discount)),components={k:float(sum(v)) for k,v in comp.items()},
      window_returns={wn:float(sum(r[mask])) for wn,mask in windows.items() if mask.any()})
    records.append(item);results.append(item)
    for ck,cv in comp.items():component_series[f'{label}/{case}/{method}/score{score_alpha:g}/{ck}']=cv.astype(np.float32)
    if score_alpha in (0.,1.):
     arrays[f'{label}/{case}/{method}/score{score_alpha:g}']=r
   trajectories[method]=results
   line,=ax.plot(d['actual_xy'][:,0],d['actual_xy'][:,1],label=method,lw=1.25,alpha=.85)
   if failed:ax.plot(*d['actual_xy'][-1],marker='x',c=line.get_color(),ms=8)
  for a in ALPHAS:
   x=trajectories['policy0'][ALPHAS.index(a)];y=trajectories['policy1'][ALPHAS.index(a)]
   pairs.append(dict(dataset=label,case=case,score_alpha=a,delta_return_policy1_minus_policy0=y['total_return']-x['total_return'],delta_discounted_policy1_minus_policy0=y['discounted_return']-x['discounted_return'],component_deltas={k:y['components'][k]-x['components'][k] for k in x['components']},window_deltas={k:y['window_returns'][k]-x['window_returns'][k] for k in x['window_returns']}))
  ax.set_title(case);ax.set_xlabel('world X (m)');ax.set_ylabel('world Y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.suptitle(label+' | local300 | saved trajectories; x = actual failure endpoint')
 fig.savefig(OUT/(label+'_XY.png'),dpi=140);plt.close(fig)
 # Full per-step and cumulative rewards, both scoring endpoints, all physical policies + baseline.
 for kind in ('step','cumulative'):
  fig,axes=plt.subplots(2,3,figsize=(15,9),layout='constrained')
  for ax,case in zip(axes.flat,cases):
   for method in dirs:
    for a in (0.,1.):
     r=arrays[f'{label}/{case}/{method}/score{a:g}'];y=np.cumsum(r) if kind=='cumulative' else r
     ax.plot(np.arange(len(r)),y,ls='-' if a==0 else '--',lw=.9,label=f'{method} scored a={a:g}',alpha=.8)
   ax.set_title(case);ax.set_xlabel('actual lower tick (5 ms)');ax.set_ylabel(kind+' reward');ax.grid(alpha=.2)
   if kind=='step':ax.set_yscale('symlog',linthresh=.01)
  axes.flat[0].legend(fontsize=6,ncol=2)
  fig.suptitle(label+' | CURRENT reward; solid score alpha0, dashed score alpha1')
  fig.savefig(OUT/(label+'_'+kind+'_rewards.png'),dpi=140);plt.close(fig)
np.savez_compressed(DATA/'reward_components.npz',**component_series)
summary=dict(scoring_config=cfg['path_reward'],reward_config_changes=changed,discount=dict(gamma=gamma,interval_s=.02,from_reset=True),source_manifests=[str(OLD/'manifest.json'),str(NEW/'manifest.json')],native_audit_max_abs_tick_error=max(x['max_abs_tick_error'] for x in audit),audits=audit,records=records,pairs=pairs,physics_executed=0,training_updates=0)
(OUT/'results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
with (OUT/'pair_scores.csv').open('w') as f:
 fields=['dataset','case','score_alpha','delta_return_policy1_minus_policy0','delta_discounted_policy1_minus_policy0'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(pairs)
for group in [x[0] for x in sets]:
 print('\n',group)
 for case in sorted({p['case'] for p in pairs if p['dataset']==group}):
  pp=[p for p in pairs if p['dataset']==group and p['case']==case and p['score_alpha'] in (0,1)]
  print(case,[(p['score_alpha'],round(p['delta_return_policy1_minus_policy0'],3),round(p['delta_discounted_policy1_minus_policy0'],3)) for p in pp])
print('records',len(records),'audit traces',len(audit),'error',summary['native_audit_max_abs_tick_error'])
