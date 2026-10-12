"""Read-only paired200 analysis, original NPZ only; no simulator or learner update."""
import json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'learning/src'))
from sttw_control.path_command_reporting import reward_series,reconstruct_costs
from sttw_control.path_command_selection import cases,summarize_task,qualify_preference
from sttw_control.preference_best import digest
OUT=Path(__file__).resolve().parent
OLD=ROOT.parent/'path-safety-preference'
roots={a:ROOT/f'runs/path_safety_{a}_200_20261011' for a in ('control','candidate')}
configs={a:json.loads((r/'manifest.json').read_text())['config'] for a,r in roots.items()}
case_names=[n for n,_,_ in cases(configs['candidate'])]
result={'schema':'paired200_offline_audit_v1','roots':{k:str(v) for k,v in roots.items()},'recovery':'NOT_RUN; no complete natural overlimit snapshots/evaluation in these runs','checkpoints':{},'selected':{},'verification':{},'budget':{},'training':{},'preference':{},'last200':{}}
records={};traces={};last={};paths={}
def same(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b
for alpha in (0,1):
 initial=[torch.load(OLD/f'runs/path_safety_{a}_50_20261011/alpha{alpha}/checkpoints/update_0000.pt',map_location='cpu',weights_only=False) for a in ('control','candidate')]
 keys=['policy','optimizer','torch_rng','numpy_rng','python_rng','cuda_rng','consecutive_rejections']
 assert all(same(initial[0][k],initial[1][k]) for k in keys)
 result['verification'][f'paired_initial_alpha{alpha}']='all policy tensors (Actor/Critic/log_std), Adam and RNG identical'
 for arm,r in roots.items():
  before=torch.load(OLD/f'runs/path_safety_{arm}_50_20261011/alpha{alpha}/checkpoints/update_0050.pt',map_location='cpu',weights_only=False)
  after=torch.load(r/f'alpha{alpha}/checkpoints/update_0050.pt',map_location='cpu',weights_only=False)
  assert all(same(before[k],after[k]) for k in keys+['identity','update'])
  result['verification'][f'resume_{arm}_alpha{alpha}']='learner identity, policy, Adam/LR, RNG and rejection count identical; physical prepared-bank reset'
max_reward_error=0.;max_cost_error=0.;audited=0;float32_boundary_checks=[]
for arm,r in roots.items():
 result['checkpoints'][arm]={};result['selected'][arm]={};result['last200'][arm]={};result['training'][arm]={}
 for alpha in (0,1):
  label=f'{arm}{alpha}';end=r/f'alpha{alpha}';cfg=configs[arm]
  result['checkpoints'][arm][str(alpha)]={}
  for folder in sorted((end/'validation').glob('update_*')):
   if not (folder/'task_metrics.json').exists():continue
   data={n:dict(np.load(folder/f'{n}.npz')) for n in case_names}
   summary=summarize_task(data,alpha,cfg);logged=json.loads((folder/'task_metrics.json').read_text())
   for k in ['qualified_task','physical_failures','tracking_domain_failures','working_failures','corridor_failures','goal_hold_failures','primary_band_failures']:
    assert summary[k]==logged[k],(arm,alpha,folder,k)
   case_detail={}
   for n,d in data.items():
    reward,components=reward_series(d,alpha,cfg)
    np.testing.assert_allclose(reward,d['scored_tick_reward'],rtol=2e-4,atol=2e-6)
    max_reward_error=max(max_reward_error,float(np.max(np.abs(reward-d['scored_tick_reward']))))
    rebuilt=reconstruct_costs(d,alpha,cfg)
    for k,value in rebuilt['effective'].items():
     logged_cost=d['effective_cost_'+k]
     bad=np.abs(value-logged_cost)>2e-5+2e-4*np.abs(logged_cost)
     if bad.any():
      # Independent float64 reveals the narrow float32 .30 threshold rounding.
      assert k=='working_roll' and cfg['path_reward'].get('reward_version')=='path_safety_preference_v1'
      c=cfg['path_reward'];p32=np.asarray(d['peak_roll'],np.float32);x=np.maximum(np.float32(0),p32-np.float32(c['working_roll_start_rad']))/np.float32(c['working_roll_scale_rad'])
      explicit32=np.float32(c['working_roll_weight'])*np.where(np.abs(x)<=1,x*x,2*np.abs(x)-1)+np.float32(c['working_exposure_weight'])*np.clip((p32-np.float32(c['working_roll_start_rad']))/np.float32(c['working_exposure_width_rad']),0,1)
      explicit32=np.minimum(explicit32,np.float32(c['component_caps'][k]))
      np.testing.assert_allclose(explicit32,logged_cost,rtol=2e-4,atol=2e-5)
      float32_boundary_checks.append(dict(arm=arm,alpha=alpha,update=folder.name,case=n,component=k,ticks=np.flatnonzero(bad).tolist(),float64_cost=value[bad].tolist(),logged_cost=logged_cost[bad].tolist(),explicit_float32_cost=explicit32[bad].tolist()))
     else:np.testing.assert_allclose(value,logged_cost,rtol=2e-4,atol=2e-5)
     max_cost_error=max(max_cost_error,float(np.max(np.abs(value-d['effective_cost_'+k]))))
    t=d['time']+.005;ey=np.abs(d['path_cross_track']);tail=t>19.;ev=d['actual_forward_speed']-d['v_user']
    corridor=rebuilt['diagnostic_subcosts']['corridor'];pcap=cfg['path_reward']['component_caps']['path']
    incremental=rebuilt['effective']['path']-np.minimum(rebuilt['raw']['path']-corridor,pcap)
    detail=dict(summary['cases'][n]);detail.update(max_path_m=float(ey.max()),path_exceed_seconds=float(np.sum(ey>.4)*.005),max_roll_rad=float(d['peak_roll'].max()),path_cap_seconds=float(np.sum(rebuilt['capped']['path'])*.005),last1s_speed_rmse=float(np.sqrt(np.mean(ev[tail]**2))),last1s_max_speed_error=float(np.abs(ev[tail]).max()),last1s_max_path_error=float(ey[tail].max()),last1s_max_heading_error=float(np.abs(d['path_heading_error'][tail]).max()),total_reward=float(reward.sum()),component_returns={k:float(v.sum()) for k,v in components.items()},new_corridor_effective_reward=float(-.0005*incremental.sum()),raw_corridor_max=float(corridor.max()),path_cap=pcap,command_chain=dict(request_speed_mean=float(d['v_user'][tail].mean()),corrected_speed_mean=float(d['governed'][tail,0].mean()),nominal_steer_mean=float(d['nominal'][tail,1].mean()),corrected_steer_mean=float(d['governed'][tail,1].mean()),final_front_rate_mean=float(d['final_command'][tail,0].mean()),final_rear_axis_rate_mean=float(d['final_command'][tail,1].mean()),measured_steer_mean=float(d['actual_delta'][tail].mean()),measured_speed_mean=float(d['actual_forward_speed'][tail].mean()),speed_offset_mean=float(d['offsets'][tail,0].mean()),steer_offset_mean=float(d['offsets'][tail,1].mean())))
    case_detail[n]=detail;audited+=1
   result['checkpoints'][arm][str(alpha)][folder.name]=dict(summary=summary,cases=case_detail)
   if folder.name=='update_0200':last[label]=data;result['last200'][arm][str(alpha)]=dict(summary=summary,cases=case_detail)
  record=json.loads((end/'best_safety_task_model.json').read_text());final=json.loads((end/'final_best/source.json').read_text());assert record==final
  for key in ['actor','checkpoint']:assert digest(record[key])==record[key+'_sha256']
  records[label]=record;traces[label]={n:dict(np.load(end/'final_best'/f'{n}.npz')) for n in case_names}
  result['selected'][arm][str(alpha)]=dict(source_update=record['source_update'],checkpoint_sha256=record['checkpoint_sha256'],qualified_task=record['qualified_task'],score_tuple=record['score_tuple'],cases=result['checkpoints'][arm][str(alpha)][f"update_{record['source_update']:04d}"]['cases'])
  rows=[]
  for stage_root in [OLD/f'runs/path_safety_{arm}_50_20261011',r]:rows += [json.loads(s) for s in (stage_root/f'alpha{alpha}/metrics.jsonl').read_text().splitlines()]
  assert [x['update'] for x in rows]==list(range(1,201))
  result['training'][arm][str(alpha)]=dict(updates=len(rows),physical_failures=sum(x['physical_failures'] for x in rows),domain_failures=sum(x['tracking_domain_failures'] for x in rows),engineering_faults=sum(x['engineering_faults'] for x in rows),rejected_updates=sum(x['optimizer']['accepted_minibatches']==0 for x in rows),sampling_seconds=sum(x['sample_seconds'] for x in rows),optimization_seconds=sum(x['optimize_seconds'] for x in rows),train_path_capped_lower_ticks=sum(x['component_cap_counts']['path'] for x in rows),last_rollout_reward=rows[-1]['mean_step_reward'],last_learning_rates=rows[-1]['learning_rates'])
  print('audited',label,'best',record['source_update'],'last gates',result['last200'][arm][str(alpha)]['summary']['score_tuple'],flush=True)
for n in case_names:paths[n]=dict(np.load(roots['control']/'zero_upper'/f'{n}_path.npz'))
b0={n:dict(np.load(roots['control']/'zero_upper'/f'{n}.npz')) for n in case_names}
for arm in ('control','candidate'):
 result['preference'][arm]=dict(selected=qualify_preference(traces[arm+'0'],traces[arm+'1'],paths,configs[arm]),last200=qualify_preference(last[arm+'0'],last[arm+'1'],paths,configs[arm]),descriptive_bins={})
 for view,source in [('selected',traces),('last200',last)]:
  result['preference'][arm]['descriptive_bins'][view]={}
  for n in ('left90_R2_2.6','right90_R2_2.6'):
   p=paths[n];end=float(p['turn_end']);edges=np.append(np.arange(3.,end,.5),end);errs=[];counts=[]
   for a in (0,1):
    d=source[arm+str(a)][n];bins=[];cnt=[]
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
     mask=(d['path_progress']>=lo)&((d['path_progress']<=hi) if i==len(edges)-2 else (d['path_progress']<hi));cnt.append(int(mask.sum()))
     if mask.any():bins.append([float(np.mean((d['actual_forward_speed'][mask]-d['v_user'][mask])**2)),float(np.mean(d['path_cross_track'][mask]**2))])
    counts.append(cnt);errs.append(np.sqrt(np.mean(bins,axis=0)).tolist() if all(cnt) else None)
   result['preference'][arm]['descriptive_bins'][view][n]=dict(errors=errs,counts=counts,Dv=None if any(x is None for x in errs) else errs[0][0]-errs[1][0],Dy=None if any(x is None for x in errs) else errs[1][1]-errs[0][1],qualification='common task failed; descriptive only')
result['verification'].update(float32_boundary_checks=float32_boundary_checks,audited_traces=audited,max_abs_tick_reward_error=max_reward_error,max_abs_component_cost_error=max_cost_error)
statuses=[]
for arm in ('control','candidate'):
 for root in [OLD/f'runs/path_safety_{arm}_50_20261011',roots[arm]]:
  s=json.loads((root/'status.json').read_text());assert s['state']=='completed' and s['protected_unchanged'];statuses.append(s)
result['budget']=dict(policy_transitions=sum(x['policy_transitions_total'] for x in statuses),training_lower_ticks=sum(x['training_control_ticks_total'] for x in statuses),dev_lower_ticks=sum(x['validation_control_ticks_total'] for x in statuses),engineering_additional=0,recovery_executed=0)
assert result['budget']['policy_transitions']==52428800
(OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
colors={'B0':'0.45','control0':'#1f77b4','control1':'#ff7f0e','candidate0':'#2ca02c','candidate1':'#d62728'}
for view,source in [('selected',traces),('last200',last)]:
 fig,axes=plt.subplots(2,3,figsize=(16,10))
 for n,ax in zip(case_names,axes.flat):
  datasets={'B0':b0[n],**{label:d[n] for label,d in source.items()}};p=paths[n];maximum=max(float(d['path_progress'].max()) for d in datasets.values());mask=p['s']<=maximum+1
  ax.plot(*p['xy'][mask].T,'k--',label='original fixed path')
  for label,d in datasets.items():
   step='B0 local300' if label=='B0' else f"{label} u{records[label]['source_update'] if view=='selected' else 200}"
   ax.plot(*d['actual_xy'].T,color=colors[label],label=step,linewidth=1);ax.plot(*d['actual_xy'][-1],marker='o',color=colors[label],ms=3)
  ax.set(title=n,xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=6)
 fig.suptitle(f'{view} | four paired upper groups + B0 local300 | train seeds911101+alpha | original20s DEV, no external disturbance\nMarkers=actual ends; selected best does not mean qualified');fig.tight_layout();fig.savefig(OUT/f'ALL_XY_{view}.png',dpi=140);plt.close(fig)
 for n in case_names:
  datasets={'B0':b0[n],**{label:d[n] for label,d in source.items()}};p=paths[n]
  fig,axes=plt.subplots(2,3,figsize=(16,9));ax=axes[0,0]
  mask=p['s']<=float(p['turn_end'])+2 if p['turn_end']>0 else p['s']<15
  ax.plot(*p['xy'][mask].T,'k--',label='original path')
  for label,d in datasets.items():
   t=d['time']+.005;xy_mask=d['path_progress']<=float(p['turn_end'])+2 if p['turn_end']>0 else d['path_progress']<15
   ax.plot(*d['actual_xy'][xy_mask].T,color=colors[label],label=label,lw=1.2)
   axes[0,1].plot(t,d['peak_roll'],color=colors[label],label=label)
   axes[0,2].plot(t,d['actual_forward_speed']-d['v_user'],color=colors[label],label=label)
   axes[1,0].plot(t,d['path_cross_track'],color=colors[label],label=label)
   if label!='B0':
    arm=label[:-1];a=int(label[-1]);reward,parts=reward_series(d,a,configs[arm]);steps=np.arange(1,len(reward)+1)
    axes[1,1].plot(steps,reward,color=colors[label],label=label+' own reward');axes[1,2].plot(steps,np.cumsum(reward),color=colors[label],label=label+' own reward')
    br,_=reward_series(b0[n],a,configs[arm]);axes[1,1].plot(np.arange(1,len(br)+1),br,color=colors[label],ls='--',lw=.7,label=label+' scored B0');axes[1,2].plot(np.arange(1,len(br)+1),np.cumsum(br),color=colors[label],ls='--',lw=.7)
  ax.set_aspect('equal',adjustable='box');ax.set(title='Turn detail XY (full20s in ALL_XY)',xlabel='World X (m)',ylabel='World Y (m)')
  axes[0,1].axhline(.302,color='k',ls='--');axes[0,1].set(title='5ms physical-substep roll peak',xlabel='Time(s)',ylabel='rad')
  axes[0,2].axhline(0,color='k',lw=.5);axes[0,2].set(title='Original requested speed error',xlabel='Time(s)',ylabel='m/s')
  axes[1,0].axhline(.4,color='k',ls='--');axes[1,0].axhline(-.4,color='k',ls='--');axes[1,0].set(title='Original path signed error',xlabel='Time(s)',ylabel='m')
  axes[1,1].set(title='Actual per-lower-step reward',xlabel='Actual steps',ylabel='reward');axes[1,2].set(title='Cumulative reward (different objectives)',xlabel='Actual steps',ylabel='reward')
  for ax in axes.flat:ax.legend(fontsize=5);ax.grid(alpha=.2)
  updates=', '.join(f'{k}=u{records[k]["source_update"] if view=="selected" else 200}' for k in source)
  fig.suptitle(f'{n} | {view} | {updates}\nOriginal20s DEV, no external disturbance; B0 frozenlocal300; reward comparisons only within same alpha/version');fig.tight_layout();fig.savefig(OUT/f'{n}_{view}.png',dpi=125);plt.close(fig)
print('verification',result['verification']);print('budget',result['budget'])
