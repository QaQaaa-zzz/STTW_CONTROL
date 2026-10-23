import json,pickle,math
from pathlib import Path
from collections import Counter
import numpy as np
from sttw_control.tolerance_state_features import snapshot_features
from sttw_control.preference_best import digest
run=Path('runs/short_recovery_post_v1_recovery1/evaluation');campaign=Path('runs/short_recovery_train_v1_recovery1')
out=run/'diagnostics';raw=json.loads((run/'results.json').read_text())['cases'];manifest=json.loads((run/'manifest.json').read_text())
plan=json.loads((campaign/'uniform_alpha0/plan.json').read_text());source=json.loads(Path(plan['parent_manifest']).read_text())['source'];cc=source['controller']
from sttw_control.tolerance_training import SPEC
rows=[];snapshots={}
for r in raw:
 name=f"{r['policy']}_{r['route']}_v{r['speed']:g}_a{r['alpha']}_t{r['onset_s']:g}_snapshot.pkl"
 if name not in snapshots:
  with (run/name).open('rb') as f:s=pickle.load(f)
  x=snapshot_features(s,speed_request=r['speed'],onset_s=r['onset_s'],controller=cc)
  lam=x['lambda_u'];b=x['b_u'];K=SPEC['gamma']*(-math.expm1(-lam*.1))/lam;W=b*b*(-math.expm1(-2*lam*min(1.,x['remaining_s'])))/(2*lam);radius=math.sqrt(SPEC['effort_budget']*W)
  snapshots[name]=dict(z=x['z_u'],K=K,radius=radius,plus=max(0,(radius-x['z_u'])/abs(K)) if abs(x['z_u'])<=radius else 0.,minus=max(0,(radius+x['z_u'])/abs(K)) if abs(x['z_u'])<=radius else 0.,feature=x)
 m=snapshots[name];risk=abs(m['z']+m['K']*r['force_N'])/m['radius']
 with np.load(run/(r['name']+'.npz')) as d:
  tail=d['time']>=19-1e-5
  ey=np.abs(d['path_cross_track']);eh=np.abs(d['path_heading_error']);ev=np.abs(d['actual_forward_speed']-r['speed'])
  post=d['time']>=r['onset_s']-1e-5
  outside=(ey>.1)|(eh>.05)|(ev>.1)
  # Earliest suffix start after which all sampled task errors remain in band.
  last=np.flatnonzero(post&outside)
  recovery=(float(d['time'][last[-1]]+.005-r['onset_s']) if len(last) else 0.) if len(d['time'])==4000 and not outside[-1] else None
  metrics=dict(max_abs_path_m=float(ey.max()),max_abs_heading_rad=float(eh.max()),tail_max_path_m=float(ey[tail].max()) if tail.any() else None,recovery_time_s=recovery,actual_duration_s=float(len(ey)*.005),max_reference_steer_offset_rad=float(np.max(np.abs(d['governed'][:,1]-d['nominal'][:,1]))))
 rows.append(dict(**r,**metrics,predicted_recoverable=risk<=1.,normalized_energy_sqrt=risk,predicted_plus_N=m['plus'],predicted_minus_N=m['minus']))
def auc(values,labels):
 fail=np.asarray(values)[~np.asarray(labels)];ok=np.asarray(values)[np.asarray(labels)]
 return float(np.mean((fail[:,None]>ok[None,:])+.5*(fail[:,None]==ok[None,:]))) if len(fail) and len(ok) else None
models=[]
for alpha in [0,1]:
 for policy in manifest['protocol']['policies']:
  rr=[r for r in rows if r['alpha']==alpha and r['policy']==policy and r['force_N']]
  models.append(dict(alpha=alpha,policy=policy,n=len(rr),false_safe=sum(r['predicted_recoverable'] and not r['success'] for r in rr),conservative=sum(not r['predicted_recoverable'] and r['success'] for r in rr),predicted_safe=sum(r['predicted_recoverable'] for r in rr),failures=sum(not r['success'] for r in rr),task_failure_rank_auc=auc([r['normalized_energy_sqrt'] for r in rr],[r['success'] for r in rr]),working_failure_rank_auc=auc([r['normalized_energy_sqrt'] for r in rr],[not r['working_failure'] for r in rr])))
exposure=[]
for alpha in [0,1]:
 for arm in ['uniform','guided']:
  job=campaign/f'{arm}_alpha{alpha}';logs=[json.loads(x) for x in (job/'metrics.jsonl').read_text().splitlines()]
  if (job/'recovery_receipt.json').exists():
   receipt=json.loads((job/'recovery_receipt.json').read_text())['source'];inherited=[json.loads(x) for x in Path(receipt['cursor_source_metrics']).read_text().splitlines()]
   logs=[r for r in inherited if r['update']<=receipt['update']]+logs
  assert [r['update'] for r in logs]==list(range(1,28))
  hist=np.sum([r['pulse']['force_histogram'] for r in logs],axis=0);joint=np.sum([r['route_force_active_lower_ticks'] for r in logs],axis=0)
  exposure.append(dict(alpha=alpha,arm=arm,retained_updates=27,force_histogram=hist.tolist(),pulse_events=int(hist.sum()),mean_abs_force_N=float(hist@np.array([8,6,4,2,2,4,6,8])/hist.sum()),joint_lower_ticks=joint.tolist(),nominal_lower_fraction=float(joint[:,0].sum()/joint.sum())))
for r in rows:r['task_recovered']=bool(r['final_hold'] and r['goal'] and not r['physical_failure'] and not r['domain_failure'])
for model in models:
 rr=[r for r in rows if r['alpha']==model['alpha'] and r['policy']==model['policy'] and r['force_N']]
 model['performance_failure_rank_auc']=model.pop('task_failure_rank_auc')
 model['performance_false_positive']=model.pop('false_safe');model['performance_false_negative']=model.pop('conservative')
 model['task_recovery_rank_auc']=auc([r['normalized_energy_sqrt'] for r in rr],[r['task_recovered'] for r in rr])
 model['task_recovery_failures']=sum(not r['task_recovered'] for r in rr)
 model['task_recovery_false_positive']=sum(r['predicted_recoverable'] and not r['task_recovered'] for r in rr)
 model['task_recovery_false_negative']=sum(not r['predicted_recoverable'] and r['task_recovered'] for r in rr)
result=dict(rows=rows,proxy_diagnostics=models,training_exposure=exposure,scope='Frozen SPEC; no refit. Original success includes working roll performance; task_recovered excludes working roll bound. Neither is a physical guarantee. AUC descriptive correlated DEV; no calibration adoption. Recovery time uses future sampled errors, descriptive not online detection.')
text=json.dumps(result,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))
(out/'result_and_guidance_audit.json').write_text(text+'\n')
print(json.dumps(json.loads(text)['proxy_diagnostics'],indent=2))
