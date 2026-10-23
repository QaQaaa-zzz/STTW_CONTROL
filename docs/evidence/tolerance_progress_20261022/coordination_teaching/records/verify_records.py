"""Offline teaching alignment, target semantics and pilot comparison."""
from pathlib import Path
import json,pickle,hashlib
import numpy as np
R=Path(__file__).resolve().parent;run=R/'evaluation'
if json.loads((run/'status.json').read_text())['phase']!='completed':raise RuntimeError('collection not complete')
d=json.loads((R/'declaration.json').read_text());rows=json.loads((run/'results.json').read_text())['cases'];old=json.loads((R.parent/'preference_train_coordination_probe_v1/evaluation/results.json').read_text())['cases'];old={r['name']:r for r in old}
checks=[];sample_obs=[];sample_actions=[]
for row in rows:
 p=run/(row['name']+'_teaching.npz')
 with np.load(p) as z:
  t=z['time'];obs=z['observations'];target=z['targets'];parent=z['parent_means']
  assert obs.shape==(500,351) and target.shape==parent.shape==(500,2)
  assert np.isfinite(obs).all() and np.isfinite(target).all() and np.isfinite(parent).all()
  assert np.max(abs(t-np.arange(500)*.02))<2e-6
  expected=parent.copy();mask=(np.rint(t/.005)>=360)&(np.rint(t/.005)<760)
  expected[mask]*=np.asarray(d['recovery_intervention']['target_channel_masks'][row['policy']])
  assert np.array_equal(expected,target)
  sample_obs.append(obs[::17]);sample_actions.append(parent[::17])
  b=old[row['name']]
  checks.append(dict(name=row['name'],samples=len(t),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),task_changed=bool(row['task_recovered']!=b['task_recovered']),peak_roll_delta=row['peak_roll_rad']-b['peak_roll_rad'],tail_path_delta=row['tail_max_path_m']-b['tail_max_path_m']))
import jax,jax.numpy as j
from sttw_control.path_command_policy import make_path_actor
actor=make_path_actor(d['config'])
with Path(d['policies']['control']['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
obs=np.concatenate(sample_obs);expected=np.concatenate(sample_actions)
f=jax.jit(lambda x:actor.apply(params,x));pred=np.concatenate([np.asarray(f(j.asarray(obs[i:i+128]))) for i in range(0,len(obs),128)])
error=float(np.max(abs(pred-expected)))
result=dict(cases=len(checks),samples=sum(x['samples'] for x in checks),sampled_actor_recomputations=len(obs),max_actor_error=error,task_label_changes=sum(x['task_changed'] for x in checks),max_roll_delta=max(abs(x['peak_roll_delta']) for x in checks),max_tail_path_delta=max(abs(x['tail_path_delta']) for x in checks),records=checks)
(R/'record_verification.json').write_text(json.dumps(result,indent=2)+'\n')
if error>1e-4:raise RuntimeError('parent action reproduction mismatch; inspect before teaching')
print(json.dumps({k:v for k,v in result.items() if k!='records'}))
