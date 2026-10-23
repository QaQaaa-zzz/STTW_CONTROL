"""Prepare equal-demo selection controls from immutable TRAIN acquisitions."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2];R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_names=['preference_train_coordination_records_v1','tolerance_recovery_opportunity_v1']
pool=[];witness=[];sources=[]
for n in source_names:
 root=ROOT/'runs'/n;d=json.loads((root/'declaration.json').read_text());rows=json.loads((root/'evaluation/results.json').read_text())['cases']
 assert d['teaching_records']=={'role':'TRAIN','observation_dim':351}
 status=json.loads((root/'evaluation/status.json').read_text());assert status['phase']=='completed'
 sources.append(dict(run=n,declaration_sha256=sha(root/'declaration.json'),results_sha256=sha(root/'evaluation/results.json'),cases=len(rows),lower_ticks=status['active_lower_ticks']))
 key=lambda x:tuple(x[k] for k in ['route','speed','alpha','force_N','onset_s'])
 controls={key(x):x for x in rows if x['policy']=='control'}
 for row in rows:
  if row['policy']!='speed_off' or row['force_N']==0 or not row['task_recovered'] or row['domain_failure'] or row['physical_failure']:continue
  baseline=controls[key(row)];record=dict(run=n,name=row['name'],file=str(root/'evaluation'/(row['name']+'_teaching.npz')),start=d['recovery_intervention']['start_s'],end=d['recovery_intervention']['end_s'],useful=not baseline['task_recovered'] and row['logged_return']>baseline['logged_return'])
  record['sha256']=sha(record['file']);pool.append(record)
  if record['useful']:witness.append(record)
assert len(pool)==34 and len(witness)==10
original=ROOT/'runs/preference_train_coordination_records_v1/dataset'
oldmanifest=json.loads((original/'manifest.json').read_text())
base=json.loads((ROOT/'runs/preference_coordination_teaching_v1/declaration.json').read_text())
plan=dict(experiment='preference_witness_selection_v1',role='TRAIN',question='Do counterfactual recovery witnesses teach better than equally many random successful interventions?',sources=sources,pool_size=len(pool),witness_count=len(witness),arms=['witness','random_success'],optimization_seeds=[918501,918502,918503],selection_seeds=[923501,923502,923503],updates=400,batch_per_group=128,acquisition_cost='same already acquired240cases charged equally; no saving claim',evaluation='predeclare before launch; same fixed DEV panels; no final TEST',training_launched=False,jobs=[])
for seed,selection_seed in zip(plan['optimization_seeds'],plan['selection_seeds']):
 for arm in plan['arms']:
  chosen=witness if arm=='witness' else [pool[i] for i in np.random.default_rng(selection_seed).choice(len(pool),len(witness),replace=False)]
  job=R/f'{arm}_{seed}';dataset=job/'dataset';dataset.mkdir(parents=True,exist_ok=False);manifest=dict(role='TRAIN only',groups={},selected=chosen,selection_seed=selection_seed if arm=='random_success' else None)
  for group in ['intervention','recovery_tail']:
   xs=[];ys=[];case_ids=[]
   for case,record in enumerate(chosen):
    assert sha(record['file'])==record['sha256']
    with np.load(record['file']) as z:
     t=z['time'];mask=(t>=record['start']-1e-5)&(t<record['end']-1e-5) if group=='intervention' else t>=record['end']-1e-5
     x=z['observations'][mask];y=z['targets'][mask]
     assert x.shape[1]==351 and len(x)>0 and np.isfinite(x).all() and np.isfinite(y).all()
     xs.append(x);ys.append(y);case_ids.extend([case]*len(x))
   out=dataset/(group+'.npz');np.savez_compressed(out,observations=np.concatenate(xs),targets=np.concatenate(ys),case_ids=np.array(case_ids))
   manifest['groups'][group]=dict(samples=len(case_ids),cases=len(chosen),sha256=sha(out))
  for group in ['control','nominal']:
   src=original/(group+'.npz');assert sha(src)==oldmanifest['groups'][group]['sha256'];shutil.copy2(src,dataset/src.name);manifest['groups'][group]=oldmanifest['groups'][group]
  (dataset/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
  declaration=dict(base,experiment_id=job.name,seed=seed,dataset=str(dataset),dataset_manifest_sha256=sha(dataset/'manifest.json'),selection_arm=arm,selection_seed=manifest['selection_seed'])
  (job/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
  plan['jobs'].append(dict(name=job.name,seed=seed,arm=arm,declaration_sha256=sha(job/'declaration.json'),useful_selected=sum(x['useful'] for x in chosen),groups=manifest['groups']))
(R/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({k:plan[k] for k in ['pool_size','witness_count','sources']},indent=2));print([(j['name'],j['useful_selected']) for j in plan['jobs']])
