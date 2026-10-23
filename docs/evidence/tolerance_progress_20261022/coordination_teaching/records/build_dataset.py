"""Assemble four explicit training strata from verified records and nominal anchors."""
from pathlib import Path
import json,hashlib
import numpy as np
from sttw_control.recovery_teaching import select_training_rows,teaching_enabled
R=Path(__file__).resolve().parent;run=R/'evaluation'
d=json.loads((R/'declaration.json').read_text());assert teaching_enabled(d)
v=json.loads((R/'record_verification.json').read_text());assert v['cases']==72 and v['max_actor_error']<=1e-4
assert json.loads((run/'status.json').read_text())['phase']=='completed'
rows=json.loads((run/'results.json').read_text())['cases'];recover,retain=select_training_rows(rows)
if not recover:raise RuntimeError('no reward-supported recovery examples; stop teaching route')
out=R/'dataset';out.mkdir(exist_ok=False);sources=[];groups={};spec=d['recovery_intervention']
for group in ('intervention','recovery_tail','control'):
 obs=[];targets=[];case_ids=[]
 for i,r in enumerate(retain if group=='control' else recover):
  p=run/(r['name']+'_teaching.npz');sha=hashlib.sha256(p.read_bytes()).hexdigest()
  assert sha==next(x['sha256'] for x in v['records'] if x['name']==r['name'])
  with np.load(p) as z:
   t=z['time'];ticks=np.rint(t/.005)
   mask=np.ones(len(t),bool) if group=='control' else ((ticks>=round(spec['start_s']/.005))&(ticks<round(spec['end_s']/.005))) if group=='intervention' else ticks>=round(spec['end_s']/.005)
   obs.append(z['observations'][mask]);targets.append(z['targets'][mask]);case_ids.append(np.full(mask.sum(),i))
  sources.append(dict(group=group,case=i,name=r['name'],file=str(p),sha256=sha))
 data=dict(observations=np.concatenate(obs),targets=np.concatenate(targets),case_ids=np.concatenate(case_ids));np.savez_compressed(out/(group+'.npz'),**data)
 groups[group]=dict(samples=len(data['targets']),cases=len(obs),sha256=hashlib.sha256((out/(group+'.npz')).read_bytes()).hexdigest())
bank=R.parent/'preference_coverage_recovery_v1/bank';p=bank/'anchors.npz';receipt=json.loads((bank/'receipt.json').read_text());sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==receipt['anchors_sha256']
with np.load(p) as z:
 obs=z['observations'];target=z['means'];assert obs.shape[1]==351 and target.shape==(len(obs),2)
 np.savez_compressed(out/'nominal.npz',observations=obs,targets=target)
groups['nominal']=dict(samples=len(obs),source=str(p),source_sha256=sha,sha256=hashlib.sha256((out/'nominal.npz').read_bytes()).hexdigest())
(out/'manifest.json').write_text(json.dumps(dict(role='TRAIN only; no DEV/TEST trajectories',groups=groups,sources=sources,recovery_cases=[r['name'] for r in recover],retention_cases=[r['name'] for r in retain],semantics='Targets are latent pre-filter commands. Group separation permits balanced sampling; no force/time/case oracle appended to actor obs. Recovery-tail teacher is restored parent evaluated on the intervention-induced state.'),indent=2)+'\n');print(json.dumps(groups))
