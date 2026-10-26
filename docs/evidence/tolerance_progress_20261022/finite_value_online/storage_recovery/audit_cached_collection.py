"""Read-only authentication of pending collection46; no physics or optimization."""
import argparse,json,time,zipfile
from pathlib import Path
import numpy as np
from sttw_control.finite_value_recovery import restore_models
from sttw_control.survival_live_batch import assemble_live_arm
from sttw_control.preference_best import atomic_json,digest

def audit(root):
 root=Path(root).resolve();folder=root/'execution/update_0046'
 plan=json.loads((root/'plan.json').read_text());pair=json.loads((root/'checkpoint_pair_0045.json').read_text())
 assert pair['completed_updates']==45 and pair['plan_sha256']==digest(root/'plan.json')
 cfg=json.loads((root/'effective_config.json').read_text())
 learners=restore_models(cfg,dict(resume_models={r['arm']:r for r in pair['models']},resume_boundary=45))
 pending=json.loads((folder/'sampling_identity.json').read_text())
 assert pending['cases']==json.loads((root/'training_cases.json').read_text())[45]
 assert pending['plan_sha256']==digest(root/'plan.json')
 for name,obj in learners.items():assert obj.freeze_collection()==pending['models'][name]
 decisions=[];physical=[];records=[];files=sorted(folder.glob('chunk_*.npz'));assert len(files)==10
 for i,file in enumerate(files):
  assert file.name==f'chunk_{i:02d}.npz'
  with zipfile.ZipFile(file) as archive:assert archive.testzip() is None
  with np.load(file,allow_pickle=False) as z:
   d={k[9:]:z[k] for k in z.files if k.startswith('decision_')}
   p={k[6:]:z[k] for k in z.files if k.startswith('lower_')}
  assert d['valid'].shape==(50,180) and p['active_tick'].shape==(200,180)
  decisions.append(d);physical.append(p)
  records.append(dict(file=str(file),sha256=digest(file),active_lower_ticks=int(p['active_tick'].sum())))
 d={k:np.concatenate([v[k] for v in decisions]) for k in decisions[0]}
 p={k:np.concatenate([v[k] for v in physical]) for k in physical[0]};rows={}
 for name,obj in learners.items():
  ids=[i for i,c in enumerate(pending['cases']) if c['arm']==name]
  _,rows[name]=assemble_live_arm(d,p,pending['cases'],ids,collection_id=pending['models'][name]['collection_id'],dt_s=.005,horizon_s=10.,gamma=plan['gamma'],gae_lambda=obj.gae_lambda)
  assert len(rows[name])==60
 consolidated={}
 for name in ['decisions.npz','physical.npz']:
  try:
   with zipfile.ZipFile(folder/name) as archive:bad=archive.testzip()
   consolidated[name]=dict(valid_zip=bad is None,bad_member=bad)
  except Exception as exc:consolidated[name]=dict(valid_zip=False,error=repr(exc))
 result=dict(phase='completed',checked_unix=time.time(),physics_steps=0,optimizer_updates=0,
  authenticated_boundary=45,optimizer_steps_by_arm={n:o.optimizer_steps for n,o in learners.items()},
  sampling_identity_reproduced=True,chunk_files=records,case_summaries=rows,
  consolidated=consolidated,cached_allocated_lower_slots=360000,
  cached_actual_lower_ticks=sum(r['active_lower_ticks'] for r in records),all_180_outcomes_known=True)
 atomic_json(root/'cached46_audit.json',result)
 return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args()
 result=audit(args.root);print(json.dumps({k:v for k,v in result.items() if k not in ['case_summaries','chunk_files']},indent=2))
