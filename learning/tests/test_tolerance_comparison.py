import pytest
from sttw_control.tolerance_comparison import chunk_schedule,paired_counts,comparison_layout

def test_chunk_schedule_hits_every_fractional_onset_exactly():
 for onset in [.8,1.5,2.5,3.5]:
  chunks=chunk_schedule(onset)
  assert sum(n for _,n,_ in chunks)==1000
  assert [i for i,_,branch in chunks if branch]==[round(onset/.02)]
  assert all(0<n<=50 for _,n,_ in chunks)
  assert all(i+n==chunks[k+1][0] for k,(i,n,_) in enumerate(chunks[:-1]))
 with pytest.raises(ValueError):chunk_schedule(.81)

def test_layout_keeps_zero_branch_in_own_policy_context():
 p=dict(policies=['parent','baseline_update50','static_update50','dynamic_update50'],routes=['left90_R2','right90_R2'],speeds=[2.45,2.65],endpoints=[0,1],forces=[0,-2,2,-4,4,-6,6,-8,8])
 lanes,indices=comparison_layout(p);assert len(lanes)==288
 for i,r in enumerate(lanes):
  z=lanes[indices[i]];assert z['force_N']==0
  assert all(z[k]==r[k] for k in ['policy','route','speed','alpha'])

def test_pairing_preserves_losses_and_rejects_missing_cases():
 def row(policy,force,success):return dict(policy=policy,route='r',speed=2.,alpha=0,onset_s=1.,force_N=force,success=success)
 rows=[row('a',0,True),row('a',2,False),row('b',0,False),row('b',2,True)]
 x=paired_counts(rows,'a','b',0);assert x['improved']==1 and x['regressed']==1 and x['difference']==0
 with pytest.raises(ValueError):paired_counts(rows[:-1],'a','b',0)

def test_checkpoint_readiness_does_not_accept_partial_training(tmp_path):
 import json
 from sttw_control.tolerance_comparison import resolve_comparison_actors
 (tmp_path/'campaign.json').write_text(json.dumps(dict(jobs=['baseline_alpha0'])))
 job=tmp_path/'baseline_alpha0';job.mkdir()
 (job/'status.json').write_text(json.dumps(dict(phase='running',completed_updates=49)))
 with pytest.raises(ValueError,match='training not complete'):
  resolve_comparison_actors(tmp_path)

def test_paired_counts_reject_duplicate_and_unknown_outcomes():
 row=dict(policy='a',route='r',speed=2.,alpha=0,onset_s=1.,force_N=0,success=True)
 with pytest.raises(ValueError,match='duplicate'):
  paired_counts([row,row,dict(row,policy='b')],'a','b',0)
 with pytest.raises(ValueError,match='binary'):
  paired_counts([row,dict(row,policy='b',success=None)],'a','b',0)


@pytest.mark.parametrize('experiment,final_update',[('actual_pulse_curriculum_v1',25),('short_recovery_v1',27)])
def test_pulse_resolver_fixed_final_and_shared_parent(tmp_path,experiment,final_update):
 import json,pickle
 import numpy as np
 import torch
 from sttw_control.tolerance_comparison import resolve_comparison_actors
 from sttw_control.preference_best import digest
 weights={};exported={}
 for i,layer in enumerate((0,2,4,6)):
  weights[f'actor.{layer}.weight']=torch.ones(2,2)*i;weights[f'actor.{layer}.bias']=torch.zeros(2)
  exported[f'Dense_{i}']={'kernel':np.ones((2,2),np.float32)*i,'bias':np.zeros(2,np.float32)}
 jobs=[]
 for alpha in (0,1):
  parent=tmp_path/f'parent{alpha}';parent.mkdir();cp=parent/'update_0050.pt';identity={'arm':'baseline','alpha':alpha}
  torch.save({'identity':identity,'update':50,'policy':weights},cp)
  with (parent/'actor_0050.pkl').open('wb') as f:pickle.dump({'params':exported},f)
  rec={'checkpoint':str(cp),'checkpoint_sha256':digest(cp),'identity':identity}
  for arm in ('uniform','guided'):
   name=f'{arm}_alpha{alpha}';jobs.append(name);job=tmp_path/name;check=job/'checkpoints';check.mkdir(parents=True);ident={'arm':arm,'alpha':alpha}
   (job/'status.json').write_text(json.dumps({'phase':'completed','completed_updates':final_update}));(job/'plan.json').write_text(json.dumps({'arm':arm,'alpha':alpha,'parent_record':rec}));(job/'manifest.json').write_text(json.dumps({'identity':ident}));final=check/f'update_{final_update:04d}.pt';torch.save({'identity':ident,'update':final_update,'policy':weights},final)
   with (check/f'actor_{final_update:04d}.pkl').open('wb') as f:pickle.dump({'params':exported},f)
   (job/'last_completed.json').write_text(json.dumps({'update':final_update,'checkpoint':str(final),'checkpoint_sha256':digest(final)}))
 (tmp_path/'campaign.json').write_text(json.dumps({'experiment':experiment,'jobs':jobs}))
 assert len(resolve_comparison_actors(tmp_path))==6
 path=tmp_path/'guided_alpha0/plan.json';plan=json.loads(path.read_text());plan['parent_record']['checkpoint_sha256']='changed';path.write_text(json.dumps(plan))
 with pytest.raises(ValueError,match='different parents'):resolve_comparison_actors(tmp_path)
