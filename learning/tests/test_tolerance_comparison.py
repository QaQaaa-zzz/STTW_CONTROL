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
