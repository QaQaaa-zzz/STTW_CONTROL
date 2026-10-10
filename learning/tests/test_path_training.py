import json
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as j
from sttw_control import path_command_training as training
from sttw_control import path_command_scenarios as scenarios
CFG=json.loads((Path(__file__).parents[1]/'configs/STTW_Path_Feedback_V1.json').read_text())

def test_reset_sampler_is_deterministic_paired_and_fixed_shape():
 pose=j.array([3.,4.,.2]);a=scenarios.sample_route(jax.random.PRNGKey(7),pose,CFG)
 b=scenarios.sample_route(jax.random.PRNGKey(7),pose,CFG)
 for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):np.testing.assert_array_equal(x,y)
 assert a.path.xy.shape==(5001,2)
 assert 1.8<=float(a.speed)<=2.6
 assert np.all(np.diff(a.path.s)>0)
 assert np.isfinite(a.path.xy).all()
 # No alpha argument: endpoint methods consume identical routes for same key.
 batch=jax.jit(jax.vmap(lambda key:scenarios.sample_route(key,pose,CFG)))(jax.random.split(jax.random.PRNGKey(8),64))
 assert set(np.asarray(batch.family).tolist())=={0,1,2}
 assert not np.shares_memory(np.asarray(a.path.xy),np.asarray(pose))

def test_terminal_fault_and_rollout_edge_are_distinct():
 cfg=CFG
 live=training.termination_flags(j.int32(128*4),False,False,False,20.,cfg)
 assert not bool(live['done']) and bool(live['bootstrap'])
 end=training.termination_flags(j.int32(4000),False,False,False,20.,cfg)
 assert bool(end['finite_task_end']) and not bool(end['bootstrap'])
 bad=training.termination_flags(j.int32(20),False,False,True,20.,cfg,'terminal_contract_v1')
 assert bool(bad['engineering_fault']) and not bool(bad['physical_failure'])
 fail=training.termination_flags(j.int32(20),True,False,False,20.,cfg)
 assert bool(fail['physical_failure']) and bool(fail['done'])

def test_path_algorithm_shapes_std_and_real_ppo_update():
 import torch
 from tensordict import TensorDict
 obs=TensorDict({'policy':torch.zeros(4,351),'critic':torch.zeros(4,352)},batch_size=[4])
 algo=training.make_algorithm(obs,2,CFG,'cpu')
 torch.testing.assert_close(algo.policy.act_inference(obs),torch.zeros(4,2))
 torch.testing.assert_close(algo.policy.log_std.exp(),torch.tensor([.25,.10]))
 with torch.no_grad():
  for step in range(2):
   z=algo.act(obs);algo.process_env_step(obs,torch.tensor([1.,2.,3.,4.]),torch.tensor([False,False,False,step==1]),{})
  algo.compute_returns(obs)
 result=algo.update()
 assert not result['nonfinite_stop']
 assert result['accepted_epochs']>0
 assert algo.storage.actions.shape==(2,4,2)
 with pytest.raises(ValueError):training.make_algorithm(TensorDict({'policy':torch.zeros(4,345),'critic':torch.zeros(4,346)},batch_size=[4]),2,CFG,'cpu')

def test_checkpoint_schema_and_lower_identity_are_enforced(tmp_path):
 import torch
 from tensordict import TensorDict
 obs=TensorDict({'policy':torch.zeros(4,351),'critic':torch.zeros(4,352)},batch_size=[4]);algo=training.make_algorithm(obs,2,CFG,'cpu')
 identity={'schema':training.SCHEMA,'lower_actor_sha256':'fixed300','config_sha256':'fixed'}
 path=training.save_checkpoint(tmp_path,algo,10,identity,rejections=1)
 other=training.make_algorithm(obs,2,CFG,'cpu');saved=training.restore_checkpoint(path,other,identity)
 assert saved['update']==10 and saved['consecutive_rejections']==1
 for k,v in algo.policy.state_dict().items():torch.testing.assert_close(v,other.policy.state_dict()[k])
 with pytest.raises(ValueError):training.restore_checkpoint(path,other,{**identity,'lower_actor_sha256':'400'})

def test_selection_uses_original_path_and_never_adopts_partial():
 from sttw_control.path_command_selection import summarize,score_key
 t=np.arange(4000)*.005
 def trace(ey=.01,ev=.01):
  return dict(time=t,physical_failure=np.zeros(4000,bool),policy_fault=np.zeros(4000,bool),lower_fault=np.zeros(4000,bool),domain_exit=np.zeros(4000,bool),peak_roll=np.full(4000,.1),actual_forward_speed=np.full(4000,2.+ev),v_user=np.full(4000,2.),path_cross_track=np.full(4000,ey),path_heading_error=np.zeros(4000),chi=np.full(4000,1.),goal_section_signed_distance=np.ones(4000),path_progress=np.full(4000,20.),goal_progress=np.full(4000,10.))
 data={k:trace() for k in ['straight_2.0','straight_2.6','left90_R2_2.0','left90_R2_2.6','right90_R2_2.0','right90_R2_2.6']}
 s=summarize(data,0,CFG);assert s['qualified']
 off_route={**data,'left90_R2_2.0':{**trace(),'path_progress':np.full(4000,5.)}}
 assert not summarize(off_route,0,CFG)['cases']['left90_R2_2.0']['goal']
 changed={**data,'left90_R2_2.0':trace(ey=.2)}
 assert score_key(s)<score_key(summarize(changed,0,CFG))
 with pytest.raises(ValueError):summarize({k:v for k,v in data.items() if k!='straight_2.0'},0,CFG)


def test_paired_reward_reconstruction_reuses_saved_physics():
 from sttw_control.path_command_reporting import reward_series
 root=Path(__file__).parents[2]/'docs/evidence/path_feedback_v1_20261010'
 for name,horizon in [('straight',10.),('left90_R2',20.)]:
  d=dict(np.load(root/'data'/(name+'.npz')))
  r0,_=reward_series(d,0,CFG,horizon)
  np.testing.assert_allclose(r0,d['scored_tick_reward'],rtol=2e-4,atol=2e-6)
  r1,_=reward_series(d,1,CFG,horizon)
  assert np.isfinite(r1).all()
  if name=='left90_R2':assert not np.allclose(r0,r1)
  failed={k:(v[:7].copy() if v.ndim else v.copy()) for k,v in d.items()};failed['physical_failure'][-1]=True
  fr,fc=reward_series(failed,1,CFG,horizon)
  assert np.all(fr[4:6]==0) and fc['failure_tail'][-1]<-5
  np.testing.assert_allclose(fr,sum(fc.values()))


def test_resume_boundary_hash_and_route_counters(tmp_path):
 import torch
 from tensordict import TensorDict
 from sttw_control.preference_best import atomic_json,digest
 obs=TensorDict({'policy':torch.zeros(4,351),'critic':torch.zeros(4,352)},batch_size=[4])
 algo=training.make_algorithm(obs,2,CFG,'cpu');identity={'schema':training.SCHEMA}
 root=tmp_path/'alpha0';path=training.save_checkpoint(root/'checkpoints',algo,10,identity,route_counters=np.arange(4))
 atomic_json(root/'last_completed.json',dict(schema=training.SCHEMA,update=10,checkpoint=str(path),checkpoint_sha256=digest(path)))
 receipt,best=training.resume_source(tmp_path,0,identity)
 restored=training.restore_checkpoint(receipt['checkpoint'],algo,identity)
 np.testing.assert_array_equal(restored['route_counters'],np.arange(4))
 assert best is None
 with path.open('ab') as f:f.write(b'changed')
 with pytest.raises(ValueError):training.resume_source(tmp_path,0,identity)
