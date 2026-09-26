import torch
from sttw_control.rsl_training import bootstrap_timeout


def test_finite_resume_reaches_environment_but_actor_only_import_is_rejected(tmp_path,monkeypatch):
 import pytest
 from dataclasses import replace
 from sttw_control import env,rsl_training
 from sttw_control.training import TrainingConfig
 class EnvironmentReached(Exception):pass
 def boundary(*args,**kwargs):raise EnvironmentReached()
 monkeypatch.setattr(env,'RecoveryEnv',boundary)
 c=TrainingConfig(trainer='rsl',resume_checkpoint=str(tmp_path/'source'))
 with pytest.raises(EnvironmentReached):
  rsl_training.train('learning/configs/priority_v2_a.json',tmp_path/'resume',c)
 with pytest.raises(ValueError,match='cold start|Actor-only'):
  rsl_training.train('learning/configs/priority_v2_a.json',tmp_path/'actor',replace(c,resume_checkpoint=None,actor_init_checkpoint='actor',actor_init_training='source'))


def test_full_resume_restores_policy_adam_rng_and_rejects_wrong_identity():
 import copy,numpy as np,pytest
 from tensordict import TensorDict
 from sttw_control import rsl_training as r
 from sttw_control.training import TrainingConfig
 obs=TensorDict({'policy':torch.randn(4,12)},batch_size=[4])
 c=TrainingConfig(trainer='rsl',hidden_sizes=(8,8),activation='elu',rsl_schedule='fixed')
 def fresh():return r.make_algorithm(obs,2,1,1,'cpu',hidden_sizes=c.hidden_sizes,activation=c.activation,schedule='fixed')
 a=fresh()
 with torch.no_grad():
  for _ in range(2):
   z=a.act(obs);a.process_env_step(obs,-(z[:,0]-.3)**2,torch.zeros(4,dtype=torch.bool),{})
  a.compute_returns(obs)
 a.update()
 saved=copy.deepcopy({'policy':a.policy.state_dict(),'optimizer':a.optimizer.state_dict(),'identity':{'task':'same'},'rsl_version':r.RSL_VERSION,'activation':c.activation,'hidden_sizes':c.hidden_sizes,'learning_rate':a.learning_rate,'torch_rng':torch.get_rng_state(),'cuda_rng':[],'jax_rng':np.array([123,456],np.uint32),'update':100,'control_transitions':13107200})
 noise=torch.rand(5);b=fresh()
 key,offset,transitions=r.restore_training_snapshot(b,saved,{'task':'same'},c,'cpu')
 assert (offset,transitions)==(100,13107200)
 np.testing.assert_array_equal(key,[123,456]);torch.testing.assert_close(torch.rand(5),noise,rtol=0,atol=0)
 for k,v in saved['policy'].items():torch.testing.assert_close(b.policy.state_dict()[k],v,rtol=0,atol=0)
 for i,s in saved['optimizer']['state'].items():
  for k,v in s.items():torch.testing.assert_close(b.optimizer.state_dict()['state'][i][k],v,rtol=0,atol=0)
 with pytest.raises(ValueError,match='identity mismatch'):r.restore_training_snapshot(b,saved,{'task':'changed'},c,'cpu')


def test_resume_keeps_earlier_development_for_three_point_gate(tmp_path):
 import json,pytest
 from sttw_control import rsl_training as r
 from sttw_control.priority_v2_analysis import development_stop_reason
 source=tmp_path/'source';cp=source/'checkpoints/update_0100';cp.mkdir(parents=True)
 identity={'task':'unchanged'}
 (source/'declaration.json').write_text(json.dumps({'policy_identity':identity}))
 a=dict(update=0,Jp=1.,Jv=5.,physical_failures=0)
 b=dict(a,update=100,Jp=1.1);later=dict(a,update=200,Jp=1.3)
 (source/'development_history.json').write_text(json.dumps([a,b,later]))
 history=r.resume_development_history(cp,100,identity)
 assert history==[a]  # Offset is re-evaluated; later checkpoints cannot leak in.
 assert development_stop_reason(history+[b,later]) is not None
 with pytest.raises(ValueError,match='identity mismatch'):r.resume_development_history(cp,100,{'task':'different'})


def test_legacy_resume_does_not_silently_clear_unknown_rollback_streak(tmp_path):
 import pytest
 from sttw_control.rsl_training import resume_rollback_count
 cp=tmp_path/'checkpoints/update_0100';cp.mkdir(parents=True)
 previous=cp.parent/'update_0099';previous.mkdir()
 torch.save({'policy':{'weight':torch.tensor([1.])}},previous/'rsl_snapshot.pt')
 assert resume_rollback_count({'consecutive_rollbacks':2},cp)==2
 assert resume_rollback_count({'update':100,'policy':{'weight':torch.tensor([2.])}},cp)==0
 with pytest.raises(ValueError,match='rollback history'):
  resume_rollback_count({'update':100,'policy':{'weight':torch.tensor([1.])}},cp)

def test_finite_task_end_does_not_bootstrap():
 r=torch.tensor([-20.,-200.,-1.]);v=torch.tensor([9.,9.,9.]);tr=torch.tensor([True,False,True]);term=torch.tensor([False,True,False]);task=torch.tensor([True,False,False])
 out=bootstrap_timeout(r,v,tr,term,.9,task_terminal=task)
 assert torch.allclose(out,torch.tensor([-20.,-200.,7.1]))

def test_legacy_timeout_unchanged():
 r=torch.tensor([1.]);assert torch.allclose(bootstrap_timeout(r,torch.tensor([2.]),torch.tensor([True]),torch.tensor([False]),.9),torch.tensor([2.8]))


def test_selection_and_three_point_stop_do_not_force_last_or_reward():
 from sttw_control.priority_v2_analysis import select,development_stop_reason
 a=dict(update=0,Jp=1.,Jv=5.,engineering_accepted=False,strict_accepted=False,physical_failures=0)
 b=dict(a,update=8,Jp=1.01,Jv=2.)
 c=dict(a,update=16,Jp=1.04,Jv=1.)
 chosen=select([a,b,c])
 assert chosen['best_diagnostic']['update']==8
 assert chosen['best_accepted'] is None
 assert development_stop_reason([a,b]) is None
 assert development_stop_reason([a,b,c]) is not None
 assert development_stop_reason([a,b,dict(c,Jp=.9)]) is None


def test_rsl_cold_start_pairing_and_zero_mean():
 from tensordict import TensorDict
 from sttw_control.rsl_training import make_algorithm
 def fresh():
  torch.manual_seed(66)
  return make_algorithm(TensorDict({'policy':torch.zeros((2,40))},batch_size=[2]),2,2,1,'cpu',std=.1,activation='elu',hidden_sizes=(128,128,128),schedule='fixed')
 a,b=fresh(),fresh()
 assert not a.optimizer.state and not b.optimizer.state
 for k,v in a.policy.state_dict().items():assert torch.equal(v,b.policy.state_dict()[k])
 assert torch.equal(a.policy.act_inference(TensorDict({'policy':torch.randn((2,40))},batch_size=[2])),torch.zeros((2,2)))


def test_replay_keeps_float32_reference_progress_at_exact_exit_crossing():
 import numpy as np
 from sttw_control.priority_return_v2 import initial_state,advance
 from sttw_control.priority_v2_analysis import replay
 n=800;dt=.005
 command=np.tile(np.array([2.3,0.],np.float32),(n+1,1));command[:400,1]=1.
 arc=np.r_[np.float32(0),np.cumsum(command[:-1,0]*np.float32(dt),dtype=np.float32)]
 trace={'time':np.arange(n+1,dtype=np.float32)*np.float32(dt),'reference_command':command,
        'raw_reference_request':command,'true_forward_speed':command[:,0].copy(),
        'path_features':np.zeros((n+1,2),np.float32),'measurement':np.zeros((n+1,2),np.float32),
        'terminated':np.zeros(n+1,bool),'path_progress':np.zeros(n+1,np.float32),
        'priority_alpha':np.zeros(n+1),'action':np.zeros((n+1,2))}
 s=initial_state(xp=np);online=[]
 for i in range(1,n+1):
  progress=float(s.exit_progress) if i>=500 else 0.
  trace['path_progress'][i]=progress
  s,_=advance(s,t=float(trace['time'][i]),dt=dt,episode_end=4.,reference_speed=float(command[i,0]),reference_yaw=float(command[i,1]),published_yaw_request=float(command[i,1]),previous_reference_speed=float(command[i-1,0]),reference_progress=arc[i],actual_progress=progress,speed_error=0.,lateral_error=0.,heading_error=0.,roll=0.,roll_rate=0.,xp=np)
  online.append(float(s.return_start))
 _,rows,_=replay(trace,{'controller':{'dt':dt},'horizon_seconds':4.})
 np.testing.assert_allclose([s['return_start'] for s in rows],online,atol=1e-6,rtol=0)
