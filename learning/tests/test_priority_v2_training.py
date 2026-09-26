import torch
from sttw_control.rsl_training import bootstrap_timeout

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
