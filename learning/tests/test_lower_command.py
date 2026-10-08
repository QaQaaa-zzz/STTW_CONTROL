import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from sttw_control.lower_command import reward_terms, command_rows, map_action, observation_from_history
SPEC=json.loads((Path(__file__).parents[1]/'configs/lower_command_tracking.json').read_text())
def test_reward_tracks_both_channels_without_flat_cost_cap():
 def r(v,d):return float(reward_terms(v,d,0.,jnp.zeros(2),jnp.zeros(2),False,SPEC)['reward'])
 assert r(0,0)>r(.1,0)>r(.5,0)>r(2.,0)
 assert r(0,0)>r(0,.03)>r(0,.15)>r(0,.6)
 assert float(reward_terms(0.,0.,.5,jnp.zeros(2),jnp.zeros(2),False,SPEC)['reward'])<r(0,0)
 assert float(reward_terms(0.,0.,0.,jnp.zeros(2),jnp.zeros(2),True,SPEC)['reward'])<r(2.,.6)
def test_schedule_randomized_reproducible_and_bounded():
 rows=[]
 for eid in range(4):
  a,s=command_rows(eid,0,2.3,SPEC);b,_=command_rows(eid,0,2.3,SPEC);np.testing.assert_array_equal(a,b)
  used=np.asarray(a)[np.asarray(a)[:,0]<99];assert np.all((used[:,1]>=1.5)&(used[:,1]<=3));assert np.max(np.abs(used[:,2]))<=.30001
  assert used[-1,0]==8 and used[-1,2]==0;rows.append(np.asarray(a))
 assert not np.array_equal(rows[0],rows[1])
def test_action_and_observation_contract():
 np.testing.assert_allclose(map_action(jnp.array([100.,-100.])),[1.,-1.])
 a,c=observation_from_history(jnp.zeros((10,20)),jnp.zeros(10),0,SPEC)
 assert a.shape==(210,) and c.shape==(211,);assert float(c[-1])==1.
 assert np.all(np.isfinite(a))
def test_schedule_uses_existing_publisher_and_returns_to_straight():
 from sttw_control.direct_command_scenarios import publish_command
 rows,slew=command_rows(1,0,2.3,SPEC)
 assert rows.shape==(16,3)
 issued,rates,target=publish_command(jnp.array([2.3,.1]),rows,1800,slew,.005)
 assert float(target[1])==0.
def test_lower_ppo_uses_global_kl_and_updates_actor():
 import torch
 from tensordict import TensorDict
 from sttw_control.lower_command_training import make_algorithm
 torch.set_num_threads(1);torch.manual_seed(5)
 obs=TensorDict({'policy':torch.randn(8,210),'critic':torch.randn(8,211)},batch_size=[8])
 algo=make_algorithm(obs,16,SPEC,'cpu')
 with torch.no_grad():
  for _ in range(16):
   algo.act(obs);algo.process_env_step(obs,torch.randn(8)*.01,torch.zeros(8,dtype=torch.bool),{})
  algo.compute_returns(obs)
 metrics=algo.update()
 assert metrics['actor_grad_norm']>0 and metrics['critic_grad_norm']>0
 assert set(metrics['kl_by_alpha'])=={'all'}
 assert metrics['accepted_epochs']>0 and not metrics['nonfinite_stop']
