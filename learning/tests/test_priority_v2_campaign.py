import importlib.util
from pathlib import Path
import numpy as np
import jax
import pytest
from sttw_control.env import config_from_dict
from sttw_control.training import TrainingConfig,normalization
from sttw_control.timed_reference import schedule
from sttw_control.observation import observation_fields
spec=importlib.util.spec_from_file_location('campaign',Path('learning/cli/priority_v2_campaign.py'))
campaign=importlib.util.module_from_spec(spec);spec.loader.exec_module(campaign)


def test_frozen_arms_and_scenes_share_exact_budget_and_shapes():
 arms=campaign.configurations();sizes=[]
 for name,arm in arms.items():
  cfg=config_from_dict(arm['task']);ppo=TrainingConfig(**arm['training'])
  assert ppo.num_envs*ppo.rollout_steps*ppo.updates==2097152
  assert ppo.actor_init_checkpoint is None and ppo.resume_checkpoint is None
  assert not ppo.training_reward_best_enabled
  sizes.append(len(normalization(cfg)[0]))
  assert sizes[-1]==10*(len(observation_fields(cfg.observation))+1)
  scenes,_=campaign.scenarios(cfg)
  assert set(scenes)==set(campaign.SCENES)
  assert all(not c.timed_reference.priority_v2_screen for c in scenes.values())
 assert len(set(sizes))==1
 a,b=arms['A']['task'].copy(),arms['B']['task'].copy()
 assert a['tracking']['objective']=='soft_budget_v1'
 a['tracking']=dict(a['tracking'],objective='priority_return_v2')
 assert a==b


def test_mixture_preserves_core_schedule_and_has_no_forcing():
 cfg=config_from_dict(campaign.configurations()['B']['task'])
 batch=np.asarray(jax.vmap(lambda k:schedule(k,cfg.timed_reference,cfg.speed_reference))(jax.random.split(jax.random.PRNGKey(66),1000)))
 assert set(batch[:,0,6])=={0.,1.}
 fraction=[np.mean(batch[:,0,6]==0),np.mean(batch[:,1,2]>0),np.mean(batch[:,1,2]<0)]
 np.testing.assert_allclose(fraction,[.3,.35,.35],atol=.04)
 core=batch[batch[:,0,6]==1]
 np.testing.assert_allclose(core[:,1,0],1.)
 np.testing.assert_allclose(core[:,2,0],1.95)
 np.testing.assert_allclose(core[:,1,1],2.5)
 np.testing.assert_allclose(abs(core[:,1,2]),1.8)


def test_absorbing_failure_once_and_blocked_screen(tmp_path):
 from sttw_control.priority_v2_analysis import absorbed_returns
 r=absorbed_returns([-1.,-200.],2000)
 assert r['undiscounted']==-201 and r['absorbing_transitions']==1998
 with pytest.raises(RuntimeError,match='screen blocked'):campaign.screen(tmp_path,campaign.configurations())
 assert not (tmp_path/'A/training').exists()
