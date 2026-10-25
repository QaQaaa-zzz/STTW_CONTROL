import numpy as np
import pytest
from sttw_control.survival_live_batch import assemble_live_arm
from sttw_control.survival_task_credit import reconstruct_task_critic


def fixture():
    valid=np.array([True,False,False,True]);obs=np.full((4,351),np.nan,dtype=np.float32);obs[valid]=0.
    tick=np.array([0,-1,-1,12]);critic=reconstruct_task_critic(obs,tick,valid,dt_s=.005,horizon_s=.08)
    scalar=lambda a:np.array(a,dtype=np.float32)
    d=dict(valid=valid,observation=obs,critic_observation=critic,task_tick=tick,
        raw_action=np.where(valid[:,None],np.zeros((4,2)),np.nan),mean=np.where(valid[:,None],np.zeros((4,2)),np.nan),
        log_prob=scalar([0.,np.nan,np.nan,0.]),old_task_value=scalar([.5,np.nan,np.nan,.75]),physical_value=scalar([.5,np.nan,np.nan,.5]))
    p=dict(active_tick=np.ones(16,dtype=bool),contact_failure=np.zeros(16,dtype=bool),
        nonfinite=np.zeros(16,dtype=bool),lower_fault=np.zeros(16,dtype=bool),policy_fault=np.zeros(16,dtype=bool),
        horizon_reached=np.arange(16)==15,physical_tick=np.arange(16),task_reward_valid=np.ones(16,dtype=bool),
        scored_tick_reward=np.repeat([1.,2.,3.,4.],4)/4,fallback_active=np.repeat(~valid,4),
        actual_forward_speed=np.ones(16),path_cross_track=np.zeros(16),applied_force_impulse=np.zeros((16,3)),
        reentry_actor_active=np.arange(16)>=12)
    return {k:v[:,None] for k,v in d.items()},{k:v[:,None] for k,v in p.items()}


def test_complete_fallback_rewards_are_folded_without_fake_policy_samples():
    d,p=fixture();batch,rows=assemble_live_arm(d,p,[dict(alpha=.5,force_N=0.)],[0],collection_id='frozen',dt_s=.005,horizon_s=.08,gamma=.9,gae_lambda=.8)
    assert batch['obs'].shape==(2,351)
    np.testing.assert_allclose(batch['task_returns'].numpy(),[6.989806,4.],rtol=1e-6)
    np.testing.assert_allclose(batch['physical_targets'].numpy(),[1.,1.])
    assert rows[0]['task_scored_return']==10.
    assert rows[0]['actor_decisions']==2 and rows[0]['fallback_ticks']==8
    assert rows[0]['reentry_decisions']==1
    assert batch['case_ranges']==[dict(case_index=0,start=0,end=2)]


def test_old_prefix_only_reward_contract_is_not_silently_accepted():
    d,p=fixture();p['task_reward_valid'][4:]=False
    with pytest.raises(ValueError,match='live task rewards'):
        assemble_live_arm(d,p,[dict(alpha=.5)],[0],collection_id='frozen',dt_s=.005,horizon_s=.08,gamma=.9,gae_lambda=.8)


def test_unknown_physical_episode_cannot_supply_failure_targets():
    d,p=fixture();p['nonfinite'][15]=True
    with pytest.raises(ValueError,match='known physical'):
        assemble_live_arm(d,p,[dict(alpha=.5)],[0],collection_id='frozen',dt_s=.005,horizon_s=.08,gamma=.9,gae_lambda=.8)
