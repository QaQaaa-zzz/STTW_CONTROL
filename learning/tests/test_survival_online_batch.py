"""Paired episode assignment and completed physical-tail batch semantics."""
import numpy as np
import pytest
from sttw_control.survival_online_batch import make_online_cases,assemble_online_arm


def test_cases_balance_three_alphas_and_pair_noise_across_arms_and_force():
    reference=[dict(route='left',force_N=f,rng_fold_in=0) for f in [0.,32.]]+[
        dict(route='right',force_N=f,rng_fold_in=1) for f in [0.,-32.]]
    cases=make_online_cases(reference,alphas=(0.,.5,1.),replicates=4)
    assert len(cases)==96
    for arm in [0,1]:
        subset=[c for c in cases if c['arm_index']==arm]
        assert [sum(c['alpha']==a for c in subset) for a in [0.,.5,1.]]==[16,16,16]
    for i in range(48):assert cases[i]['rng_fold_in']==cases[48+i]['rng_fold_in']
    assert cases[0]['rng_fold_in']==cases[12]['rng_fold_in']


def trace():
    n=24;steps=n//4;valid=np.array([True,True,False,False,False,False])
    d=dict(valid=valid[:,None],observation=np.full((steps,1,351),np.nan,dtype=np.float32),
        critic_observation=np.full((steps,1,352),np.nan,dtype=np.float32),
        raw_action=np.full((steps,1,2),np.nan,dtype=np.float32),mean=np.full((steps,1,2),np.nan,dtype=np.float32),
        log_prob=np.full((steps,1),np.nan,dtype=np.float32),old_task_value=np.full((steps,1),np.nan,dtype=np.float32),
        physical_value=np.full((steps,1),np.nan,dtype=np.float32),task_tick=np.full((steps,1),-1,dtype=np.int32))
    for name in ['observation','raw_action','mean','log_prob','old_task_value']:d[name][:2]=0.
    d['physical_value'][:2]=.5;d['task_tick'][:2,0]=[0,4]
    d['critic_observation'][:2,0,:351]=0.;d['critic_observation'][:2,0,-1]=[1.,1.-4/n]
    p={k:np.zeros((n,1),dtype=bool) for k in ['contact_failure','nonfinite','lower_fault','policy_fault','horizon_reached','domain_exit','task_reward_valid','fallback_active']}
    p['active_tick']=np.ones((n,1),bool);p['physical_tick']=np.arange(n,dtype=np.int32)[:,None]
    p['task_reward_valid'][:8]=True;p['domain_exit'][7:]=True;p['fallback_active'][8:]=True
    p['contact_failure'][23]=True
    p['scored_tick_reward']=np.zeros((n,1),np.float32);p['scored_tick_reward'][:8]=-1.
    p['actual_forward_speed']=np.ones((n,1),np.float32);p['path_cross_track']=np.zeros((n,1),np.float32)
    p['time']=np.arange(n,dtype=np.float32)[:,None]*.005;p['applied_force_impulse']=np.zeros((n,1,3),np.float32)
    return d,p


def test_contact_after_domain_hold_labels_only_the_two_earlier_actor_decisions():
    d,p=trace();cases=[dict(alpha=0.,force_N=0.,pulse_duration_s=.1,world_direction=[0,1,0])]
    batch,summary=assemble_online_arm(d,p,cases,[0],collection_id='frozen',dt_s=.005,horizon_s=.12,gamma=.9,gae_lambda=.8)
    assert len(batch['obs'])==2
    np.testing.assert_array_equal(batch['physical_targets'],[0.,0.])
    np.testing.assert_allclose(batch['task_returns'],[-6.88,-4.])
    assert summary[0]['hold_ticks']==16 and summary[0]['outcome']=='contact'


def test_incomplete_physical_tail_cannot_become_a_training_failure():
    d,p=trace();p['contact_failure'][:]=False
    with pytest.raises(ValueError,match='complete'):
        assemble_online_arm(d,p,[dict(alpha=0.)],[0],collection_id='frozen',dt_s=.005,horizon_s=.12,gamma=.9,gae_lambda=.8)
