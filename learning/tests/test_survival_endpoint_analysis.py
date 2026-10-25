"""Outcome and utility calculations must keep physical and task facts separate."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
import pytest
from sttw_control.path_command_reward import costs, configured_cost_keys
from sttw_control.survival_endpoint_analysis import rescore_v3, summarize_case


def reward_config():
    return dict(path_reward=json.loads((Path(__file__).parents[1]/'configs/STTW_Tolerance_V3_Reward.json').read_text()),
                timing={'lower_dt_s': .005})


def test_cross_scoring_matches_original_reward_and_preserves_terminal_penalty():
    cfg=reward_config(); n=4
    values=dict(chi=j.array([0.,.3,1.,1.]),ev=j.array([.01,-.4,2.,-.8]),
        ey=j.array([.01,.2,4.,-.3]),eh=j.array([.01,.1,-2.,.4]),
        peak_roll=j.array([.1,.3,.4,.2]),roll_rate=j.array([.1,1.,2.,.2]),
        speed=j.array([1.51,1.1,3.5,.7]),offsets=j.zeros((n,2)),offset_rate=j.zeros((n,2)))
    evaluate=lambda alpha:jax.vmap(lambda **kw:costs(alpha=alpha,config=cfg,**kw))(**values)
    initial=evaluate(.5); keys=configured_cost_keys(cfg)
    p=dict(active_tick=np.ones(n,bool),task_reward_valid=np.ones(n,bool),
        actual_forward_speed=np.asarray(values['speed']),requested_speed=np.full(n,1.5),
        path_cross_track=np.asarray(values['ey']),path_heading_error=np.asarray(values['eh']),
        chi=np.asarray(values['chi']),effective_costs=np.stack([initial['effective'][k] for k in keys],axis=1),
        scored_tick_reward=np.asarray(initial['reward']).copy())
    p['scored_tick_reward'][-1]-=37.  # Preserve a factual terminal charge.
    for alpha in [0.,.25,.5,.75,1.]:
        result=rescore_v3(p,user_alpha=alpha,config=cfg)
        target=np.asarray(evaluate(alpha)['reward']).copy();target[-1]-=37.
        np.testing.assert_allclose(result['per_lower_tick_reward'],target,atol=1e-5,rtol=1e-5)
        assert result['task_valid_ticks']==n
    # A held-reference tail has no invented task costs/rewards, even with NaNs.
    tail={k:np.concatenate([v,np.full((1,)+v.shape[1:],False if v.dtype==bool else np.nan)]) for k,v in p.items()}
    assert rescore_v3(tail,user_alpha=.5,config=cfg)['task_valid_ticks']==4


def episode():
    n=12;active=np.zeros(n,bool);active[[0,1,2,4,5,6,7,8]]=True
    clock=np.full(n,-1,dtype=np.int32);clock[active]=np.arange(8)
    p={k:np.zeros(n,bool) for k in ['nonfinite','lower_fault','policy_fault','contact_failure','horizon_reached','domain_exit']}
    p.update(active_tick=active,physical_tick=clock,task_reward_valid=np.arange(n)<3,
        fallback_active=active & (np.arange(n)>=4),actual_forward_speed=np.ones(n),
        phi=np.zeros(n),peak_roll=np.zeros(n),path_cross_track=np.zeros(n),path_heading_error=np.zeros(n),
        goal_section_signed_distance=np.full(n,-1.),scored_tick_reward=np.full(n,-1.),
        applied_force_impulse=np.zeros((n,3)),time=np.maximum(clock,0)*.005)
    p['horizon_reached'][8]=True;p['domain_exit'][2:]=True
    d={'valid':np.array([True,False,False])}
    case=dict(case_id='example',role='parent',alpha=.5,speed=1.5,force_N=0.)
    return case,p,d


def test_domain_exit_then_physical_survival_is_not_called_task_recovery():
    case,p,d=episode(); row=summarize_case(case,p,d,horizon_s=.04,dt_s=.005,policy_id='frozen')
    assert row['outcome']=='survived' and row['domain_exit']
    assert row['physical_ticks']==8 and row['task_valid_ticks']==3 and row['hold_ticks']==5
    assert row['task_scored_return']==-3.
    assert not row['goal_section_observed_passed']
    assert row['nonzero_pulse_fully_delivered'] is None
    assert 'task_success' not in row


def test_contact_at_window_boundary_wins_and_fault_remains_unknown():
    case,p,d=episode();p['contact_failure'][8]=True
    assert summarize_case(case,p,d,horizon_s=.04,dt_s=.005,policy_id='frozen')['outcome']=='contact'
    p['lower_fault'][8]=True
    assert summarize_case(case,p,d,horizon_s=.04,dt_s=.005,policy_id='frozen')['outcome']=='unknown'


def test_incomplete_observation_is_pending_not_failure():
    case,p,d=episode();p['horizon_reached'][:]=False
    assert summarize_case(case,p,d,horizon_s=.04,dt_s=.005,policy_id='frozen')['outcome']=='pending'


def test_rescored_components_support_same_user_alpha_baseline_plots():
    cfg=reward_config();keys=configured_cost_keys(cfg)
    kw=dict(chi=j.float32(.8),ev=j.float32(-.3),ey=j.float32(.2),eh=j.float32(.05),
        peak_roll=j.float32(.2),roll_rate=j.float32(.3),speed=j.float32(1.2),offsets=j.zeros(2),offset_rate=j.zeros(2))
    actual=costs(alpha=0.,config=cfg,**kw);target=costs(alpha=1.,config=cfg,**kw)
    p=dict(active_tick=np.array([True,True]),task_reward_valid=np.array([True,False]),chi=np.array([.8,np.nan]),
        actual_forward_speed=np.array([1.2,1.2]),requested_speed=np.array([1.5,1.5]),
        path_cross_track=np.array([.2,np.nan]),path_heading_error=np.array([.05,np.nan]),
        effective_costs=np.array([[float(actual['effective'][k]) for k in keys],[np.nan]*len(keys)]),
        scored_tick_reward=np.array([float(actual['reward'])-37.,0.]))
    got=rescore_v3(p,user_alpha=1.,config=cfg,include_components=True)
    np.testing.assert_allclose(got['per_lower_tick_effective_costs'][0],
        [float(target['effective'][k]) for k in keys],rtol=1e-5,atol=1e-5)
    assert np.isnan(got['per_lower_tick_effective_costs'][1]).all()
    assert got['per_lower_tick_reward'][0]==pytest.approx(float(target['reward'])-37.,abs=1e-5)
