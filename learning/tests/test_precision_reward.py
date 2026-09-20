from dataclasses import replace,asdict
import numpy as np
import jax
import jax.numpy as jp
import pytest
from sttw_control.tracking_reward import TrackingConfig,directional_speed_path_rates,directional_tolerances,initial_return,transition,within_final
from sttw_control.network import make_policy_identity

def config():
    return TrackingConfig(objective='asymmetric_geometric_huber',geometric=True,reward_mode='huber',return_bonus=0.,tail_rate=0.,return_rate=0.,priority_ratio=34.2,speed_scale=.1,lateral_scale=.1,heading_scale=.15,precision_reward=True,speed_tight=.05,final_speed_tolerance=.05,final_overspeed_tolerance=.05,overspeed_band=.05,shrink_tolerances=True,deadline_penalty=5.)

def test_numeric_candidates_and_overspeed_symmetry():
    c=config();states=[(-.25,.21),(-.225,.28),(-.06,.45),(-.04,.55)]
    costs=np.array([[-sum(directional_speed_path_rates(v,y,0.,a,2.5,c,xp=np).values()) for v,y in states] for a in [0.,1.]])
    np.testing.assert_allclose(costs,[[21.68,29.02,44.0632,56.0192],[88.72,81.66,24.08,22.12]],atol=.01)
    assert costs[0].argmin()==0 and costs[1].argmin()==3
    for a in [0.,.5,1.]:
        r=directional_speed_path_rates(.2,0.,0.,a,0.,c,xp=np)
        assert r['overspeed_tracking']==pytest.approx(-56.)
    np.testing.assert_allclose(directional_tolerances(.5,0.,c,xp=np),[.275,.05,.25])
    np.testing.assert_allclose(directional_tolerances(.5,2.5,c,xp=np),[.05,.05,.1])
    assert not within_final(0,0,-.051,0,0,c,xp=np)

def step(c,xp=np,**kwargs):
    args=dict(roll=0.,roll_rate=0.,speed_error=-.23,lateral_error=.28,heading_error=0.,action=xp.zeros(2),alpha=1.,dt=.005,alive_rate=1.,failure_penalty=200.,failed=False,enabled=True,config=c,xp=xp)
    args.update(kwargs)
    return transition(initial_return(xp=xp),**args)

def test_bounded_rewards_failure_and_ideal_tracking():
    c=config()
    for error in [0.,.04,.2,2.,1000.]:
        _,p=step(c,speed_error=-error,lateral_error=error)
        assert -.0495001 <= sum(p.values()) <= .0005001
        _,p=step(c,speed_error=error,lateral_error=error,failed=True)
        assert sum(p.values())==-200.
    _,p=step(c,speed_error=0.,lateral_error=0.)
    assert sum(p.values())==pytest.approx(.0005)
    gamma=.9995;n=2000;discount=gamma**np.arange(n)
    survivor_lower=-.0495*discount.sum()-5.
    failure_upper=max(.0005*discount[:t].sum()-200*discount[t] for t in range(n))
    assert survivor_lower>failure_upper

def test_numpy_jax_and_nonresetting_debt():
    c=config()
    for a in [0.,.5,1.]:
        n,p=step(c,alpha=a);j,q=jax.jit(lambda:step(c,jp,alpha=a))()
        np.testing.assert_allclose([p[k] for k in sorted(p)],[q[k] for k in sorted(p)],atol=1e-7)
    n,p=step(c,clock_from_departure=False)
    assert n.elapsed==0
    args=dict(roll=0.,roll_rate=0.,speed_error=-.2,lateral_error=.2,heading_error=0.,action=np.zeros(2),alpha=1.,dt=.005,alive_rate=1.,failure_penalty=200.,failed=False,enabled=True,config=c,xp=np,clock_from_departure=False,recovery_trigger=True)
    n,_=transition(n,**args);n,_=transition(n,**args)
    assert n.elapsed==pytest.approx(.01)

def test_disabled_precision_fields_do_not_change_old_identity():
    c=replace(config(),precision_reward=False)
    expanded={'tracking':asdict(c)}
    legacy={'tracking':{k:v for k,v in expanded['tracking'].items() if not k.startswith('precision_')}}
    assert make_policy_identity({},expanded,10)==make_policy_identity({},legacy,10)

def test_formal_config_budget_and_hold_with_single_deadline():
    import json
    from pathlib import Path
    from sttw_control.env import load_config
    from sttw_control.training import TrainingConfig
    c=load_config('learning/configs/precision_speed_ecbc1.json')
    train=TrainingConfig(**json.loads(Path('learning/configs/ppo_precision_speed.json').read_text()))
    assert c.priority.training_alphas==(0.,.5,1.) and tuple(c.priority.validation_alphas)==(0.,.5,1.)
    assert train.training_reward_best_enabled and train.training_reward_selection
    assert train.num_envs*train.rollout_steps*train.updates==26214400
    assert c.failure_penalty==200. and c.actuator.base_output_scale==1.
    assert (c.actuator.steer_residual_scale,c.actuator.rear_residual_scale)==(1.5,10.)
    tc=config();state=initial_return(xp=np);penalty=0.
    for tick in range(1000):
        state,p=transition(state,roll=0.,roll_rate=0.,speed_error=-.2,lateral_error=.2,heading_error=0.,action=np.zeros(2),alpha=1.,dt=.005,alive_rate=1.,failure_penalty=200.,failed=False,enabled=True,config=tc,xp=np)
        penalty+=p['deadline']
    assert penalty==-5. and state.deadline_missed
    for tick in range(100):
        state,p=transition(state,roll=0.,roll_rate=0.,speed_error=0.,lateral_error=0.,heading_error=0.,action=np.zeros(2),alpha=1.,dt=.005,alive_rate=1.,failure_penalty=200.,failed=False,enabled=True,config=tc,xp=np)
    assert not state.pending and state.hold==pytest.approx(.5) and state.deadline_missed
