"""Compare production V5 reward with the supplied independent numerical reference."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import jax.numpy as jp
from sttw_control.controller import ControllerConfig
from sttw_control.smooth_command_config import resolve
from sttw_control.direct_command_reward import interval_cost, upper_motion_cost, failure_reward

p = Path(__file__).resolve().parents[2]/'docs/preference_v5/attachment/reference_math.py'
m = importlib.util.spec_from_file_location('preference_v5_reference_math', p)
ref = importlib.util.module_from_spec(m); sys.modules[m.name] = ref; m.loader.exec_module(ref)

@pytest.mark.parametrize('alpha', [0,1])
@pytest.mark.parametrize('stage', [1,2])
def test_resolved_runtime_contract(alpha,stage):
    s=resolve(alpha,preference_v5=True,stage=stage)
    v=s['preference_v5']; p=s['ppo']
    assert v['upper_alpha']==s['upper_alpha']==alpha
    assert s['lower_controller']['alpha']==1 and s['lower_reference_centered']
    assert s['training_stage']==stage
    assert s['critic_horizon_normalizer_s']==16.0
    assert s['episode_duration_s']==(5 if stage==1 else 16)
    assert s['commands']['episode_seconds']==(5 if stage==1 else 16)
    assert p['seed']==83 and p['actor_learning_rate']==1e-4
    assert p['latent_std_min']==[.08,.03] and p['latent_std_max']==[.4,.2]
    assert p['initial_latent_std']==[.3,.1] and p['value_only_initial_rollouts']==0
    assert p['default_updates']==(40 if stage==1 else 120)
    assert p['stage_additional_updates']==(40 if stage==1 else 80)
    assert not s['budget']['wall_limits_enabled']
    assert isinstance(s['budget']['training_wall_seconds'], (int,float))
    assert set(s['smooth_v4'])=={'training_commands'}
    assert [s['commands']['speed_min_m_s'],s['commands']['speed_max_m_s']]==[2.,2.6]
    assert sum(s['reward']['independent_component_caps'].values())==1680
    old=resolve(alpha)
    assert s['action']==old['action'] and s['limits']==old['limits'] and s['plant']==old['plant']

@pytest.mark.parametrize('alpha',[0,1])
@pytest.mark.parametrize('chi,g,debt',[(0,0,0),(1,0,.13),(.4,.5,.03),(0,1,.2)])
@pytest.mark.parametrize('roll',[.33,.345,.4,.69])
def test_physical_cost_matches_reference(alpha,chi,g,debt,roll):
    s=resolve(alpha,preference_v5=True); raw=jp.array([2.6,.25]); gov=jp.array([2.19,.17])
    rc=2.6*np.cos(np.deg2rad(25)) * np.tan(.25)/.408
    expected=ref.costs(alpha,np.array(raw),np.array(gov),2.3,.2,roll,.9,debt,.4,rc,chi,g)
    got=interval_cost(alpha=alpha,chi=chi,g=g,raw=raw,actual_speed=2.3,actual_steer=.2,
        heading_error=debt,roll=roll,roll_rate=.9,executed_offsets=gov-raw,
        final_command=jp.array([3.,-60.]),previous_final_command=jp.zeros(2),spec=s,
        yaw_rate=.4,cc=ControllerConfig(fixed_roll_reference=None))
    for k,val in got['raw_components'].items():
        np.testing.assert_allclose(val,expected[k],rtol=2e-5,atol=2e-5,err_msg=k)
        np.testing.assert_allclose(got['effective_components'][k],min(expected[k],s['reward']['independent_component_caps'][k]),rtol=2e-5,atol=2e-5)
    assert float(got['effective_components']['roll'])>99

@pytest.mark.parametrize('g',[0,1])
def test_deadband_and_priority_gate(g):
    s=resolve(0,preference_v5=True)
    got=interval_cost(alpha=0,chi=0,g=g,raw=jp.array([2.6,.25]),actual_speed=2.62,actual_steer=.252,
        heading_error=0.,roll=0.,roll_rate=0.,executed_offsets=jp.zeros(2),final_command=jp.zeros(2),
        previous_final_command=jp.zeros(2),spec=s,cc=ControllerConfig())
    assert float(got['raw_components']['speed'])==0
    assert float(got['raw_components']['reference_priority'])==0
    if g==0: assert float(got['raw_components']['steer'])==0

@pytest.mark.parametrize('chi,debt',[(0,0),(1,0),(.3,.2)])
def test_motion_rho_and_reset_acceleration_mask(chi,debt):
    s=resolve(0,preference_v5=True)
    rate,acc,raw,eff=upper_motion_cost(jp.array([-.01,.002]),jp.zeros(2),jp.zeros(2),True,s,chi=chi,heading_error=debt)
    expected=ref.costs(0,[2.6,.25],[2.6,.25],2.6,.25,0,0,debt,0,0,chi,0,np.array(rate),np.array(acc))
    for k in raw: np.testing.assert_allclose(raw[k],expected[k],rtol=2e-6)
    assert upper_motion_cost(jp.array([-.01,.002]),jp.zeros(2),jp.zeros(2),False,s,chi=chi,heading_error=debt)[2]['upper_acceleration']==0

@pytest.mark.parametrize('n',[1,250,800])
def test_failure_absorbing_remainder(n):
    s=resolve(1,preference_v5=True)
    np.testing.assert_allclose(failure_reward(n,s),ref.failure_reward(n),rtol=2e-5)

@pytest.mark.parametrize('alpha',[0,1])
@pytest.mark.parametrize('governed',[(2.6,.25),(2.19,.25),(2.6,.17)])
def test_priority_compatibility_and_synthetic_preference(alpha,governed):
    s=resolve(alpha,preference_v5=True); raw=jp.array([2.6,.25]); gov=jp.array(governed)
    expected=ref.preference_and_compatibility(alpha,np.array(raw),np.array(gov))
    got=interval_cost(alpha=alpha,chi=1.,g=0.,raw=raw,actual_speed=gov[0],actual_steer=gov[1],
        heading_error=0.,roll=.26,roll_rate=0.,executed_offsets=gov-raw,final_command=jp.array([3.,-60.]),
        previous_final_command=jp.zeros(2),spec=s,cc=ControllerConfig())
    np.testing.assert_allclose(got['raw_components']['reference_priority'],expected[0],rtol=2e-5,atol=1e-6)
    np.testing.assert_allclose(got['raw_components']['command_compatibility'],expected[1],rtol=2e-5,atol=1e-6)
    if governed==(2.6,.25): assert got['raw_components']['command_compatibility']>0
