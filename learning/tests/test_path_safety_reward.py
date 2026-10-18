"""Reward identity, raw-measurement reconstruction and four-tick scoring contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as j
import numpy as np
import pytest
from flax import struct

from sttw_control.path_command_reward import costs, failure_reward
from sttw_control.path_command_reporting import reward_series, reconstruct_costs
from sttw_control.path_command_env import PathCommandEnv

BASE = json.loads((Path(__file__).parents[1] / 'configs/STTW_Path_Feedback_V1.json').read_text())


def safety_config():
    cfg = copy.deepcopy(BASE)
    cfg['path_reward'].update(reward_version='path_safety_preference_v1',
        risk_prediction_s=.10, predictive_roll_weight=20.,
        predictive_roll_start_rad=.28, predictive_roll_scale_rad=.02,
        working_exposure_weight=30., working_exposure_width_rad=.002,
        corridor_weight=40., corridor_band_m=.40, corridor_scale_m=.10)
    return cfg


def sample(cfg, **changes):
    args = dict(alpha=.5, chi=1., ev=.1, ey=.1, eh=.1, peak_roll=.30,
                roll_rate=1.6, speed=1.3, offsets=j.array([.5, .1]),
                offset_rate=j.array([.5, .2]), config=cfg)
    args.update(changes)
    return costs(**args)


def test_historical_numerical_fixture_keeps_ten_costs_and_failure_tail():
    # Detects changes to historical coefficients, normalization, caps or dt.
    result = sample(BASE)
    expected = dict(speed=5.5, path=8.25, primary=2.5, roll=40.,
                    working_roll=0., roll_rate=1., overspeed=.5, low_speed=4.,
                    offset_magnitude=.01, offset_rate=.015)
    assert set(result['raw']) == set(expected)
    for key, value in expected.items():
        assert float(result['raw'][key]) == pytest.approx(value, abs=2e-5)
    assert float(result['reward']) == pytest.approx(-.0308875, abs=2e-8)
    assert float(failure_reward(1, BASE)) == pytest.approx(-15.1604, abs=1e-4)
    assert sum(BASE['path_reward']['component_caps'].values()) == pytest.approx(5080.2)


def test_new_subcosts_merge_before_caps_without_new_primary_keys():
    result = sample(safety_config(), ey=.55, peak_roll=.3015, roll_angle=.29, roll_rate=.4)
    assert float(result['raw']['roll']) == pytest.approx(123., abs=2e-4)
    assert float(result['raw']['working_roll']) == pytest.approx(23.175, abs=5e-4)
    assert float(result['raw']['path']) == pytest.approx(137.75, abs=5e-5)
    assert set(result['raw']) == set(BASE['path_reward']['component_caps'])
    for key, value in dict(predictive_roll=80., working_exposure=22.5, corridor=80.).items():
        assert float(result['diagnostic_subcosts'][key]) == pytest.approx(value, abs=5e-4)
    assert float(result['reward']) == pytest.approx(-.0005 * sum(float(x) for x in result['effective'].values()), abs=1e-7)
    capped = sample(safety_config(), roll_angle=10., peak_roll=10., ey=10.)
    for key in ('roll', 'working_roll', 'path'):
        assert bool(capped['capped'][key])
        assert float(capped['effective'][key]) == BASE['path_reward']['component_caps'][key]


@pytest.mark.parametrize('phi,rate,want',[(.29,.4,80.),(-.29,-.4,80.),(.29,-.4,5.),(-.29,.4,5.),(.1,-4.,20.)])
def test_predictive_roll_mirrors_inward_motion_and_crossing_zero(phi, rate, want):
    # A signed linear prediction must not treat inward motion as outward.
    result = sample(safety_config(), roll_angle=phi, roll_rate=rate)
    assert float(result['diagnostic_subcosts']['predictive_roll']) == pytest.approx(want, abs=1e-4)


def test_new_version_requires_signed_roll_and_rejects_unknown_version():
    with pytest.raises(ValueError, match='roll'):
        sample(safety_config())
    cfg = safety_config(); cfg['path_reward']['reward_version'] = 'typo'
    with pytest.raises(ValueError, match='version'):
        sample(cfg, roll_angle=.1)
    with pytest.raises(ValueError, match='version'):
        reward_series(trace(), 0., cfg)


@pytest.mark.parametrize('version',[None,'legacy',''])
def test_explicit_invalid_versions_fail_closed_in_all_reward_entrypoints(version):
    cfg=copy.deepcopy(BASE);cfg['path_reward']['reward_version']=version
    for operation in (lambda:sample(cfg,roll_angle=.1),lambda:failure_reward(1,cfg),
                      lambda:reward_series(trace(),0.,cfg)):
        with pytest.raises(ValueError,match='version'):
            operation()


def test_numpy_new_reward_requires_raw_signed_roll():
    d=trace();del d['phi']
    with pytest.raises(ValueError,match='roll'):
        reconstruct_costs(d,.5,safety_config())


def test_common_safety_additions_are_alpha_independent_and_caps_bound_reward():
    for alpha in (0.,.5,1.):
        result=sample(safety_config(),alpha=alpha,roll_angle=.29,ey=.55,peak_roll=.3015,roll_rate=.4)
        for key,value in dict(predictive_roll=80.,working_exposure=22.5,corridor=80.).items():
            assert float(result['diagnostic_subcosts'][key])==pytest.approx(value,abs=5e-4)
    saturated=sample(safety_config(),ev=100.,ey=100.,eh=100.,peak_roll=100.,
                     roll_angle=100.,roll_rate=100.,speed=-100.,
                     offsets=j.full(2,100.),offset_rate=j.full(2,100.))
    assert all(bool(v) for v in saturated['capped'].values())
    assert float(saturated['reward'])==pytest.approx(-2.5401,abs=1e-6)


def trace(n=8):
    ticks = np.arange(n)
    offsets = np.column_stack([.005 * (ticks + 1), -.002 * (ticks + 1)])
    return dict(time=ticks*.005, chi=np.linspace(0,1,n),
        actual_forward_speed=np.linspace(1.3,2.,n), v_user=np.full(n,1.8),
        path_cross_track=np.linspace(-.6,.6,n), path_heading_error=np.linspace(-.12,.12,n),
        peak_roll=np.full(n,.3015), phi=np.linspace(-.29,.29,n), phi_dot=np.full(n,.4),
        offsets=offsets, physical_failure=np.zeros(n,bool),
        tracking_domain_failure=np.zeros(n,bool))


def context(d, n=None):
    d = copy.deepcopy(d); ticks = np.rint(d['time']/.005).astype(int)
    starts = ticks//4*4
    d['interval_start_tick'] = starts
    d['interval_active_ticks'] = np.minimum(4, (len(ticks) if n is None else n)-starts)
    d['interval_offset_before'] = np.column_stack([starts*.005, -starts*.002])
    ends = starts+d['interval_active_ticks']
    d['interval_offset_end'] = np.column_stack([ends*.005, -ends*.002])
    d['interval_failed'] = np.zeros(len(ticks),bool)
    return d


@pytest.mark.parametrize('cfg',[BASE,safety_config()])
def test_numpy_rebuilds_all_ten_costs_from_raw_measurements(cfg):
    d = trace(); d.update({'effective_cost_'+k:np.full(8,99999.) for k in BASE['path_reward']['component_caps']})
    result, parts = reward_series(d,.5,cfg)
    rebuilt=reconstruct_costs(d,.5,cfg)
    rate = j.array([1.,-.4])
    for i in range(8):
        row = costs(alpha=.5,chi=d['chi'][i],ev=d['actual_forward_speed'][i]-d['v_user'][i],
            ey=d['path_cross_track'][i],eh=d['path_heading_error'][i],peak_roll=d['peak_roll'][i],
            roll_rate=d['phi_dot'][i],roll_angle=d['phi'][i],speed=d['actual_forward_speed'][i],
            offsets=j.asarray(d['offsets'][i]),offset_rate=rate,config=cfg)
        assert result[i] == pytest.approx(float(row['reward']),abs=2e-6)
        for key in BASE['path_reward']['component_caps']:
            assert parts[key][i] == pytest.approx(-.0005*float(row['effective'][key]),abs=2e-6)
            assert rebuilt['raw'][key][i] == pytest.approx(float(row['raw'][key]),rel=2e-5,abs=5e-4)
    # Unit interval rate cost=.03*(1^2+(-.4/.4)^2)=.06.
    np.testing.assert_allclose(parts['offset_rate'], -.00003)


def test_fragments_require_boundaries_and_preserve_original_block_rates():
    full = context(trace()); expected, _ = reward_series(full,.5,safety_config())
    fragment = {k:v[2:7] for k,v in full.items()}
    actual, _ = reward_series(fragment,.5,safety_config())
    np.testing.assert_allclose(actual,expected[2:7])
    # Neither a nonzero starting tick nor an unfinished live block implies zero.
    for invalid in [ {k:v[4:] for k,v in trace().items()}, {k:v[:6] for k,v in trace().items()} ]:
        with pytest.raises(ValueError, match='context|boundar|partial'):
            reward_series(invalid,.5,safety_config())


def test_explicit_interval_scoring_retains_legacy_domain_engineering_fault():
    # terminal_contract_v1 logs domain diagnostics but does not apply failure tail.
    full=context(trace());expected,_=reward_series(full,.5,BASE)
    full['tracking_domain_failure'][-1]=True
    actual,parts=reward_series(full,.5,BASE)
    np.testing.assert_allclose(actual,expected)
    np.testing.assert_array_equal(parts['failure_tail'],0.)


@pytest.mark.parametrize('terminal_ticks',[1,2,3,4])
@pytest.mark.parametrize('kind',['physical_failure','tracking_domain_failure'])
def test_partial_failure_block_replaces_ordinary_rewards_once(terminal_ticks,kind):
    d = trace(4+terminal_ticks); d[kind][-1] = True
    rebuilt=reconstruct_costs(d,.5,safety_config())
    np.testing.assert_allclose(rebuilt['offset_rate'][4:],np.tile([terminal_ticks*.25,-terminal_ticks*.1],(terminal_ticks,1)))
    scored, parts = reward_series(d,.5,safety_config(),horizon=20.)
    np.testing.assert_array_equal(scored[4:-1],0.)
    np.testing.assert_array_equal(parts['offset_rate'][4:],0.)
    assert scored[-1] == pytest.approx(float(failure_reward(999,BASE)),rel=2e-6)
    assert np.count_nonzero(parts['failure_tail']) == 1
    full = context(d); full['interval_failed'][4:] = True
    # A slice before the final failure frame still knows the block is replaced.
    fragment = {k:v[4:-1] for k,v in full.items()}
    if terminal_ticks>1:
        fragment_scored,_ = reward_series(fragment,.5,safety_config())
        np.testing.assert_array_equal(fragment_scored,0.)


@struct.dataclass
class Physical:
    failed: object

@struct.dataclass
class State:
    physical: object
    fault: object
    domain_exit: object
    offsets: object
    tick: object


@pytest.mark.parametrize('terminal_ticks',[1,2,3,4])
def test_policy_step_logs_actual_partial_block_rate_and_tail(terminal_ticks):
    env = PathCommandEnv.__new__(PathCommandEnv)
    env.config=BASE; env.cc=SimpleNamespace(dt=.005)
    env.terminal_contract='terminal_contract_v2'; env.record_frame=lambda s:s
    env.zero_log=dict(active_tick=j.bool_(False),tick_reward=j.array(0.),
        raw_components={'offset_rate':j.array(0.)},effective_components={'offset_rate':j.array(0.)},
        component_capped={'offset_rate':j.bool_(False)})
    def tick(s,z,path,speed,**kwargs):
        end=s.replace(tick=s.tick+1,offsets=s.offsets+j.array([.005,-.002]),
                      physical=Physical(s.tick+1==terminal_ticks))
        return end,{**env.zero_log,'active_tick':j.bool_(True),'tick_reward':j.array(-.01)}
    env.tick=tick
    start=State(Physical(j.bool_(False)),j.bool_(False),j.bool_(False),j.array([.2,.1]),j.int32(0))
    end,logs=jax.jit(lambda s:env.policy_step(s,j.zeros(2),None,2.,20.))(start)
    np.testing.assert_allclose(logs['offset_rate'],np.tile([terminal_ticks*.25,-terminal_ticks*.1],(4,1)),atol=1e-6)
    np.testing.assert_allclose(logs['interval_offset_before'],np.tile([.2,.1],(4,1)),atol=1e-6)
    np.testing.assert_allclose(logs['interval_offset_end'],np.tile(np.asarray(end.offsets),(4,1)),atol=1e-6)
    np.testing.assert_array_equal(logs['interval_active_ticks'],terminal_ticks)
    assert np.count_nonzero(logs['scored_tick_reward'])==1
    assert float(logs['scored_tick_reward'].sum()) == pytest.approx(float(failure_reward(1000,BASE)),rel=2e-6)
