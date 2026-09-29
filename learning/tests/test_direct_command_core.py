"""Pure V3 contract checks. No simulator or training is started by this file."""
import json
from pathlib import Path

import jax.numpy as jp
import numpy as np

from sttw_control.controller import ControllerConfig
from sttw_control.direct_command_policy import (
    initial_correction, map_latent, correction_tick, raw_context,
    initial_history, push_history, assemble_observation, make_frame)
from sttw_control.direct_command_reward import interval_cost, failure_reward, COMPONENTS


SPEC = json.loads((Path(__file__).parents[1]/'configs'/'STTW_Direct_Command_V3.json').read_text())
CC = ControllerConfig(fixed_roll_reference=None)


def test_residual_clip_flag_reports_permission_limit_not_float_roundtrip(monkeypatch):
    from types import SimpleNamespace
    from sttw_control import closed_loop_kernel as kernel
    from sttw_control.actuator import ActuatorConfig,initial_actuator
    monkeypatch.setattr(kernel,'controller_step',lambda state,row,*args:(
        state,SimpleNamespace(steer_rate=row[-1])))
    ac=ActuatorConfig(steer_residual_scale=1.5,rear_residual_scale=10.)
    for requested,expected in [(.1,False),(.07,False),(1.6,True)]:
        _,_,log=kernel.controls(None,initial_actuator(ac),jp.zeros(7),
            jp.array([2.3,0.]),jp.array([2.3,requested]),True,False,CC,ac)
        assert bool(log['residual_clipped']) is expected


def test_zero_latent_preserves_raw_reference_and_asymmetric_map():
    for speed, steer in [(2., -.3), (2.3, 0.), (2.6, .3)]:
        raw = jp.array([speed, steer])
        offsets, governed, flags = correction_tick(initial_correction(), jp.zeros(2), raw, SPEC)
        np.testing.assert_allclose(governed, raw, rtol=0, atol=0)
        np.testing.assert_allclose(offsets, 0, rtol=0, atol=0)
        assert not bool(flags['reference_clipped'])
    np.testing.assert_allclose(map_latent(jp.array([-100., 100.]), SPEC), [-1., .2], atol=1e-6)
    np.testing.assert_allclose(map_latent(jp.array([100., -100.]), SPEC), [.25, -.2], atol=1e-6)


def test_correction_is_rate_limited_and_back_calculated():
    old = jp.array([.249, .199])
    new, governed, flags = correction_tick(old, jp.array([100., 100.]), jp.array([2.9, .3]), SPEC)
    np.testing.assert_allclose(governed, [3., .35], atol=1e-6)
    np.testing.assert_allclose(new, [.1, .05], atol=1e-6)
    assert bool(flags['reference_clipped'])
    new, governed, flags = correction_tick(jp.zeros(2), jp.array([-100., -100.]), jp.array([2.3, 0.]), SPEC)
    np.testing.assert_allclose(new, [-.005, -.004], atol=1e-6)
    assert bool(flags['rate_clipped'])


def test_recovery_gate_drops_on_new_ineligible_command():
    chi, g, eligible, clock = raw_context(jp.array([2.3, 0.]), jp.zeros(2), .8, CC, SPEC)
    assert bool(eligible) and float(g) == 1. and np.isclose(float(clock), .8) and float(chi) == 0.
    chi, g, eligible, clock = raw_context(jp.array([2.6, .25]), jp.zeros(2), .8, CC, SPEC)
    assert not bool(eligible) and float(g) == 0. and float(clock) == 0. and float(chi) == 1.
    _, g, _, clock = raw_context(jp.array([2.3, 0.]), jp.zeros(2), .4, CC, SPEC)
    assert float(g) == 0. and np.isclose(float(clock), .405)


def test_history_order_context_and_nonfinite_fault():
    h = initial_history(SPEC)
    f = make_frame(measurement=jp.array([.03, .1, .02, .2, .3, 23., 22.]),
        forward_speed=2.25, previous_governed=jp.array([2.3, 0.]),
        previous_final_command=jp.array([.2, 23.]),
        previous_bounded_residual=jp.array([.1, -.2]), raw=jp.array([2.3, 0.]),
        raw_rates=jp.zeros(2), eso_equilibrium_shift=.01, cc=CC)
    h = push_history(h, f)
    obs, fraction, fault = assemble_observation(h, 1., .2, 0., 0., .4, jp.zeros(2), SPEC)
    assert obs.shape == (345,) and np.isclose(float(obs[19*0+300]), .1)
    assert float(h.mask[-1]) == 1. and float(h.mask[-2]) == 0.
    assert float(obs[336]) == 1. and not bool(fault) and float(fraction) == 0.
    bad = push_history(h, f.at[0].set(jp.nan))
    _, _, fault = assemble_observation(bad, 1., .2, 0., 0., .4, jp.zeros(2), SPEC)
    assert bool(fault)


def test_reward_components_reconstruct_capped_cost_and_failure_tail():
    d = interval_cost(alpha=0., chi=1., g=0., raw=jp.array([2.6, .3]),
        actual_speed=2.2, actual_steer=.31, heading_error=.5, roll=.1,
        roll_rate=.1, executed_offsets=jp.array([-.4, .01]),
        final_command=jp.array([.5, 22.]), previous_final_command=jp.array([.4, 23.]), spec=SPEC)
    assert set(d['raw_components']) == set(COMPONENTS)
    assert np.isclose(float(d['raw_cost']), sum(float(v) for v in d['raw_components'].values()))
    assert np.isclose(float(d['effective_cost']), sum(float(v) for v in d['effective_components'].values()))
    assert np.isclose(float(d['reward']), -.1*.005*float(d['effective_cost']))
    assert -65.7 < float(failure_reward(800, SPEC)) < -65.5
    assert np.isclose(float(failure_reward(1, SPEC)), -5.2, atol=1e-6)


def test_frozen_lower_composition_shares_preview_and_total_authority(monkeypatch):
    from types import SimpleNamespace
    from sttw_control import closed_loop_kernel as kernel
    from sttw_control.actuator import ActuatorConfig,initial_actuator
    calls=[]
    def controller(state,row,*args):
        calls.append(row)
        return state+1,SimpleNamespace(steer_rate=row[-1])
    monkeypatch.setattr(kernel,'controller_step',controller)
    ac=ActuatorConfig(steer_residual_scale=1.5,rear_residual_scale=10.)
    raw=jp.array([2.3,0.]);goal=jp.array([2.55,.2]);m=jp.zeros(7)
    preview=kernel.preview_controls(0,m,raw,goal,True,CC)
    cs,_,log=kernel.controls(0,initial_actuator(ac),m,raw,goal,True,False,CC,ac,
        lower_action=jp.array([1.,1.]),previewed=preview)
    assert cs==1 and len(calls)==2
    np.testing.assert_allclose(log['applied_residual'],[1.5,10.])
    assert bool(log['residual_clipped'])
    _,_,zero=kernel.controls(0,initial_actuator(ac),m,raw,raw,True,True,CC,ac,
        lower_action=jp.array([.2,.4]))
    np.testing.assert_allclose(zero['requested_residual'],[.3,4.],atol=1e-7)
