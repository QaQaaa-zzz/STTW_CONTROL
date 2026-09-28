from dataclasses import replace

import numpy as np
import pytest

from sttw_control.actuator import ActuatorConfig, effective_base
from sttw_control.tracking_reward import (
    TrackingConfig,
    directional_speed_path_rates,
    initial_return,
    transition,
    within_final,
)


def config(**changes):
    base = TrackingConfig(
        objective="asymmetric_geometric_huber",
        reward_mode="huber",
        geometric=True,
        return_bonus=0.0,
        tail_rate=0.0,
        return_rate=0.0,
        tracking_rate=4.0,
        budget_rate=2.0,
        priority_ratio=10.0,
        speed_scale=0.1,
        lateral_scale=0.1,
        heading_scale=0.15,
        heading_tail_weight=0.3,
        speed_relaxed=0.5,
        speed_tight=0.2,
        overspeed_band=0.05,
        final_speed_tolerance=0.2,
        final_overspeed_tolerance=0.05,
        shrink_tolerances=True,
        return_seconds=3.0,
        hold_seconds=0.5,
    )
    return replace(base, **changes)


def test_alpha_zero_allows_underspeed_but_never_discounts_overspeed():
    c = config()
    under = directional_speed_path_rates(-0.2, 0.0, 0.0, 0.0, 0.0, c, xp=np)
    over = directional_speed_path_rates(+0.2, 0.0, 0.0, 0.0, 0.0, c, xp=np)
    assert -under["underspeed_tracking"] < -over["overspeed_tracking"]
    for alpha in (0.0, 0.5, 1.0):
        rates = directional_speed_path_rates(+0.2, 0.0, 0.0, alpha, 0.0, c, xp=np)
        assert rates["overspeed_tracking"] == pytest.approx(over["overspeed_tracking"])


def test_synthetic_candidates_select_slow_path_at_alpha_zero_and_speed_at_one():
    c=config(priority_ratio=34.2)
    def cost(ev, lateral, alpha):
        return -sum(directional_speed_path_rates(ev,lateral,0.,alpha,0.,c,xp=np).values())
    slow_path=[cost(-.4,0.,a) for a in (0.,1.)]
    overspeed_path=[cost(+.1,0.,a) for a in (0.,1.)]
    speed_detour=[cost(0.,.2,a) for a in (0.,1.)]
    ideal=[cost(0.,0.,a) for a in (0.,1.)]
    assert slow_path[0] < overspeed_path[0]
    assert slow_path[0] < speed_detour[0]
    assert speed_detour[1] < slow_path[1]
    assert ideal == [0.,0.]


def test_return_tightening_and_final_speed_band_are_asymmetric():
    c = config()
    early = directional_speed_path_rates(-0.4, 0.0, 0.0, 0.0, 0.0, c, xp=np)
    late = directional_speed_path_rates(-0.4, 0.0, 0.0, 0.0, 2.5, c, xp=np)
    assert early["underspeed_budget"] == pytest.approx(0.0)
    assert late["underspeed_budget"] < 0.0
    assert within_final(0.0, 0.0, -0.2, 0.0, 0.0, c, xp=np)
    assert within_final(0.0, 0.0, +0.05, 0.0, 0.0, c, xp=np)
    assert not within_final(0.0, 0.0, +0.051, 0.0, 0.0, c, xp=np)


def test_published_recovery_trigger_starts_one_nonresetting_debt_clock():
    c = config(start_seconds=0.0)
    state = initial_return(xp=np)
    common = dict(
        roll=0.0, roll_rate=0.0, speed_error=0.0, lateral_error=0.0,
        heading_error=0.0, action=np.zeros(2), alpha=0.5, dt=0.1,
        alive_rate=1.0, failure_penalty=100.0, failed=False, enabled=True,
        config=c, clock_from_departure=False, xp=np,
    )
    state, _ = transition(state, recovery_trigger=True, **common)
    assert state.pending and state.elapsed == pytest.approx(0.1)
    state, _ = transition(state, recovery_trigger=False, **common)
    assert state.elapsed == pytest.approx(0.2)
    state, _ = transition(state, recovery_trigger=True, **common)
    assert state.elapsed == pytest.approx(0.3)


def test_asymmetric_objective_cannot_be_overwritten_by_symmetric_reward_mode():
    c=config()
    _,parts=transition(initial_return(xp=np),roll=0.,roll_rate=0.,speed_error=-.2,
        lateral_error=0.,heading_error=0.,action=np.zeros(2),alpha=0.,dt=.005,
        alive_rate=1.,failure_penalty=100.,failed=False,enabled=False,config=c,xp=np)
    assert parts['speed_tracking']==0.
    assert parts['speed_budget']==0.
    assert parts['underspeed_tracking']<0.
    assert parts['overspeed_tracking']==0.


def test_base_output_scale_is_legacy_compatible_and_scales_both_channels():
    raw = np.array([1.5, 23.0])
    np.testing.assert_allclose(effective_base(raw, ActuatorConfig()), raw)
    np.testing.assert_allclose(
        effective_base(raw, ActuatorConfig(base_output_scale=0.8)),
        np.array([1.2, 18.4]),
    )


def test_new_endpoint_configs_match_attachment_candidate_costs():
    from pathlib import Path
    from sttw_control.env import load_config
    root=Path(__file__).resolve().parents[1]/'configs'
    expected={0:[3.4545454545,37.6,12.9090909091],1:[31.5454545455,37.4690909091,1.0909090909]}
    for alpha in [0,1]:
        cfg=load_config(root/f'fixed_endpoint_directional_alpha{alpha}.json')
        c=cfg.tracking
        costs=[-sum(directional_speed_path_rates(ev,ey,0.,alpha,0.,c,xp=np).values()) for ev,ey in [(-.4,.05),(.4,.02),(0.,.2)]]
        np.testing.assert_allclose(costs,expected[alpha],rtol=1e-8)
        assert c.shrink_tolerances and c.final_speed_tolerance==.2
        assert cfg.priority.fixed_alpha==alpha and not cfg.observation.include_priority
