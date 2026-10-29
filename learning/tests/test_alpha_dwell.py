"""Causal dwell state machine, no plant/model fitting or rollout."""
import json
from pathlib import Path
import jax
import jax.numpy as j
import pytest
from sttw_control.tolerance_coordination import apply_alpha_dwell,DwellOnlineCarry,OnlineCarry
from sttw_control.native_goal_run import five_alpha_online_inventory


def step(t,proposal,prev=.25,latch=True,deadline=40,permission=True,feasible=True,raw_latch=False):
    mask=j.ones(5,bool).at[1].set(feasible)
    return apply_alpha_dwell(j.array(proposal),j.array(raw_latch),mask,j.array(permission),j.array(prev),j.array(latch),j.array(deadline),j.array(t))


def test_no_initial_hold_and_every_real_change_including_fallback_starts_dwell():
    r=step(0,.25,prev=.5,latch=False,deadline=0,raw_latch=True)
    assert r['alpha']==.25 and r['hold_until_tick']==40 and r['alpha_changed'] and not r['held']
    # After expiry, the original chooser's fallback .5 is held like any change.
    r=step(40,.5,raw_latch=True)
    assert r['alpha']==.5 and r['latched'] and r['hold_until_tick']==80
    assert r['release_reason']==1


def test_q_or_raw_latch_change_does_not_interrupt_hold_and_exact_expiry_releases():
    for t in range(4,40,4):
        r=step(t,.5)
        assert r['alpha']==.25 and r['latched'] and r['held'] and r['hold_until_tick']==40
    r=step(40,.5)
    assert r['alpha']==.5 and not r['latched'] and not r['held'] and r['hold_until_tick']==80


def test_unchanged_proposal_never_restarts_and_expiry_report_is_once():
    r=step(20,.25)
    assert r['hold_until_tick']==40 and not r['alpha_changed']
    r=step(40,.25,raw_latch=True)
    assert r['hold_until_tick']==0 and r['release_reason']==1
    r=step(44,.25,deadline=0,raw_latch=True)
    assert r['hold_until_tick']==0 and r['release_reason']==0


def test_task_mask_infeasible_bypasses_immediately_but_denied_permission_has_priority():
    r=step(8,.75,feasible=False)
    assert r['alpha']==.75 and r['hold_until_tick']==48 and r['release_reason']==2
    r=step(8,.75,feasible=False,permission=False)
    assert r['alpha']==.25 and r['latched'] and r['hold_until_tick']==40 and r['release_reason']==3
    r=step(44,.75,permission=False)
    assert r['alpha']==.25 and r['remaining_ticks']==0 and r['hold_until_tick']==40
    r=step(48,.75)
    assert r['alpha']==.75 and r['hold_until_tick']==88


def test_rebinding_preserves_optional_sidecar_and_old_interface_is_unchanged():
    carry=DwellOnlineCarry(object(),True,.25,100)
    new=carry._replace(native='new geometry only')
    assert new[1:]==carry[1:] and len(OnlineCarry(object(),False,.5))==3
    old=json.loads((Path(__file__).parents[1]/'configs/h75_five_alpha_online.json').read_text())
    hold=json.loads((Path(__file__).parents[1]/'configs/h75_five_alpha_online_hold.json').read_text())
    assert 'alpha_dwell_s' not in old and hold['alpha_dwell_s']==.2
    assert five_alpha_online_inventory(old)==five_alpha_online_inventory(hold)


def test_dwell_jit_vmap_and_static_tick_validation():
    f=jax.jit(jax.vmap(lambda t:step(t,.5)))
    r=f(j.array([0,36,40,44]))
    assert r['alpha'].tolist()==[.25,.25,.5,.5]
    with pytest.raises(ValueError):apply_alpha_dwell(.5,False,j.ones(5,bool),True,.25,True,40,0,dwell_ticks=0)
