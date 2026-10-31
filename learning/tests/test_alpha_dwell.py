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


def test_support_commitment_preserves_user_dwell_and_feasibility_interrupt():
    def call(t,proposal,prev,deadline,mask=j.ones(5,bool)):
        return apply_alpha_dwell(j.array(proposal),j.array(True),mask,j.array(True),j.array(prev),j.array(True),j.array(deadline),j.array(t),support_commitment_ticks=750,user_alpha=0.)
    r=call(284,1.,0.,0)
    assert r['hold_until_tick']==1034
    r=call(324,.5,1.,1034)
    assert r['held'] and r['alpha']==1. and r['hold_until_tick']==1034
    r=call(364,0.,1.,1034)
    assert r['held'] and r['alpha']==1.
    r=call(1036,0.,1.,1034)
    assert not r['held'] and r['alpha']==0. and r['hold_until_tick']==1076
    r=call(324,0.,1.,1034,j.ones(5,bool).at[4].set(False))
    assert r['release_reason']==2 and r['alpha']==0. and r['hold_until_tick']==364


def test_commitment_protocol_is_opt_in_and_single_declared_window():
    p=json.loads((Path(__file__).parents[1]/'configs/u170_lower200_commitment_online.json').read_text())
    assert len(five_alpha_online_inventory(p))==6
    for value in (True,1.,float('nan')):
        with pytest.raises(ValueError):five_alpha_online_inventory({**p,'support_commitment_s':value})


def test_recovery_release_requires_preparation_and_qualified_user():
    def call(t,ready=True,proposal=0.,raw_latch=False,permission=True):
        return apply_alpha_dwell(j.array(proposal),j.array(raw_latch),j.ones(5,bool),j.array(permission),j.array(1.),j.array(True),j.array(850),j.array(t),support_commitment_ticks=750,user_alpha=0.,recovery_ready=j.array(ready),recovery_min_ticks=120)
    assert call(216)['held']
    assert call(220,ready=False)['held']
    assert call(220,proposal=.5)['held']
    assert call(220,raw_latch=True)['held']
    assert call(220,permission=False)['alpha']==1.
    r=call(220)
    assert r['alpha']==0. and r['release_reason']==4 and r['hold_until_tick']==260


def test_recovery_settling_is_elapsed_time_and_rebind_preserves_state():
    from sttw_control.tolerance_coordination import recovery_settling_start,RecoveryOnlineCarry
    rule=dict(max_abs_roll_rad=.26,max_abs_roll_rate_rad_s=.1)
    start=-1
    for tick in range(100,144,4):
        start=int(recovery_settling_start(start,tick,.08,.05,True,rule))
        assert start==100
        assert (tick-start>=40)==(tick==140)
    assert recovery_settling_start(start,144,.08,.2,True,rule)==-1
    assert recovery_settling_start(start,144,.08,.05,False,rule)==-1
    assert recovery_settling_start(-1,148,.08,.05,True,rule)==148
    c=RecoveryOnlineCarry(None,True,1.,850,100)
    assert c._replace(native='rebound')[1:]==c[1:]
    p=json.loads((Path(__file__).parents[1]/'configs/u170_lower200_recovery_release_online.json').read_text())
    assert len(five_alpha_online_inventory(p))==6
