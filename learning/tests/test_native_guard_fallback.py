"""Opt-in reuse of the existing guard decision; no plant or training."""
import json
from pathlib import Path
import jax
import jax.numpy as j
import pytest
from sttw_control.native_goal_run import five_alpha_online_inventory
from sttw_control.tolerance_coordination import coordinate,choose_alpha,apply_alpha_dwell


def read(name):return json.loads((Path(__file__).parents[1]/'configs'/name).read_text())


@pytest.mark.parametrize('base,guard',[
    ('h75_five_alpha_online_nominal_users.json','h75_five_alpha_online_nominal_users_guard.json'),
    ('h75_five_alpha_online_hold.json','h75_five_alpha_online_hold_guard.json')])
def test_guard_configs_only_change_declared_mechanism_identity_and_explanation(base,guard):
    before=read(base);after=read(guard)
    assert before['guard_fallback'] is False and after['guard_fallback'] is True
    assert after['task_aware_fallback'] is False
    assert {k for k in before if before[k]!=after[k]}=={'experiment_id','guard_fallback','stop_rule'}
    assert five_alpha_online_inventory(before)==five_alpha_online_inventory(after)
    for bad in [1,0,'true',None]:
        after['guard_fallback']=bad
        with pytest.raises(ValueError,match='boolean'):five_alpha_online_inventory(after)


@pytest.mark.parametrize('user',[0.,.25,.5,.75,1.])
def test_coordinate_guard_keeps_feasible_user_but_remains_degraded_and_latched(user):
    params={k:j.zeros(shape) for k,shape in dict(w0=(1,3,41,16),b0=(1,3,16),w1=(1,3,16,16),b1=(1,3,16),w2=(1,3,16,1),b2=(1,3,1)).items()}
    interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2,speed_reference_bounds_m_s=[1.,3.5],steer_reference_bound_rad=.65)
    f=jax.jit(lambda permission:coordinate(params,j.zeros(351),j.zeros((5,2)),0.,j.array([2.6,0.]),2.6,0.,j.array([0.,10.]),j.zeros(2),permission,True,.75,interface,arm=0,user_alpha=user,candidate_alphas=(0.,.25,.5,.75,1.),hypothetical_duration_s=.15,guard_fallback=True,task_aware_fallback=False))
    out=f(True)
    assert out['alpha']==user and out['degraded'] and out['latched'] and out['guard_fallback_selected']
    denied=f(False)
    assert denied['alpha']==.75 and denied['latched'] and not denied['guard_fallback_selected']


def test_guard_infeasible_user_chooses_feasible_candidate_and_no_task_still_half():
    targets=j.tile(j.array([[2.6,0.]]),(5,1)).at[1,0].set(1.5)
    decide=lambda t:choose_alpha(t,1.,j.array([2.6,0.]),2.6,j.zeros(5),True,False,1.,user_alpha=1.,candidate_alphas=(0.,.25,.5,.75,1.),guard_fallback=True)
    r=decide(targets)
    assert r['alpha']==.25 and r['guard_fallback_selected'] and r['degraded'] and r['latched']
    r=decide(targets.at[1,0].set(2.6))
    assert r['alpha']==.5 and not r['guard_fallback_selected'] and r['degraded']


def test_guard_request_cannot_bypass_active_feasible_hold_or_publication_permission():
    grid=(0.,.25,.5,.75,1.);targets=j.tile(j.array([[2.6,0.]]),(5,1))
    raw=choose_alpha(targets,0.,j.array([2.6,0.]),2.6,j.zeros(5),True,True,.5,user_alpha=0.,candidate_alphas=grid,guard_fallback=True)
    assert raw['alpha']==0 and raw['latched'] and raw['guard_fallback_selected']
    held=apply_alpha_dwell(raw['alpha'],raw['latched'],raw['task_mask'],j.array(True),j.array(.5),j.array(True),j.array(40),j.array(4))
    assert held['alpha']==.5 and held['held'] and held['latched']
    released=apply_alpha_dwell(raw['alpha'],raw['latched'],raw['task_mask'],j.array(True),j.array(.5),j.array(True),j.array(40),j.array(40))
    assert released['alpha']==0 and released['latched'] and not released['held']
