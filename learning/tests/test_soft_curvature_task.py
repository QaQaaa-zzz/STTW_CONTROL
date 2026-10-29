"""Opt-in task ranking contract, without physical integration or training."""
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
import pytest
from sttw_control.tolerance_coordination import choose_alpha,coordinate,apply_alpha_dwell
from sttw_control.native_goal_run import five_alpha_online_inventory
GRID=(0.,.25,.5,.75,1.)


def choose(mode='hard',targets=None,q=None,user=.5,guard=True,permission=True,**kw):
    return choose_alpha(j.tile(j.array([[2.6,0.]]),(5,1)) if targets is None else targets,.6,j.array([2.6,0.]),2.6,j.zeros(5) if q is None else q,permission,True,.75,user_alpha=user,candidate_alphas=GRID,guard_fallback=guard,curvature_task_mode=mode,**kw)


def test_hard_default_bit_identity_and_soft_penalty_explicit():
    targets=j.array([[2.,0.],[2.3,.1],[2.6,0.],[2.7,-.1],[2.8,0.]])
    a=choose_alpha(targets,.6,j.array([2.6,0.]),2.6,j.ones(5),True,True,.75,candidate_alphas=GRID)
    b=choose_alpha(targets,.6,j.array([2.6,0.]),2.6,j.ones(5),True,True,.75,candidate_alphas=GRID,curvature_task_mode='hard')
    assert set(a)==set(b)
    for k in a:np.testing.assert_array_equal(a[k],b[k])
    soft=choose('soft_penalty',targets=targets)
    expected=np.maximum(np.asarray(targets[:,0])**2*.6/3-1,0)**2
    np.testing.assert_allclose(soft['curvature_excess_penalty'],expected,atol=1e-7)
    np.testing.assert_allclose(soft['proxy_cost'],soft['original_proxy_cost']+expected,atol=1e-7)
    assert soft['curvature_excess_penalty'][0]==0 and soft['task_mask'].all()


def test_user_priority_and_degraded_guard_retention_ignore_ranking_penalty():
    assert not choose()['task_mask'].any()
    qualified=choose('soft_penalty',q=j.ones(5),user=1.)
    assert qualified['alpha']==1. and not qualified['latched']
    degraded=choose('soft_penalty',user=0.)
    assert degraded['alpha']==0. and degraded['degraded'] and degraded['latched']
    held=apply_alpha_dwell(degraded['alpha'],degraded['latched'],degraded['task_mask'],j.array(True),j.array(.5),j.array(True),j.array(40),j.array(4))
    assert held['held'] and held['alpha']==.5 and not degraded['historical_hard_task_mask'].any()
    denied=choose('soft_penalty',permission=False)
    assert denied['alpha']==.75 and denied['latched']


def test_invalid_candidates_are_masked_and_invalid_fallback_is_explicit_fault():
    targets=j.tile(j.array([[2.6,0.]]),(5,1)).at[2,0].set(j.inf)
    r=choose('soft_penalty',targets=targets,user=.5)
    assert not r['valid_target_mask'][2] and not r['candidate_reference_fault'] and r['selected_reference_valid']
    # With guard disabled and no q-qualified candidate, midpoint fallback is
    # invalid: keep previous execution/latch and explicitly withhold/fault.
    r=choose('soft_penalty',targets=targets,guard=False)
    assert r['candidate_reference_fault'] and r['publication_withheld'] and r['alpha']==.75 and r['latched']
    r=choose('soft_penalty',targets=j.full((5,2),j.nan))
    assert not r['task_mask'].any() and r['candidate_reference_fault'] and r['publication_withheld']


def test_coordinate_rejects_infinite_original_latent_even_if_tanh_clips_to_finite():
    params={k:j.zeros(shape) for k,shape in dict(w0=(1,3,41,16),b0=(1,3,16),w1=(1,3,16,16),b1=(1,3,16),w2=(1,3,16,1),b2=(1,3,1)).items()}
    interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.,steer_scale_rad=.2,speed_reference_bounds_m_s=[1.,3.5],steer_reference_bound_rad=.65)
    f=jax.jit(lambda z:coordinate(params,j.zeros(351),z,0.,j.array([2.6,0.]),2.6,0.,j.array([0.,10.]),j.ones(2)*.6,True,True,.75,interface,arm=0,user_alpha=.5,candidate_alphas=GRID,hypothetical_duration_s=.15,guard_fallback=True,curvature_task_mode='soft_penalty'))
    r=f(j.zeros((5,2)).at[2,0].set(j.inf))
    assert np.isfinite(r['targets']).all() and not r['valid_target_mask'][2]
    assert not r['candidate_reference_fault']
    r=f(j.full((5,2),j.inf))
    assert np.isfinite(r['targets']).all() and not r['valid_target_mask'].any() and r['candidate_reference_fault']


def test_soft_config_changes_only_explicit_mode_identity_and_explanation():
    directory=Path(__file__).parents[1]/'configs'
    old=json.loads((directory/'h75_five_alpha_online_hold_guard.json').read_text());new=json.loads((directory/'h75_five_alpha_online_soft_task.json').read_text())
    assert new['curvature_task_mode']=='soft_penalty' and 'curvature_task_mode' not in old
    assert {k for k in old if old[k]!=new[k]}=={'experiment_id','stop_rule'}
    assert five_alpha_online_inventory(old)==five_alpha_online_inventory(new)
    new['curvature_task_mode']='relaxed'
    with pytest.raises(ValueError):five_alpha_online_inventory(new)
