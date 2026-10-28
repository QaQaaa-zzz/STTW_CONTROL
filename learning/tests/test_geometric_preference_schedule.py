"""CPU-only rule checks: no plant, checkpoint inference, or physical step."""
import copy
import numpy as np
import pytest
import jax
import jax.numpy as j
from sttw_control.geometric_preference_schedule import (
    advance,cap_speed_latent,preview_max_curvature,validate_rule)
from sttw_control.native_zero_comparison import scheduling_cases

RULE=dict(preview_m=1.5,engage_accel_m_s2=3.,release_accel_m_s2=2.7,
          active_alpha=.5,alpha_rate_per_s=1.,upper_dt_s=.02)
CFG=dict(residual_interface=dict(speed_positive_scale_m_s=.25,speed_negative_scale_m_s=1.))


def test_full_knot_preview_includes_narrow_peak_and_interpolated_endpoints():
    s=j.array([0.,.1,.11,.12,1.,2.,3.]);k=j.array([0.,0.,.8,0.,0.,.4,0.])
    assert float(preview_max_curvature(s,k,0.,1.5))==pytest.approx(.8)
    # No covered interior knot: both interpolated endpoints still participate.
    assert float(preview_max_curvature(s,k,1.2,.1))==pytest.approx(.12)


def test_rule_is_causal_spatial_prefix_invariant_and_mirror_invariant():
    s=j.arange(0.,5.,.1);k=j.where((s>=.5)&(s<1.),.55,0.)
    altered=k.at[s>2.].set(100.)
    a=advance(j.array(1.),j.array(False),2.8,s,k,0.,RULE)
    b=advance(j.array(1.),j.array(False),2.8,s,altered,0.,RULE)
    c=advance(j.array(1.),j.array(False),2.8,s,-k,0.,RULE)
    for x,y,z in zip(jax.tree.leaves(a),jax.tree.leaves(b),jax.tree.leaves(c)):
        np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(x,z)
    # There is no route ID, tick, elapsed-time or future-state input in this API.
    assert a[1]


def decision_for_accel(accel,latched,alpha=1.):
    # Unit original speed makes curvature exactly the acceleration proxy.
    return advance(j.array(alpha),j.array(latched),1.,j.array([0.,1.,2.]),j.full(3,accel),0.,RULE)


def test_strict_hysteresis_boundaries_slew_and_return_to_user():
    assert not decision_for_accel(3.,False)[1]
    assert decision_for_accel(3.01,False)[1]
    assert decision_for_accel(2.7,True)[1]
    assert not decision_for_accel(2.69,True)[1]
    a,l,_=decision_for_accel(3.01,False)
    assert float(a)==pytest.approx(.98)
    for _ in range(40):a,l,_=decision_for_accel(3.01,l,a)
    assert float(a)==pytest.approx(.5)
    for _ in range(40):a,l,_=decision_for_accel(0.,l,a)
    assert float(a)==pytest.approx(1.) and not l


def test_neutral_rule_and_speed_cap_preserve_exact_h75_latent_identity():
    s=j.array([0.,1.,2.]);k=j.zeros(3)
    a,l,d=advance(j.array(1.),j.array(False),2.8,s,k,0.,RULE)
    assert a==1. and not l and d['rule_speed_target']==2.8
    for value in (-8.,-1.,0.,.4,8.):
        z=j.array([value,.731])
        out,_=cap_speed_latent(z,j.array([2.8,.1]),2.,False,CFG)
        np.testing.assert_array_equal(out,z)


def test_cap_is_within_original_permission_preserves_steer_and_stronger_deceleration():
    nominal=j.array([2.8,.1]);z=j.array([.5,-.731])
    out,d=cap_speed_latent(z,nominal,2.3,True,CFG)
    assert float(d['rule_realizable_speed'])==pytest.approx(2.3)
    assert out[1]==z[1] and d['rule_speed_cap_binding']
    slow=j.array([-2.,.731]);out,d=cap_speed_latent(slow,nominal,2.3,True,CFG)
    np.testing.assert_array_equal(out,slow)
    assert not d['rule_speed_cap_binding']
    out,d=cap_speed_latent(z,nominal,.5,True,CFG)
    assert d['rule_speed_authority_limited']
    assert float(d['rule_realizable_speed'])>1.8
    assert float(d['rule_requested_speed'])==pytest.approx(.5)
    assert out[1]==z[1]


def test_rule_compiles_with_state_carry_and_has_no_python_route_dependency():
    def tick(carry,k):
        alpha,latch=carry
        a,l,d=advance(alpha,latch,2.8,j.array([0.,1.,2.]),j.full(3,k),0.,RULE)
        return (a,l),j.stack([a,l.astype(a.dtype),d['rule_speed_target']])
    f=jax.jit(lambda k:jax.lax.scan(tick,(j.array(1.),j.array(False)),k))
    _,logs=f(j.array([0.,.5,.5,0.,0.]))
    np.testing.assert_allclose(logs[:,0],[1.,.98,.96,.98,1.],atol=1e-6)


@pytest.mark.parametrize('field,value',[
    ('preview_m',0.),('engage_accel_m_s2',2.7),('release_accel_m_s2',-1.),
    ('active_alpha',1.1),('alpha_rate_per_s',0.),('upper_dt_s',.005),
    ('preview_m',float('nan')),('active_alpha',True)])
def test_invalid_rule_configuration(field,value):
    rule=dict(RULE);rule[field]=value
    with pytest.raises(ValueError):validate_rule(rule)


def contexts():
    return [dict(route=f'S_{int(a)}_{int(d)}',angle_deg=a,radius_m=r,speed=v,direction=d)
            for a,r,v in [(60.,2.2,2.6),(90.,1.8,2.8)] for d in (-1.,1.)]


def protocol():
    return dict(rule=RULE,original_user_alpha=1.,expected_cases=8,
                maximum_lower_ticks=16000,horizon_s=10.)


def test_exact_eight_inventory_and_route_labels_do_not_affect_rule():
    cs=contexts();rows=scheduling_cases(protocol(),cs)
    assert len(rows)==8
    assert all(c['alpha']==c['original_user_alpha']==1. and c['force_N']==0 for c in rows)
    assert [c['policy'] for c in rows]==['alpha_schedule','curvature_speed']*4
    renamed=copy.deepcopy(cs)
    for i,c in enumerate(renamed):c['route']=f'renamed{i}'
    aa=scheduling_cases(protocol(),renamed)
    assert [{k:v for k,v in c.items() if k!='route'} for c in rows]==[{k:v for k,v in c.items() if k!='route'} for c in aa]


@pytest.mark.parametrize('field,value',[('original_user_alpha',.5),('expected_cases',16),
    ('maximum_lower_ticks',32000),('horizon_s',16.)])
def test_invalid_scheduling_inventory(field,value):
    p=protocol();p[field]=value
    with pytest.raises(ValueError):scheduling_cases(p,contexts())


def test_no_silent_geometry_reselection():
    p=protocol();p['contexts']=contexts();p['contexts'][0]['speed']=2.5
    with pytest.raises(ValueError,match='cannot change'):scheduling_cases(p,contexts())
