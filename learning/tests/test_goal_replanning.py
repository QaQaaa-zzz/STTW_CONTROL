"""Geometry, unchanged destination, causal trigger and full-state preservation."""
from types import SimpleNamespace
from dataclasses import dataclass,replace,fields
import numpy as np
import pytest
from sttw_control.goal_replanning import dubins_goal_path,replan_requested,rebind_geometry


def test_free_heading_tangent_avoids_observed_near_goal_loop():
    # Recorded shared-switch pose previously selected a 10.480m LRL loop
    # because its short tangent heading lay between the discrete candidates.
    goal=[12.949442863464355,7.052823066711426]
    path,meta=dubins_goal_path([12.400750160217285,4.727596759796143,2.0595498085021973],goal,numpy_output=True)
    assert meta['goal_arclength_m']==pytest.approx(2.60483533266846,abs=1e-6)
    assert meta['modes']==['R','S','R']
    np.testing.assert_allclose(path.xy[np.argmin(abs(path.s-path.goal))],goal)


@pytest.mark.parametrize('pose,goal',[
    ([0,0,0],[8,0]),([0,0,0],[3,4]),([2,-1,2.9],[-3,-4]),
    ([0,0,0],[-4,0]),([0,0,0],[0,.2]),([4,3,-2],[4,3.01])])
def test_arc_tangent_keeps_goal_and_forward_heading_under_large_errors(pose,goal):
    path,meta=dubins_goal_path(pose,goal)
    s,xy,h,k=map(np.asarray,[path.s,path.xy,path.heading,path.curvature])
    assert len(s)==4001 and np.all(np.diff(s)>0)
    np.testing.assert_allclose(xy[0],pose[:2],atol=2e-6)
    np.testing.assert_allclose([np.interp(float(path.goal),s,xy[:,i]) for i in (0,1)],goal,atol=3e-6)
    assert np.max(abs(k))<=1/1.8+1e-7 and np.isfinite(xy).all()
    assert h[0]==pytest.approx(pose[2],abs=1e-6)
    # Every sampled segment advances approximately along its analytic heading;
    # a loop behind the robot is permitted, a backwards/cusp shortcut is not.
    dxy=np.diff(xy,axis=0);mid=(h[1:]+h[:-1])/2
    forward=dxy[:,0]*np.cos(mid)+dxy[:,1]*np.sin(mid)
    assert np.min(forward)>0
    assert not meta['safety_certificate'] and not meta['curvature_continuous']


def test_replan_trigger_uses_current_geometry_and_never_overwrites_arrival_or_failure():
    args=dict(xy=[0,0],goal_xy=np.array([5,2]),cross_track_m=1.2,
              follower_invalid=False,already_replanned=False,contact=False,fault=False)
    assert replan_requested(**args)
    assert replan_requested(**dict(args,cross_track_m=0.,follower_invalid=True))
    assert not replan_requested(**dict(args,cross_track_m=.5))
    for key in ('already_replanned','contact','fault','ever_arrived'):
        assert not replan_requested(**dict(args,**{key:True}))
    assert not replan_requested(**dict(args,goal_xy=np.array([.5,.5])))
    with pytest.raises(ValueError):replan_requested(**dict(args,cross_track_m=np.nan))


def test_geometry_binding_preserves_every_other_state_and_carry_field(monkeypatch):
    import sttw_control.finite_value_evaluation as module
    @dataclass(frozen=True)
    class State:
        projection:object;follower:object;physical:object;lower:object;history:object
        filtered:object;offsets:object;tick:object;alpha:object;domain_exit:object
        replace=replace
    @dataclass(frozen=True)
    class Carry:
        state:object;gate:object;relay:object;clean_blocks:object
        replace=replace
    state=State(*[object() for _ in fields(State)])
    carry=Carry(state,object(),object(),object());seen=[]
    def bind(s,p,v,e):
        seen.append((s,p,v,e));return s.replace(projection='new projection',follower='new follower')
    monkeypatch.setattr(module,'rebind_path_context',bind)
    out=rebind_geometry(carry,'new path',1.1,'env')
    assert seen==[(state,'new path',1.1,'env')]
    for f in fields(State):
        if f.name not in ('projection','follower'):assert getattr(out.state,f.name) is getattr(state,f.name)
    for name in ('gate','relay','clean_blocks'):assert getattr(out,name) is getattr(carry,name)


def test_degenerate_or_out_of_support_goal_is_explicit_instead_of_moved():
    with pytest.raises(ValueError,match='already'):dubins_goal_path([0,0,0],[0,0])
    with pytest.raises(ValueError,match='beyond'):dubins_goal_path([0,0,0],[100,0])
    with pytest.raises(ValueError):dubins_goal_path([0,0,np.nan],[2,3])


def test_prescribed_goal_heading_preserves_outgoing_original_direction():
    angle=-1.95
    path,meta=dubins_goal_path([13.67,-6.72,-1.98],[11.66,-9.06],goal_heading_rad=angle)
    end=np.asarray(path.xy)[-1]-np.asarray(path.xy)[-2]
    np.testing.assert_allclose(end/np.linalg.norm(end),[np.cos(angle),np.sin(angle)],atol=1e-3)
    assert meta['goal_heading_mode']=='prescribed' and meta['goal_heading_candidates']==1
    assert meta['goal_heading_rad']==angle


def test_terminal_heading_fallback_preserves_feasible_preferred_path():
    preferred,old=dubins_goal_path([0,0,0],[3,4],goal_heading_rad=1.2)
    selected,new=dubins_goal_path([0,0,0],[3,4],goal_heading_rad=1.2,fallback_max_length_m=100.)
    np.testing.assert_allclose(selected.xy,preferred.xy)
    assert not new['heading_fallback_attempted'] and not new['heading_fallback_selected']
    assert new['goal_heading_candidates']==1 and new['goal_heading_mode']=='prescribed'


def test_terminal_heading_fallback_recovers_rejected_geometry_without_moving_goal():
    pose=[12.457101821899414,.12004201859235764,-.10719804465770721]
    goal=[13.314137259798894,4.9078766422536395]
    path,meta=dubins_goal_path(pose,goal,goal_heading_rad=1.3671023845672607,fallback_max_length_m=8.775142669677734)
    assert meta['preferred_candidate_length_m']>8.775 and float(path.goal)<8.775
    assert meta['heading_fallback_attempted'] and meta['heading_fallback_selected']
    assert meta['goal_heading_mode']=='finite_fallback' and meta['goal_heading_candidates']>=9
    np.testing.assert_allclose([np.interp(float(path.goal),path.s,path.xy[:,k]) for k in (0,1)],goal,atol=3e-6)
    # An unavailable short route stays too long; do not clip or move the goal.
    path2,meta2=dubins_goal_path(pose,goal,goal_heading_rad=1.3671023845672607,fallback_max_length_m=1.)
    assert float(path2.goal)>1. and meta2['maximum_goal_length_m']==1.
    with pytest.raises(ValueError):dubins_goal_path(pose,goal,fallback_max_length_m=1.)
