"""Geometry, unchanged destination, causal trigger and full-state preservation."""
from types import SimpleNamespace
from dataclasses import dataclass,replace,fields
import numpy as np
import pytest
from sttw_control.goal_replanning import dubins_goal_path,replan_requested,rebind_geometry


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
