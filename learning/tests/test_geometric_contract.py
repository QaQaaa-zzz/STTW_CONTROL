"""Reward contracts and projection tests; no MuJoCo/Flax or training required."""
from dataclasses import replace
import numpy as np
import jax
import jax.numpy as jp
import pytest
from sttw_control.tracking_reward import TrackingConfig, initial_return, transition
from sttw_control.timed_reference import (TimedReferenceConfig, geometry_table,
                                         project_geometry, errors)

C=TrackingConfig(objective='geometric_huber',return_bonus=0.,tail_rate=0.,
                 speed_scale=.1,lateral_scale=.1,deadline_penalty=5.,overdue_rate=2.)

def step(s=None,**overrides):
    kw=dict(roll=0.,roll_rate=0.,speed_error=0.,lateral_error=0.,heading_error=0.,
            action=np.zeros(2),alpha=.5,dt=.005,alive_rate=1.,failure_penalty=100.,
            failed=False,enabled=True,config=C,xp=np)
    kw.update(overrides)
    return transition(initial_return(xp=np) if s is None else s,**kw)

@pytest.mark.parametrize('a',[0.,.25,.5,.75,1.])
def test_shared_ideal_is_allowed_to_win_every_alpha(a):
    ideal=sum(step(alpha=a)[1].values())
    assert ideal==pytest.approx(.005)
    assert ideal>sum(step(alpha=a,speed_error=.25)[1].values())
    assert ideal>sum(step(alpha=a,lateral_error=.2)[1].values())

def test_genuine_conflict_changes_endpoint_preference_without_forced_diversity():
    def value(a,v,y):return sum(step(alpha=a,speed_error=v,lateral_error=y,heading_error=.05)[1].values())
    assert value(0,.25,.05)>value(0,.05,.2)
    assert value(1,.05,.2)>value(1,.25,.05)

def test_dominated_trace_cannot_be_made_a_required_winner():
    for a in (0,.5,1):
        assert sum(step(alpha=a,speed_error=.02,lateral_error=.03)[1].values())>sum(step(alpha=a,speed_error=.04,lateral_error=.06)[1].values())

def test_no_yaw_or_time_lag_terms_and_no_recovery_bonus():
    _,base=step()
    _,extra=step(longitudinal_error=-3.,yaw_rate_error=4.)
    assert base==extra
    assert not any('yaw' in k or 'longitudinal' in k for k in base)
    s,_=step(roll_rate=.3001)
    reward=0.
    for _ in range(100):s,p=step(s);reward+=sum(p.values());assert p['recovery']==0
    assert s.credited and not s.pending and reward<=100*.005

def test_single_self_excursion_does_not_earn_reward():
    s=initial_return(xp=np);a=0.
    for i in range(130):
        s,p=step(s,roll_rate=.3001 if i==5 else 0.)
        a+=sum(p.values())
    assert a<130*.005

def test_deadline_is_once_not_dt_scaled_and_late_hold_not_credited():
    s=initial_return(xp=np);events=[]
    for i in range(605):
        s,p=step(s,lateral_error=.2);events.append(p['deadline'])
        if i==599:assert not s.deadline_missed
    assert s.deadline_missed and sum(events)==-5. and events[600]==-5.
    assert p['return_overdue']==pytest.approx(-.01)
    for _ in range(100):s,p=step(s)
    assert s.deadline_missed and not s.credited and not s.pending
    assert p['return_overdue']==0 and p['deadline']==0

def test_failure_still_replaces_every_component():
    _,p=step(failed=True,lateral_error=30.)
    assert sum(p.values())==-100 and p['failure']==-100
    assert all(v==0 for k,v in p.items() if k!='failure')

def test_legacy_defaults_preserve_positive_peak_and_bonus():
    legacy=TrackingConfig()
    s,p=step(config=legacy)
    assert sum(p.values())==pytest.approx(.025)
    assert 'deadline' not in p and 'return_overdue' not in p
    with pytest.raises(ValueError):replace(C,timed=True)
    with pytest.raises(ValueError):replace(C,return_bonus=2.)

def test_curved_path_lag_is_not_geometric_error_and_curve_is_fixed():
    cfg=TimedReferenceConfig(mode='geometry',fixed=((0.,2.,0.),(1.,2.,.4)))
    table=np.asarray(geometry_table(jp.asarray(cfg.fixed),cfg,2.,jp.zeros(3),4.,.005))
    i=40;u=.3
    pose=table[i,1:4]*(1-u)+table[i+1,1:4]*u
    old=table[i,0]
    s,foot,k=project_geometry(pose,table,old,.2,xp=np)
    np.testing.assert_allclose(foot,pose,atol=2e-6)
    assert np.linalg.norm(errors(pose,table[i+20,1:4],[2.,.4],xp=np)[[0,3]])>.5
    np.testing.assert_allclose(errors(pose,foot,[2.,2*k],xp=np)[:2],0,atol=2e-6)
    # Static table does not depend on vehicle speed, disturbance or alpha.
    assert s==pytest.approx(table[i,0]+u*(table[i+1,0]-table[i,0]),abs=2e-6)

def test_continuity_window_prevents_jump_to_later_crossing():
    table=np.array([[0,0,0,np.pi/4,0],[2.828,2,2,np.pi/4,0],
                    [4.828,0,2,-np.pi/4,0],[7.656,2,0,-np.pi/4,0]])
    s,_,_=project_geometry(np.array([1.,1.,-np.pi/4]),table,1.4,.2,xp=np)
    assert 1.2<=s<=1.6

def test_jit_vmap_scan_matches_numpy_contract():
    def run(a):
        def tick(s,y):
            return transition(s,roll=0.,roll_rate=0.,speed_error=.04,lateral_error=y,
                heading_error=.02,action=jp.zeros(2),alpha=a,dt=.005,alive_rate=1.,
                failure_penalty=100.,failed=False,enabled=True,config=C)
        s,p=jax.lax.scan(tick,initial_return(),jp.ones(605)*.2)
        return s,sum(p.values())
    states,values=jax.jit(jax.vmap(run))(jp.array([0.,.5,1.]))
    assert np.all(np.asarray(states.deadline_missed))
    for j,a in enumerate([0.,.5,1.]):
        s=initial_return(xp=np);expected=[]
        for _ in range(605):
            s,p=step(s,alpha=a,speed_error=.04,lateral_error=.2,heading_error=.02)
            expected.append(sum(p.values()))
        np.testing.assert_allclose(values[j],expected,atol=2e-6,rtol=2e-5)
