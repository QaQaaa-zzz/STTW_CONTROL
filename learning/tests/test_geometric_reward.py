"""Opt-in geometric loss, shrinking return bands and deadline contract."""
from dataclasses import replace
import jax
import jax.numpy as jp
import numpy as np
import pytest
from sttw_control.tracking_reward import TrackingConfig, initial_return, transition, within_final


def config(**kw):
    return replace(TrackingConfig(timed=True, geometric=True, reward_mode='huber',
        shrink_tolerances=True, return_bonus=0., deadline_penalty=5., over_deadline_rate=3.,
        speed_scale=.3, lateral_scale=.1, heading_scale=.15, heading_tail_weight=.2,
        priority_ratio=20.), **kw)


def tick(state=None, c=None, xp=np, **kw):
    values=dict(roll=0.,roll_rate=0.,speed_error=0.,lateral_error=0.,heading_error=0.,
        longitudinal_error=0.,yaw_rate_error=0.,action=xp.zeros(2),alpha=.5,dt=.02,
        alive_rate=1.,failure_penalty=100.,failed=False,enabled=True)
    values.update(kw)
    return transition(initial_return(xp=xp) if state is None else state,
                      config=config() if c is None else c,xp=xp,**values)


def test_geometric_ignores_time_phase_and_yaw_reward():
    c=config()
    assert within_final(0.,0.,0.,0.,0.,c,longitudinal_error=100.,yaw_rate_error=100.,xp=np)
    _, a=tick(c=c)
    _, b=tick(c=c,longitudinal_error=100.,yaw_rate_error=100.)
    assert a==b
    assert a['speed_tracking']==a['path_tracking']==a['speed_tail']==a['path_tail']==0.


def test_huber_preference_changes_direction_under_common_cost():
    c=config(budget_rate=0.,return_rate=0.)
    # Equal normalized errors: each endpoint prefers its designated dimension.
    totals=[]
    for alpha in (0.,.5,1.):
        _, speed=tick(c=c,alpha=alpha,speed_error=.3)
        _, path=tick(c=c,alpha=alpha,lateral_error=.1)
        totals.append((sum(speed.values()),sum(path.values())))
    assert totals[0][0]>totals[0][1]
    assert totals[1][0]==pytest.approx(totals[1][1])
    assert totals[2][0]<totals[2][1]
    ideal=sum(tick(c=c)[1].values())
    assert ideal>sum(tick(c=c,speed_error=-1.5)[1].values())
    assert ideal>sum(tick(c=c,lateral_error=.5)[1].values())


def test_shrinking_bands_reach_common_final_before_hold():
    c=config()
    state=initial_return(xp=np)._replace(pending=np.asarray(True),elapsed=np.asarray(2.48))
    for alpha in (0.,1.):
        _, parts=tick(state,c=c,alpha=alpha,speed_error=.35,lateral_error=.25)
        assert parts['speed_budget']==pytest.approx(-.02*c.budget_rate*(.15/.3)**2)
        assert parts['path_budget']==pytest.approx(-.02*c.budget_rate*(2*.15/.1-1))


def test_deadline_once_and_persistent_overdue_until_recovered():
    c=config(return_seconds=.1,hold_seconds=.04)
    state=None
    charges=[]
    for _ in range(10):
        state,p=tick(state,c=c,speed_error=1.)
        charges.append(p['deadline'])
    assert sum(charges)==-5.
    assert p['over_deadline']==pytest.approx(-.06)
    for _ in range(2):
        state,p=tick(state,c=c)
    assert not state.pending
    assert p['over_deadline']==0.
    assert p['recovery']==0.
    # A second departure must not earn another recovery event bonus.
    state,_=tick(state,c=c,speed_error=1.)
    for _ in range(2):
        state,p=tick(state,c=c)
    assert p['recovery']==0.


def test_depart_and_recover_never_beats_ideal():
    c=config(return_seconds=.1,hold_seconds=.04)
    state=None
    total=0.
    for ev in [0.]*3+[.4]*4+[0.]*8:
        state,p=tick(state,c=c,speed_error=ev)
        total+=sum(p.values())
    assert total<15*sum(tick(c=c)[1].values())


def test_numpy_jit_vmap_matches_for_deadline_and_failure():
    c=config()
    state=initial_return(xp=np)._replace(pending=np.asarray(True),elapsed=np.asarray(3.))
    errors=np.asarray([0.,.25,2.,np.nan],dtype=np.float32)
    def run(error):
        return tick(state,c=c,xp=jp,speed_error=error,lateral_error=.3,failed=jp.isnan(error))
    batched=jax.jit(jax.vmap(run))(jp.asarray(errors))
    for i,error in enumerate(errors):
        expected=tick(state,c=c,speed_error=error,lateral_error=.3,failed=np.isnan(error))
        for got,want in zip(jax.tree_util.tree_leaves(batched),jax.tree_util.tree_leaves(expected)):
            np.testing.assert_allclose(np.asarray(got)[i],want,rtol=2e-5,atol=2e-6)
    assert sum(float(v[-1]) for v in batched[1].values())==-100.


@pytest.mark.parametrize('kw',[dict(reward_mode='bad'),dict(geometric=1),dict(shrink_tolerances=1),
    dict(deadline_penalty=-1.),dict(over_deadline_rate=-1.),dict(return_seconds=.5,hold_seconds=.5),
    dict(speed_tight=.1),dict(lateral_tight=.05)])
def test_invalid_config(kw):
    with pytest.raises(ValueError):config(**kw)
