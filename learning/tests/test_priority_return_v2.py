import dataclasses
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import jax
import jax.numpy as jp
from sttw_control import priority_return_v2 as v2

spec = importlib.util.spec_from_file_location('priority_reference', Path(__file__).parent/'fixtures/priority_return_v2_reference.py')
ref = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = ref
spec.loader.exec_module(ref)

@pytest.mark.parametrize('alpha', [0., .5, 1.])
def test_scalar_reference_and_jit(alpha):
    rng = np.random.default_rng(23)
    for _ in range(30):
        s = ref.Sample(alpha, *rng.uniform(-1,1,3), roll=.4, roll_rate=.7,
                       action=(.4,-.7),previous_action=(-.2,.8),relaxation=rng.random(),
                       missed_deadline_now=True)
        expected = ref.reward_terms(s)
        for xp in (np, jp):
            actual = v2.reward_terms(**dataclasses.asdict(s), xp=xp)
            for group in ('raw_costs','reward_parts','bands'):
                for key in expected[group]:
                    np.testing.assert_allclose(actual[group][key],expected[group][key],rtol=2e-6,atol=2e-6)
            np.testing.assert_allclose(actual['reward'],expected['reward'],rtol=2e-6)
    assert np.isfinite(jax.jit(lambda a: v2.reward_terms(alpha=a,speed_error=.1,lateral_error=.2,heading_error=0.)['reward'])(jp.array(alpha)))

@pytest.mark.parametrize('bad', [.2,np.nan,np.inf])
def test_reject_alpha(bad):
    with pytest.raises(ValueError):
        v2.reward_terms(alpha=bad,speed_error=0,lateral_error=0,heading_error=0,xp=np)

@pytest.mark.parametrize('bad', [np.nan,np.inf,-np.inf])
def test_nonfinite_is_failure(bad):
    assert v2.reward_terms(alpha=0.,speed_error=bad,lateral_error=0.,heading_error=0.)['reward'] == -200

def step(state,t,**kw):
    args=dict(t=t,dt=.05,episode_end=6.,reference_speed=2.,previous_reference_speed=2.,reference_yaw=0.,reference_progress=2.,actual_progress=0.,speed_error=0.,lateral_error=0.,heading_error=0.,roll=0.,roll_rate=0.)
    args.update(kw)
    return v2.advance(state,**args)

def exit_state():
    s,_=step(v2.initial_state(),.05,reference_yaw=.4)
    for t in np.arange(.1,.351,.05):
        s,_=step(s,float(t))
    assert s.exit_valid
    return s

def test_stopped_robot_cannot_freeze_clock_and_no_repeated_debt():
    s=exit_state()
    assert np.isclose(s.exit_progress,2.)
    assert np.isclose(s.deadline,s.exit_time+3)
    s,_=step(s,float(s.exit_time+1.5))
    assert 0 < s.q < 1
    s,e=step(s,float(s.deadline))
    assert s.q==0 and s.deadline_missed and e['new_task_miss']
    s,e=step(s,6.)
    assert not e['new_task_miss'] and not s.task_complete

def test_late_recovery_keeps_miss_and_final_departure_revokes_completion():
    s=exit_state()
    s,_=step(s,float(s.deadline),lateral_error=.5)
    for t in np.arange(4.,4.56,.05):
        s,_=step(s,float(t),actual_progress=3.1)
    assert s.task_complete and s.late_but_finally_recovered and s.deadline_missed
    s,e=step(s,6.,actual_progress=3.2,lateral_error=.2)
    assert not s.task_complete and not e['new_task_miss']

def test_numpy_jax_state_and_observation_equivalence():
    a,b=v2.initial_state(xp=np),v2.initial_state()
    for t in np.arange(.05,6.051,.05):
        kw=dict(reference_yaw=.4 if t<.5 else 0.,actual_progress=float(t))
        a,ea=step(a,float(t),xp=np,**kw)
        b,eb=step(b,float(t),**kw)
        for field in dataclasses.fields(a):
            np.testing.assert_allclose(getattr(a,field.name),getattr(b,field.name),atol=2e-5)
    ctx=jax.jit(lambda s:v2.observation_context(s,t=6.,episode_end=6.,actual_progress=6.))(b)
    assert ctx.shape==(5,)

def test_no_exit_and_no_turn_terminal_contract():
    s=v2.initial_state()
    for t in np.arange(.05,.6,.05):
        s,_=step(s,float(t))
    assert s.task_complete and s.q==0
    s,_=step(s,1.,reference_yaw=.4)
    s,e=step(s,6.,reference_yaw=.4)
    assert not s.task_complete and e['new_task_miss']

def test_preference_order_and_cost_independence_and_dt():
    candidates=[(-.35,.04),(-.10,.10),(-.02,.30)]
    for i,alpha in enumerate((0.,.5,1.)):
        def reward(ev,ey,**kw):
            return v2.reward_terms(alpha=alpha,speed_error=ev,lateral_error=ey,heading_error=0.,xp=np,**kw)
        assert np.argmax([reward(ev,ey)['reward'] for ev,ey in candidates])==i
        assert reward(-.1,.1)['raw_costs']['speed']==reward(-.1,2.)['raw_costs']['speed']
        assert reward(-.1,.1,dt=.02)['reward']==pytest.approx(4*reward(-.1,.1,dt=.005)['reward'])
        assert reward(0.,0.,failed=True,missed_deadline_now=True)['reward']==-200.
        assert sum(reward(-.1,.1)['reward_parts'].values())==reward(-.1,.1)['reward']
        assert all(v>=0 for v in reward(-1.,1.)['raw_costs'].values())


def test_insufficient_window_is_reported_and_jit_advance():
    s,_=step(v2.initial_state(),5.65,reference_yaw=.4)
    for t in np.arange(5.7,5.951,.05):
        s,_=step(s,float(t),actual_progress=2.)
    assert s.exit_valid and s.insufficient_return_window and s.q==0
    compiled=jax.jit(lambda state:step(state,6.,actual_progress=0.))
    s,e=compiled(s)
    assert e['new_task_miss'] and not s.task_complete


def test_progress_target_required_and_repeated_band_entries_never_pay_bonus():
    s=exit_state()
    for t in np.arange(.4,3.6,.05):
        s,e=step(s,float(t),actual_progress=2.5)
    assert not s.task_complete and s.deadline_missed
    deadline=float(s.deadline)
    for t in np.arange(3.6,5.9,.05):
        s,e=step(s,float(t),actual_progress=3.5,lateral_error=.2 if int(t*20)%2 else 0.)
        assert not e['new_task_miss']
        assert s.deadline==deadline and s.deadline_missed
    assert not s.task_complete


def test_exit_requires_stable_speed_and_original_progress_is_immutable():
    s,_=step(v2.initial_state(),.05,reference_yaw=.4)
    for t in np.arange(.1,.6,.05):
        s,_=step(s,float(t),previous_reference_speed=1.9)
    assert not s.exit_valid
    for t in np.arange(.6,.9,.05):
        s,_=step(s,float(t),reference_progress=2.+t)
    assert s.exit_valid and np.isclose(s.exit_progress,2.6)
    original=float(s.exit_progress)
    s,_=step(s,1.,reference_progress=20.,reference_yaw=.4)
    assert s.exit_progress==original


def test_deadline_adjudicated_once_but_final_departure_still_fails():
 s=exit_state()
 for t in np.arange(.4,float(s.deadline)+.051,.05):s,_=step(s,float(t),actual_progress=3.5)
 assert s.task_complete and not s.deadline_missed
 s,e=step(s,4.,actual_progress=3.5,lateral_error=.2)
 assert not s.task_complete and not s.deadline_missed and not e['new_task_miss']
 for t in np.arange(4.1,5.91,.05):s,_=step(s,float(t),actual_progress=3.5)
 assert s.task_complete and not s.late_but_finally_recovered
 s,e=step(s,6.,actual_progress=3.5,lateral_error=.2)
 assert not s.task_complete and not s.deadline_missed and e['new_task_miss']


def test_published_request_enters_before_slew_exceeds_exit_threshold():
 s,e=step(v2.initial_state(),1.,reference_speed=2.305,previous_reference_speed=2.3,reference_yaw=.012,published_yaw_request=1.8)
 assert s.seen_maneuver and s.q==1 and not s.exit_valid
