from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from sttw_control.motion_commands import MotionCommands, sample_schedule, reference_at, reward_terms
from sttw_control.env import RecoveryEnv, TaskConfig
from sttw_control.observation import ObservationConfig, observation_fields
from sttw_control.priority import PriorityConfig

def task(**kw):
    return TaskConfig(motion_commands=MotionCommands(**kw), priority=PriorityConfig(risk_gate=False), observation=ObservationConfig(history_steps=10,include_priority=True,include_attitude_risk=False,include_motion=True), horizon_seconds=.1)

def test_schedule_random_and_replayable():
    c=MotionCommands();a=sample_schedule(jax.random.PRNGKey(2),c,2.)
    np.testing.assert_array_equal(a,sample_schedule(jax.random.PRNGKey(2),c,2.))
    assert not np.array_equal(a,sample_schedule(jax.random.PRNGKey(3),c,2.))
    assert np.all(np.diff(a[:,0])>0)
    assert np.unique(np.asarray(a[:,3])).size>1

def test_priority_history_and_command_visibility():
    e=RecoveryEnv(task(fixed=((0.,2.,0.,.2),(.005,2.2,.4,.8))))
    s=e.reset(2);n=e.step(s,jnp.zeros(2))
    i=observation_fields(e.config.observation).index('speed_priority')
    np.testing.assert_allclose(n.history.frames[-2:,i],[.2,.8])
    locked=e.set_priority(s,.5);n=e.step(locked,jnp.zeros(2))
    assert float(n.priority_alpha)==.5
    assert n.command_schedule.shape==(2,4)

def test_reward_has_fixed_balance_and_correct_preferences():
    c=MotionCommands()
    lo=reward_terms(.1,.2,.3,.4,jnp.zeros(2),0.,c)
    hi=reward_terms(.1,.2,.3,.4,jnp.zeros(2),1.,c)
    assert lo['attitude']==hi['attitude']
    assert hi['speed']>lo['speed'] and hi['yaw']<lo['yaw']

def test_zero_residual_preserves_straight_baseline_physics():
    cfg=task(fixed=((0.,2.,0.,.5),))
    e=RecoveryEnv(cfg);old=RecoveryEnv(TaskConfig(horizon_seconds=.1))
    s=e.reset(8);b=old.reset(8)
    for _ in range(4):
        s=e.step(s,jnp.zeros(2));b=old.step(b,jnp.zeros(2))
        np.testing.assert_allclose(s.data.qpos,b.data.qpos,atol=1e-10)
        np.testing.assert_allclose(s.data.ctrl,b.data.ctrl,atol=1e-10)

def test_reward_reconstruction_with_switch_and_dynamic_alpha(tmp_path):
    from sttw_control.evaluation import evaluate
    from sttw_control.command_diagnostics import reconstruct
    e=RecoveryEnv(task(fixed=((0.,2.,0.,.2),(.01,2.2,.4,.8))))
    evaluate(e,tmp_path/'trace',seed=7)
    x=reconstruct(tmp_path/'trace')
    assert x['reconstruction_error']<3e-5
    assert x['trace']['user_command'][1,1]==0
    assert x['trace']['user_command'][2,1]>.3


def test_current_command_config_restores_point_three_roll_threshold():
    from pathlib import Path
    from sttw_control.env import load_config
    c=load_config(Path(__file__).parents[1]/'configs/command_recovery.json')
    assert np.isclose(c.motion_commands.roll_working_limit,0.3)
    assert c.actuator.composition=='full_range'
    cost=lambda deg:reward_terms(np.deg2rad(deg),0.,0.,0.,jnp.zeros(2),.5,c.motion_commands)['attitude']
    assert cost(9)==0 and cost(17)==0 and cost(18)>0


def test_full_range_zero_preserves_physics_and_records_new_reward(tmp_path):
    from pathlib import Path
    from sttw_control.env import load_config
    from sttw_control.actuator import ActuatorConfig
    from sttw_control.evaluation import evaluate
    from sttw_control.command_diagnostics import reconstruct
    c=load_config(Path(__file__).parents[1]/'configs/command_recovery.json')
    c=replace(c,horizon_seconds=.025)
    e=RecoveryEnv(c);old=RecoveryEnv(replace(c,actuator=ActuatorConfig()))
    s=e.reset(4);b=old.reset(4)
    for _ in range(5):
        s=e.step(s,jnp.zeros(2));b=old.step(b,jnp.zeros(2))
        np.testing.assert_allclose(s.data.qpos,b.data.qpos,atol=1e-10)
    evaluate(e,tmp_path/'trace',seed=4)
    assert reconstruct(tmp_path/'trace')['reconstruction_error']<3e-5


def test_signed_reward_components_match_reward_and_replace_failure():
    from sttw_control.motion_commands import signed_reward_components
    c=MotionCommands()
    action=jnp.array([.2,-.3])
    costs=reward_terms(.4,.3,.2,-.1,action,.7,c)
    parts=signed_reward_components(.4,.3,.2,-.1,action,.7,c,.005,1.,100.,False)
    np.testing.assert_allclose(sum(parts.values()),.005*(1-sum(costs.values())),atol=1e-7)
    failed=signed_reward_components(.4,.3,.2,-.1,action,.7,c,.005,1.,100.,True)
    assert failed['failure']==-100
    assert all(float(v)==0 for k,v in failed.items() if k!='failure')
    assert parts['alive']>0 and parts['speed']<0 and parts['yaw']<0
