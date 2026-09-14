import numpy as np
import pytest


def reward_record(rewards,*,failed=False):
    count=len(rewards)
    terms={'alive':np.full(count,.5),'speed':np.asarray(rewards)-.5,'failure':np.zeros(count)}
    if failed:
        terms['alive'][-1]=0.
        terms['speed'][-1]=0.
        terms['failure'][-1]=rewards[-1]
    trace={'time':np.arange(count+1)*.005,'reward':np.r_[0.,rewards],
           'terminated':np.r_[np.zeros(count,bool),failed],
           'user_command':np.tile([2.,0.],(count+1,1)),
           'priority_alpha':np.full(count+1,.5)}
    return {'trace':trace,'rewards':terms,'config':{'controller':{'dt':.005},'horizon_seconds':.02}}


def test_paired_returns_retain_failure_and_distinguish_common_window():
    from sttw_control.command_diagnostics import paired_reward_summary
    result=paired_reward_summary(reward_record([.2,-100.],failed=True),reward_record([.1,.3,.2,.4]))
    assert result['baseline']['observed_return']==pytest.approx(-99.8)
    assert result['residual']['observed_return']==pytest.approx(1.)
    assert result['delta_residual_minus_baseline']==pytest.approx(100.8)
    assert result['observed_windows_match'] is False
    assert result['baseline']['failure_time_seconds']==.01
    assert result['baseline']['component_returns']['failure']==-100.
    common=result['common_window']
    assert common['steps']==2 and common['end_seconds']==.01
    assert common['baseline']['observed_return']==pytest.approx(-99.8)
    assert common['residual']['observed_return']==pytest.approx(.4)
    assert common['delta_residual_minus_baseline']==pytest.approx(100.2)
    assert common['baseline']['terminal_failure_included'] is True
    for scope in (result,common):
        for label in ('baseline','residual'):
            item=scope[label]
            assert item['component_sum']==pytest.approx(item['observed_return'])


def test_paired_returns_reject_different_commands_or_control_timestamps():
    from sttw_control.command_diagnostics import paired_reward_summary
    baseline=reward_record([.2,.3])
    changed=reward_record([.2,.3])
    changed['trace']['time'][1]+=.001
    with pytest.raises(ValueError,match='time'):
        paired_reward_summary(baseline,changed)
    changed=reward_record([.2,.3])
    changed['trace']['user_command'][0,1]=.5
    with pytest.raises(ValueError,match='command'):
        paired_reward_summary(baseline,changed)


def test_step_reward_artifact_retains_actual_cumulative_and_common_delta(tmp_path):
    from sttw_control.command_diagnostics import plot_step_rewards
    plot_step_rewards(tmp_path,reward_record([.2,-100.],failed=True),reward_record([.1,.3,.2,.4]),[[0.,2.,0.,.5]],'fixture')
    with np.load(tmp_path/'reward_steps.npz') as saved:
        np.testing.assert_allclose(saved['baseline_cumulative_return'],[.2,-99.8])
        np.testing.assert_allclose(saved['residual_cumulative_return'],[.1,.4,.6,1.])
        np.testing.assert_allclose(saved['common_cumulative_delta'],[-.1,100.2])
        np.testing.assert_allclose(saved['residual_component_sum'],[.1,.3,.2,.4])
    assert (tmp_path/'reward_vs_steps.png').exists()
    assert (tmp_path/'reward_vs_steps.pdf').exists()


def test_generate_indexes_paired_returns_and_component_totals(tmp_path):
    from dataclasses import asdict
    import json
    from sttw_control.command_diagnostics import generate
    from sttw_control.env import TaskConfig
    from sttw_control.motion_commands import MotionCommands
    from sttw_control.observation import ObservationConfig
    from sttw_control.priority import PriorityConfig
    config=asdict(TaskConfig(horizon_seconds=.02,failure_penalty=100.,motion_commands=MotionCommands(fixed=((0.,2.,0.,.5),)),priority=PriorityConfig(risk_gate=False),observation=ObservationConfig(include_motion=True,include_priority=True)))
    case=tmp_path/'evaluation/alpha_0/seed_7/command_case'
    for label,failed,count in [('baseline',True,3),('residual',False,5)]:
        path=case/label;path.mkdir(parents=True)
        qpos=np.zeros((count,7));qpos[:,3]=1.
        qvel=np.zeros((count,6));qvel[:,0]=2.
        reward=np.r_[0.,np.full(count-1,.005)]
        terminated=np.zeros(count,bool)
        if failed:reward[-1]=-100.;terminated[-1]=True
        np.savez(path/'trace.npz',time=np.arange(count)*.005,qpos=qpos,qvel=qvel,pose=np.zeros((count,3)),measurement=np.zeros((count,7)),reference_roll=np.zeros(count),motion_command=np.tile([0.,2.],(count,1)),priority_alpha=np.full(count,.5),user_command=np.tile([2.,0.],(count,1)),yaw_rate_world=np.zeros(count),action=np.zeros((count,2)),reward=reward,terminated=terminated)
        (path/'declaration.json').write_text(json.dumps({'config':config}))
        (path/'commands.json').write_text(json.dumps({'schedule':[[0.,2.,0.,.5]]}))
    rows=generate(tmp_path)
    assert rows[0]['observed_return']==pytest.approx(-99.995)
    assert rows[1]['observed_return']==pytest.approx(.02)
    assert rows[0]['component_returns']['failure']==-100.
    pairs=json.loads((tmp_path/'analysis/paired_rewards.json').read_text())
    assert pairs[0]['delta_residual_minus_baseline']==pytest.approx(100.015)
    assert pairs[0]['common_window']['delta_residual_minus_baseline']==pytest.approx(100.005)
    assert pairs[0]['common_surviving_window']['steps']==1
    assert pairs[0]['common_surviving_window']['baseline_mean_penalties']['speed']==0.
    assert rows[0]['return_scope'].startswith('each_observed_episode')
    assert (tmp_path/'analysis/alpha_0/seed_7/command_case/reward_comparison.json').exists()
