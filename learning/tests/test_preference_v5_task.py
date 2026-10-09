import copy
import json
from pathlib import Path
import jax
import numpy as np
import torch
from sttw_control.smooth_command_config import resolve
from sttw_control.direct_command_scenarios import schedule
from sttw_control.direct_command_env import task_horizon_steps, remaining_time_feature
from sttw_control.smooth_command_training import validate_preference_stage2_parent

def test_finite_task_horizon_and_critic_clock():
    s=resolve(0,fresh=True);s['episode_duration_s']=5.;s['critic_horizon_normalizer_s']=16.
    assert task_horizon_steps(s)==250
    assert remaining_time_feature(0,s)==5/16
    assert remaining_time_feature(1000,s)==0
    assert task_horizon_steps(resolve(0))==800

def test_stage1_conflict_never_returns_and_uses_prescribed_range():
    s=resolve(0,fresh=True)
    s['preference_v5']={'training_stage':1}
    s['training_stage']=1
    s['commands'].update(nominal_fraction=0.,conflict_fraction=1.,random_fraction=0.)
    c={'conflict_speed_target_range_m_s':[2.5,2.6], 'turn_start_range_s':[1.,1.5], 'steer_magnitude_range_rad':[.24,.28], 'turn_sign_probability_positive':.5}
    s['preference_v5']['training']={'stage1':c}
    rows,family,slew=schedule(0,0,2.6,s)
    rows=np.asarray(rows);active=rows[rows[:,0]<99]
    assert family==1 and len(active)==2
    assert 2.5<=active[0,1]<=2.6 and 1<=active[1,0]<=1.5
    assert .24<=abs(active[1,2])<=.28


def test_stage2_parent_requires_complete_failed_gate_and_full_learner_state(tmp_path):
    parent=tmp_path/'stage1'
    (parent/'alpha0/checkpoints').mkdir(parents=True)
    (parent/'alpha1/checkpoints').mkdir(parents=True)
    (parent/'status.json').write_text(json.dumps(dict(
        state='complete',stage='stage1_gate_failed_stopped',training_stage=1,
        completed_updates={'0':40,'1':40},stage1_gate_passed=False,stage2_started=False)))
    for alpha in (0,1):
        checkpoint=parent/f'alpha{alpha}/checkpoints/update_0040.pt'
        torch.save(dict(update=40,warmup=0,accepted_policy_updates=40,
                        policy={'actor.weight':torch.ones(1)},
                        optimizer={'state':{0:{'step':torch.tensor(40.)}},'param_groups':[{}]},
                        config={'preference_v5_stage':1},torch_rng=torch.get_rng_state(),
                        cuda_rng=[],numpy_rng=np.random.get_state(),python_rng=__import__('random').getstate()),checkpoint)
        (parent/f'alpha{alpha}/last_completed.json').write_text(json.dumps(dict(
            update=40,warmup=0,accepted_policy_updates=40,checkpoint=str(checkpoint),stop=None)))

    receipt=validate_preference_stage2_parent(parent)
    assert receipt['stage1_gate_passed'] is False
    assert receipt['user_override_required'] is True
    assert set(receipt['endpoints'])=={'0','1'}
    assert all(row['update']==40 and row['optimizer_state_present'] for row in receipt['endpoints'].values())

    status=json.loads((parent/'status.json').read_text());status['completed_updates']['1']=39
    (parent/'status.json').write_text(json.dumps(status))
    try:
        validate_preference_stage2_parent(parent)
    except ValueError as exc:
        assert 'exactly 40' in str(exc)
    else:
        raise AssertionError('incomplete Stage1 parent was accepted')
