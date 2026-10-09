import copy
from pathlib import Path
import jax
import numpy as np
from sttw_control.smooth_command_config import resolve
from sttw_control.direct_command_scenarios import schedule
from sttw_control.direct_command_env import task_horizon_steps, remaining_time_feature

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
