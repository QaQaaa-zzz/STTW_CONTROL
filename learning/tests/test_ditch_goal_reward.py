import torch
from sttw_control.ditch_goal_reward import ditch_goal_components


def test_goal_and_velocity_rewards_follow_task_not_height():
 c=dict(goal_x_m=6.2,goal_y_m=0.,target_speed_m_s=1.5,position_weight=12.,velocity_weight=8.,position_scale_m=2.,velocity_scale_m_s=.7)
 s=dict(x=torch.tensor([2.,5.,5.]),y=torch.zeros(3),forward_velocity=torch.tensor([1.5,1.5,-1.5]),lateral_velocity=torch.zeros(3))
 r=ditch_goal_components(s,c)
 assert r['ditch_goal_position'][1]>r['ditch_goal_position'][0]
 assert r['ditch_goal_velocity'][1]>r['ditch_goal_velocity'][2]
 assert torch.isfinite(sum(r.values())).all()


def test_exact_goal_is_finite():
 c=dict(goal_x_m=6.2,goal_y_m=0.,target_speed_m_s=1.5,position_weight=12.,velocity_weight=8.,position_scale_m=2.,velocity_scale_m_s=.7)
 s=dict(x=torch.tensor([6.2]),y=torch.zeros(1),forward_velocity=torch.zeros(1),lateral_velocity=torch.zeros(1))
 assert torch.isfinite(sum(ditch_goal_components(s,c).values())).all()


def test_signed_progress_cannot_pay_for_waiting_or_backtracking():
 from sttw_control.ditch_goal_reward import ditch_event_components
 c=dict(start_x_m=1.5,goal_x_m=5.6,progress_per_m=20.,takeoff_bonus=10.,clear_bonus=60.,recover_bonus=150.,failure_penalty=400.)
 x=torch.tensor([1.5,2.,2.,1.5,5.6,6.])
 s=dict(previous_x=x[:-1],x=x[1:])
 e={k:torch.zeros(5,dtype=torch.bool) for k in ('first_ditch_flight','first_ditch_clear','first_platform_hold','ditch_failed')}
 r=ditch_event_components(s,e,c)['ditch_progress']
 assert r[1]==0 and r[2]<0 and r[-1]==0
 torch.testing.assert_close(r.sum(),torch.tensor(82.))


def test_failure_cannot_keep_positive_return_even_after_all_milestones():
 from sttw_control.ditch_goal_reward import ditch_event_components
 c=dict(start_x_m=1.5,goal_x_m=5.6,progress_per_m=20.,takeoff_bonus=10.,clear_bonus=60.,recover_bonus=150.,failure_penalty=400.)
 s=dict(previous_x=torch.tensor([1.5]),x=torch.tensor([5.6]))
 e={k:torch.ones(1,dtype=torch.bool) for k in ('first_ditch_flight','first_ditch_clear','first_platform_hold','ditch_failed')}
 torch.testing.assert_close(sum(ditch_event_components(s,e,c).values()),torch.tensor([-98.]))


def test_event_task_has_no_near_bank_bonus_or_terminal_mode_loophole():
 import json
 from pathlib import Path
 from sttw_control.phase_u_reward import phase_u_reward
 c=json.loads(Path('learning/configs/isaac_phase_u_task_ditch_goal.json').read_text())['task']['reward']
 s={k:torch.zeros(4) for k in ('roll','pitch','yaw','pitch_rate','hip_velocity','knee_velocity','hip_force','knee_force','y','lateral_velocity','yaw_rate')}
 s.update(x=torch.full((4,),2.),previous_x=torch.full((4,),2.),z=torch.full((4,),.1),forward_velocity=torch.full((4,),1.5))
 e={k:torch.zeros(4,dtype=torch.bool) for k in ('physical_failure_transition','roll_pitch_failure_transition','stuck_transition','yaw_limit_transition','jump_zone_missed_transition','first_apex_success','illegal_contact','timeout_transition','jump_signal','jump_zone_seen','platform_supported','first_platform_hold','first_ditch_flight','first_ditch_clear','ditch_failed')}
 a=torch.zeros(4,4)
 torch.testing.assert_close(phase_u_reward(s,a,a,e,c)[0],torch.zeros(4))
 e['physical_failure_transition'][0]=True;e['roll_pitch_failure_transition'][1]=True;e['yaw_limit_transition'][2]=True;e['timeout_transition'][3]=True;e['ditch_failed'][:]=True
 torch.testing.assert_close(phase_u_reward(s,a,a,e,c)[0],torch.full((4,),-400.))
