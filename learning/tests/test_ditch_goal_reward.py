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
