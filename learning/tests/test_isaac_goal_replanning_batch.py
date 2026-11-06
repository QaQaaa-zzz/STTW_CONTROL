from types import SimpleNamespace
import numpy as np
import torch
from sttw_control.goal_replanning import IsaacGoalReplanning
from sttw_control.torch_path_reference import TorchPathReference


def make_env(tmp_path, batched):
    cfg=dict(path=dict(projection_backward_m=.05,projection_forward_floor_m=.3,projection_forward_motion_multiplier=2.,projection_forward_padding_m=.1),follower=dict(lookahead_time_s=.7,lookahead_min_speed_m_s=1.5,lookahead_min_m=.8,lookahead_max_m=1.8,wheelbase_m_expected=.408,caster_deg_expected=25.,steer_reference_max_rad=.6,preview_body_x_min_m=.05))
    paths=[];data=[]
    for sign in (1.,-1.):
        row=dict(s=np.arange(4.),xy=np.array([[0.,0.],[1.,0.],[1.,sign],[1.,2*sign]]),heading=np.array([0.,0.,sign*np.pi/2,sign*np.pi/2]),curvature=np.zeros(4),turn_end=1.)
        file=tmp_path/f'route_{sign}.npz';np.savez(file,**row);paths.append(str(file));data.append(row)
    bank=dict(s=data[0]['s'],**{key:np.stack([data[0][key]]*2+[data[1][key]]*2) for key in ('xy','heading','curvature')}) if batched else data[0]
    route=TorchPathReference(bank,cfg);pose=torch.zeros(4,3);speed=torch.ones(4)
    projection=route.project(pose[:,:2],torch.zeros(4),torch.zeros(4))
    middle=SimpleNamespace(projection=projection,follower=route.pursuit(projection['progress'],pose,speed,speed),fault=torch.zeros(4,dtype=torch.bool))
    env=SimpleNamespace(n=4,route=route,config=cfg,device='cpu',middle=middle,plant=SimpleNamespace(observe=lambda:(None,pose,speed)),alive=torch.ones(4,dtype=torch.bool),speed=speed,tick=0)
    config=dict(original_route=[paths[0]]*2+[paths[1]]*2 if batched else paths[0],goal_after_turn_m=1.,arrival_radius_m=.1,period_s=.5,enabled=[True]*4,arrival_reference='original_route')
    return env,pose,config,data


def test_parallel_goals_and_arrival_rebind_use_each_lanes_original_route(tmp_path):
    env,pose,cfg,data=make_env(tmp_path,True)
    planner=IsaacGoalReplanning(env,cfg,tmp_path)
    np.testing.assert_allclose(planner.goals,[[1,1],[1,1],[1,-1],[1,-1]])
    planner.replanned[2]=True;pose[2]=torch.tensor([1.,-1.,-np.pi/2])
    env.route.xy[2,:,1]+=.1
    planner.update()
    np.testing.assert_allclose(env.route.xy[2].numpy(),data[1]['xy'])
    np.testing.assert_allclose(env.route.xy[0].numpy(),data[0]['xy'])
    assert planner.arrived.tolist()==[False,False,True,False]
    assert planner.events[0]['goal_xy']==[1.,-1.]


def test_shared_route_keeps_legacy_goal_and_lane_geometry(tmp_path):
    env,pose,cfg,data=make_env(tmp_path,False)
    planner=IsaacGoalReplanning(env,cfg,tmp_path)
    np.testing.assert_allclose(planner.goal,[1,1])
    np.testing.assert_allclose(planner.goals,[[1,1]]*4)
    np.testing.assert_allclose(env.route.xy.numpy(),np.stack([data[0]['xy']]*4))
