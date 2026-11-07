import torch
from sttw_control.phase_u_reward import phase_u_episode_reward


def test_failed_episode_total_is_fixed_without_reclipping_terminal_reward():
    # Matches original Phase U: terminal correction can exceed [-400, 50].
    raw=torch.tensor([-400.,-100.,5.]);previous=torch.tensor([700.,-300.,20.])
    reward,total,override=phase_u_episode_reward(raw,previous,torch.tensor([True,False,False]),torch.tensor([False,True,False]),{'failed_episode_return':-100.})
    torch.testing.assert_close(reward,torch.tensor([-800.,200.,5.]))
    torch.testing.assert_close(total,torch.tensor([-100.,-100.,25.]))
    torch.testing.assert_close(override,torch.tensor([-400.,300.,0.]))


def test_traversal_reward_preserves_momentum_instead_of_stationary_height():
    import json
    from pathlib import Path
    from sttw_control.phase_u_reward import phase_u_reward
    c=json.loads(Path('learning/configs/isaac_phase_u_task_traversal.json').read_text())['task']['reward']
    state={k:torch.zeros(4) for k in ('roll','pitch','yaw','pitch_rate','hip_velocity','knee_velocity','hip_force','knee_force')}
    state.update(x=torch.full((4,),3.),z=torch.full((4,),.5),forward_velocity=torch.tensor([0.,1.,1.8,2.]))
    state.update(previous_forward_velocity=state['forward_velocity'].clone(),policy_dt=.02)
    events={k:torch.zeros(4,dtype=torch.bool) for k in ('physical_failure_transition','roll_pitch_failure_transition','stuck_transition','yaw_limit_transition','jump_zone_missed_transition','first_apex_success','illegal_contact','timeout_transition')}
    events.update(jump_signal=torch.ones(4),jump_zone_seen=torch.ones(4,dtype=torch.bool))
    action=torch.zeros(4,4)
    reward,_,parts=phase_u_reward(state,action,action,events,c)
    assert reward[0]<reward[1]<reward[2]<=reward[3]
    assert parts['traversal_low_speed'][0]==-8
    assert parts['height'][0]==15 and parts['height'][2]==60
    state['previous_forward_velocity']+=.04
    slowed=phase_u_reward(state,action,action,events,c)[2]['traversal_deceleration']
    torch.testing.assert_close(slowed,torch.tensor([-2.,-2.,-2.,0.]))
    original={k:v for k,v in c.items() if k!='traversal'}
    for before in (True,False):
        events['jump_zone_seen'].fill_(not before)
        state['x'].fill_(3. if before else 4.2)
        shaped=phase_u_reward(state,action,action,events,c)[0]
        base=phase_u_reward(state,action,action,events,original)[0]
        torch.testing.assert_close(shaped,base)


def test_disabled_failed_return_overwrite_retains_dense_cost():
    r,total,override=phase_u_episode_reward(torch.tensor([-15.]),torch.tensor([80.]),torch.tensor([True]),torch.tensor([False]),{'failed_episode_return':None})
    assert r.item()==-15 and total.item()==65 and override.item()==0


def test_landing_reward_requires_support_and_prefers_continued_motion():
    import json
    from pathlib import Path
    from sttw_control.phase_u_reward import phase_u_reward
    c=json.loads(Path('learning/configs/isaac_phase_u_task_platform.json').read_text())['task']['reward']
    state={k:torch.zeros(4) for k in ('roll','pitch','yaw','pitch_rate','hip_velocity','knee_velocity','hip_force','knee_force')}
    state.update(x=torch.full((4,),5.),z=torch.full((4,),.3),forward_velocity=torch.tensor([1.5,1.5,0.,1.5]),previous_forward_velocity=torch.tensor([1.5,1.5,0.,1.5]),policy_dt=.02)
    events={k:torch.zeros(4,dtype=torch.bool) for k in ('physical_failure_transition','roll_pitch_failure_transition','stuck_transition','yaw_limit_transition','jump_zone_missed_transition','first_apex_success','illegal_contact','timeout_transition','jump_signal','jump_zone_seen')}
    events.update(platform_supported=torch.tensor([False,True,True,True]),first_platform_hold=torch.tensor([False,False,False,True]))
    action=torch.zeros(4,4);parts=phase_u_reward(state,action,action,events,c)[2]
    assert parts['platform_support'][0]==0
    assert parts['platform_support'][1]>parts['platform_support'][2]>0
    torch.testing.assert_close(parts['platform_hold'],torch.tensor([0.,0.,0.,40.]))


def test_online_platform_support_rejects_missing_contact_and_outside_platform():
    from sttw_control.pendulum_landing import platform_support_mask
    p=torch.tensor([[[4.,0.,.23],[4.3,0.,.23]]]).repeat(4,1,1)
    f=torch.zeros(4,2,3);f[:,:,2]=10
    f[1,0,2]=0;p[2,1,1]=1.1;p[3,:,0]=3.
    mask=platform_support_mask(p,f,torch.zeros(4),torch.zeros(4),3.6,7.6)
    assert mask.tolist()==[True,False,False,False]
