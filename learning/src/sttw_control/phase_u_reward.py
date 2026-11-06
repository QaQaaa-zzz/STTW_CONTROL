"""Batched Torch port of DVGC/JIT Phase U rewards.py, with original 21 terms.

Original equations and precedence retained; only the final action axis is reduced.
Inputs use clean physical signals, never randomized policy observations.
"""
import torch


def phase_u_reward(state, action, last_action, events, config):
    """Return clipped total, unclipped total, and per-lane signed components.

    state/events are dicts of tensors with matching leading batch dimensions.
    Config is the original resolved reward dictionary, without PPO reward scaling.
    """
    c=config;zero=torch.zeros_like(state['roll'])
    degrees=lambda name:torch.rad2deg(state[name].abs())
    r=degrees('roll');p=degrees('pitch');y=degrees('yaw');z=state['z']
    roll=torch.where(r<=5,1-r/5,-.1*(r-5))
    pitch=torch.where(p<=3,1-p/3*.1,torch.where(p<=8,.9-(p-3)/5*.4,
        torch.where(p<=10,.5-(p-8)/2*.5,torch.where(p<=15,-.1*(p-10),-.5-.5*(p-15)))))
    yaw=torch.where(y<=3,1-y/3*.1,torch.where(y<=8,.9-(y-3)/5*.4,
        torch.where(y<=15,.5-(y-8)/7*.3,torch.where(y<=25,.2-(y-15)/10*.15,(.05-(y-25)*.002).clamp_min(0)))))
    rising=1+(z-c['jump_reward_min_height'])/(c['peak_reward_height']-c['jump_reward_min_height'])*.5
    falling=1.5*(1-(z-c['peak_reward_height'])/(c['max_beneficial_height']-c['peak_reward_height'])*.6)
    height=torch.where(z>=c['jump_reward_min_height'],torch.where(z<=c['peak_reward_height'],rising,
        torch.where(z<=c['max_beneficial_height'],falling,.4)),0.)
    low=((c['jump_reward_min_height']-z)/(c['jump_reward_min_height']-c['low_height_penalty_full_at'])).clamp(0,1)
    delta=action-last_action
    physical=events['physical_failure_transition'].bool()
    roll_pitch=events['roll_pitch_failure_transition'].bool()
    stuck=events['stuck_transition'].bool()&~physical
    yaw_failure=events['yaw_limit_transition'].bool()&~physical&~stuck
    missed=events['jump_zone_missed_transition'].bool()&~roll_pitch
    components=dict(
        roll=c['roll_coeff']*roll,pitch=c['pitch_coeff']*pitch,yaw=c['yaw_coeff']*yaw,
        speed=c['speed_coeff']*torch.exp(-.5*((state['forward_velocity']-c['desired_velocity'])/c['speed_sigma']).square()),
        survival=zero+c['survival_reward'],height=c['height_coeff']*height*events['jump_signal'],
        low_height=-c['low_height_penalty']*low*events['jump_signal'],
        action_smoothness=-c['action_coeff']*c['action_smoothness_scale']*delta.square().sum(-1)
            -c['steering_action_diff_penalty']*delta[...,0].square()-c['rear_wheel_action_diff_penalty']*delta[...,1].square(),
        action_magnitude=-c['action_coeff']*c['action_magnitude_scale']*action.abs().pow(1.5).sum(-1)
            -c['steering_magnitude_penalty']*action[...,0].abs().pow(1.5),
        roll_rate=zero,pitch_rate=-c['pitch_angular_velocity_coeff']*.125*state['pitch_rate'].square(),yaw_rate=zero,
        joint_energy=-c['joint_energy_penalty_coeff']*.02*((state['hip_force']*state['hip_velocity']).abs()+(state['knee_force']*state['knee_velocity']).abs()),
        apex_success=c['apex_success_bonus']*events['first_apex_success'],
        illegal_contact=-c['illegal_contact_penalty']*events['illegal_contact'],
        physical_failure=-c['physical_failure_penalty']*physical,
        roll_pitch_failure=-c['roll_pitch_failure_penalty']*roll_pitch,
        jump_zone_missed=-c['jump_zone_missed_penalty']*missed,
        stuck=-c['stuck_penalty']*stuck,yaw_limit=-c['yaw_limit_penalty']*yaw_failure,
        timeout=-c['timeout_penalty']*events['timeout_transition'])
    if 'traversal' in c:
        traversal=c['traversal']
        active=events['jump_zone_seen'].bool()&(state['x']<traversal['end_x_m'])
        velocity=state['forward_velocity']
        fraction=(velocity/traversal['minimum_forward_m_s']).clamp(0,1)
        height_factor=traversal['stationary_height_fraction']+(1-traversal['stationary_height_fraction'])*fraction
        components['height']=components['height']*torch.where(active,height_factor,torch.ones_like(fraction))
        components['apex_success']=components['apex_success']*torch.where(active,fraction,torch.ones_like(fraction))
        components['traversal_low_speed']=-traversal['low_speed_weight']*(1-fraction).square()*active
        components['traversal_forward']=traversal['forward_weight']*(velocity/traversal['target_forward_m_s']).clamp(-1,1)*active
        deceleration=((state['previous_forward_velocity']-velocity)/state['policy_dt']).clamp_min(0)
        components['traversal_deceleration']=-traversal['deceleration_weight']*(deceleration/traversal['deceleration_scale_m_s2']).clamp(max=1)*active*(velocity<traversal['target_forward_m_s'])
    total=sum(components.values())
    return total.clamp(c['total_min'],c['total_max']),total,components


def phase_u_episode_reward(clipped_reward, previous_return, stuck, yaw_limit, config):
    """Original env.py failure-return correction, applied before PPO scaling.

    Stuck/yaw terminal transitions fix the whole unscaled episode return, not
    just the last reward. This override may lie outside component clipping.
    """
    target=config['failed_episode_return']
    reward=clipped_reward if target is None else torch.where(stuck|yaw_limit,target-previous_return,clipped_reward)
    return reward,previous_return+reward,reward-clipped_reward
