"""Batched Torch translation of original Phase U semantics.py.

Keep event and terminal evaluation separate: original env evaluates preliminary
termination, advances events, then evaluates final termination at the old step.
Apex is a one-shot bonus event, not terminal success or landing certification.
"""
import torch


def initial_events(x,config):
    e=config['events'];inside=(x>=e['jump_zone_x_min'])&(x<=e['jump_zone_x_max'])
    false=torch.zeros_like(x,dtype=torch.bool);ticks=torch.zeros_like(x,dtype=torch.long)
    return dict(jump_signal=inside,jump_zone_seen=inside,jump_zone_consumed=x>e['jump_zone_x_max'],
        ascending_seen=false.clone(),height_seen=false.clone(),apex_seen=false.clone(),
        stuck_anchor_x=x.clone(),stuck_ticks=ticks.clone(),stuck=false.clone(),episode_step=ticks)


def advance_events(previous,signals,config):
    e=config['events'];x=signals['x'];z=signals['z'];false=torch.zeros_like(x,dtype=torch.bool)
    inside=(x>=e['jump_zone_x_min'])&(x<=e['jump_zone_x_max'])
    safe_x=torch.where(torch.isfinite(x),x,previous['stuck_anchor_x'])
    consumed=previous['jump_zone_consumed']|(previous['jump_signal']&~inside)|(x>e['jump_zone_x_max'])
    jump=inside&~previous['jump_zone_consumed'];seen=previous['jump_zone_seen']|jump
    ascending=previous['ascending_seen']|(seen&(signals['vertical_velocity']>=e['min_ascent_velocity']))
    height=previous['height_seen']|(seen&(z>=e['apex_height']))
    apex=seen&ascending&height&(signals['vertical_velocity']<=-e['min_descent_velocity'])&~signals['physical_failure']
    ticks=torch.zeros_like(previous['stuck_ticks']);anchor=safe_x
    if e.get('stuck_forward_velocity_threshold') is not None:
        stuck=previous['stuck']|(jump&~height&(signals['forward_velocity']<=e['stuck_forward_velocity_threshold']))
    elif e.get('stuck_window_steps') is None:
        stuck=false
    else:
        monitor=jump&~height&(z<config['reward']['jump_reward_min_height'])
        progress=x-previous['stuck_anchor_x']>=e['stuck_min_progress']
        reset=~monitor|progress
        anchor=torch.where(reset,safe_x,previous['stuck_anchor_x'])
        ticks=torch.where(reset,ticks,previous['stuck_ticks']+1)
        stuck=previous['stuck']|(monitor&(ticks>=e['stuck_window_steps']))
    return dict(jump_signal=jump,jump_zone_seen=seen,jump_zone_consumed=consumed,
        ascending_seen=ascending,height_seen=height,apex_seen=previous['apex_seen']|apex,
        stuck_anchor_x=anchor,stuck_ticks=ticks,stuck=stuck,episode_step=previous['episode_step']+1)


def classify_terminal(inputs,config):
    p=config['physical_limits'];e=config['events'];false=torch.zeros_like(inputs['roll'],dtype=torch.bool)
    roll=inputs['roll'].abs()>p['max_abs_roll'];pitch=inputs['pitch'].abs()>p['max_abs_pitch']
    failures=(inputs['nonfinite'],roll,pitch,inputs['illegal_contact']&p['terminate_on_prohibited_contact'],inputs['backward_exit'])
    physical=false.clone()
    for f in failures:physical|=f
    raw_yaw=false if p.get('max_abs_yaw') is None else inputs['yaw'].abs()>p['max_abs_yaw']
    raw_stuck=false if e.get('stuck_window_steps') is None and e.get('stuck_forward_velocity_threshold') is None else inputs['stuck']
    horizon=inputs['episode_step']>=config['ppo']['episode_horizon']-1
    missed=config['schema'].endswith('_v4')&(~inputs['jump_zone_seen']&(physical|horizon))
    stuck=raw_stuck&~physical;yaw=raw_yaw&~physical&~stuck
    conditions=failures+(missed,stuck,yaw);codes=(2,3,4,5,7,11,8,10)
    terminated=physical|missed|stuck|yaw;timeout=horizon&~terminated
    code=torch.zeros_like(inputs['episode_step'])
    for f,c in reversed(tuple(zip(conditions,codes))):code=torch.where(f,c,code)
    code=torch.where(terminated,code,torch.where(timeout,9,0))
    return dict(terminated=terminated,truncated=timeout,success=false,physical_failure=physical,
        roll_limit=roll,pitch_limit=pitch,jump_zone_missed=missed,stuck=stuck,yaw_limit=yaw,timeout=timeout,end_code=code)
