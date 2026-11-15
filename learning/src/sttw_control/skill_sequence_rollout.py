"""Causal ground->RB->ground predictions from one real initial state."""
import torch
from .playground_history_model import GroundPredictionState
from .playground_prediction import rotate,euler,BodyEnvelope
from .skill_history_bridge import bridge_features


def handoff_joint_errors(qpos,qvel,home,schema):
    """Necessary lock checks; passing does not certify contact/body readiness."""
    joints=['hip_joint','knee_joint']
    qi=[schema['qpos_names'].index(n) for n in joints]
    vi=[schema['qvel_names'].index(n) for n in joints]
    return (qpos[...,qi]-home).abs().amax(-1),qvel[...,vi].abs().amax(-1)


def handoff_body_ready(qpos,qvel,contact,speed_request,hand,schema):
    """Predicted support/body conditions; qvel roll rate approximates the gyro."""
    angles=euler(qpos[:,3:7]);roll,pitch,yaw=angles.unbind(-1)
    supported=contact[:,2 if hand.get('require_both_far_wheels',False) else 1]>=.5
    body=(roll.abs()<.35)&(pitch.abs()<.45)&(qvel[:,2].abs()<1.5)
    if hand.get('landing_control')=='native':
        body &= (roll.abs()<.08)&(pitch.abs()<.15)&(qvel[:,2].abs()<.4)&(qvel[:,3].abs()<.3)
    speed=qvel[:,0]*yaw.cos()+qvel[:,1]*yaw.sin()
    body &= (qvel[:,5].abs()<hand.get('handoff_yaw_rate_max_rad_s',float('inf')))&(qvel[:,schema['steer_dof']].abs()<hand.get('handoff_steer_rate_max_rad_s',float('inf')))
    body &= (speed-speed_request).abs()<=hand.get('handoff_speed_error_max_m_s',float('inf'))
    return supported,body


class HandoffEventGuard:
    """GPU trace checks for the frozen handoff, not a contact certificate."""
    def __init__(self,physics,schema,hand,world,speeds):
        self.schema=schema;self.hand=hand;self.speeds=speeds
        self.home=speeds.new_tensor(physics.key('initial_state').qpos[[schema['qpos_names'].index(n) for n in ['hip_joint','knee_joint']]])
        self.geometry=BodyEnvelope(physics,device=speeds.device,dimensions=3)
        geoms=[g for g in range(physics.ngeom) if physics.geom_bodyid[g]>0 and (physics.geom_contype[g] or physics.geom_conaffinity[g])]
        self.wheels=[geoms.index(physics.geom(n).id) for n in ['rearwheel_collision','frontwheel_collision']]
        self.far_x=world['terrain']['front_x_m']+world['terrain']['width_m']

    def __call__(self,values):
        shape=values['qpos'].shape[:2];q=values['qpos'].reshape(-1,12);v=values['qvel'].reshape(-1,11)
        contact=values['contact'].reshape(-1,3);speeds=self.speeds[:,None].expand(shape).reshape(-1)
        error,rate=handoff_joint_errors(q,v,self.home,self.schema)
        lock=(error<self.hand.get('lock_position_tolerance_rad',.08))&(rate<self.hand.get('lock_velocity_tolerance_rad_s',.5))
        support,body=handoff_body_ready(q,v,contact,speeds,self.hand,self.schema)
        pose,_=native_to_ground(q,v,self.schema);_,upper=self.geometry(pose)
        far=upper[:,self.wheels,0]>=self.far_x
        geometry=far.all(-1) if self.hand.get('require_both_far_wheels',False) else far.any(-1)
        self.diagnostics={k:x.reshape(shape) for k,x in dict(handoff_joints_ready=lock,handoff_joint_error_rad=error,handoff_joint_speed_rad_s=rate,handoff_support_ready=support,handoff_body_ready=body,handoff_support_geometry=geometry).items()}
        return (lock&support&body&geometry).reshape(shape)


def native_to_ground(qpos,qvel,schema):
    order=[schema['qpos_names'].index(n) for n in ['steering_joint','frontwheel_joint','rearwheel_joint','hip_joint','knee_joint']]
    inv=qpos[:,3:7]*qpos.new_tensor([1,-1,-1,-1])
    pose=torch.cat((qpos[:,:7],qpos[:,order]),-1)
    velocity=torch.cat((rotate(inv,qvel[:,:3]),qvel[:,3:6],qvel[:,[schema['steer_dof'],schema['front_dof'],schema['rear_dof']]]),-1)
    return pose,velocity


def ground_to_native(pose,velocity,schema):
    names=['steering_joint','frontwheel_joint','rearwheel_joint','hip_joint','knee_joint']
    qpos=pose.new_zeros(len(pose),12);qvel=velocity.new_zeros(len(pose),11)
    qpos[:,:7]=pose[:,:7]
    for j,name in enumerate(names):qpos[:,schema['qpos_names'].index(name)]=pose[:,7+j]
    qvel[:,:3]=rotate(pose[:,3:7],velocity[:,:3]);qvel[:,3:6]=velocity[:,3:6]
    qvel[:,[schema['steer_dof'],schema['front_dof'],schema['rear_dof']]]=velocity[:,6:9]
    # STTW holds hip/knee position; this ground model assumes zero locked rates.
    return qpos,qvel


def predict_skill_sequence(approach,bridge,jump,departure,qpos,qvel,initial_history,
                           schema,path_s,path_xy,path_heading,path_curvature,speed,
                           jump_x,context,ground_steps=250,jump_steps=125,departure_steps=250,typical=None,diagnostic_entry=None,handoff_reset=None,handoff_guard=None):
    """C uses initial real state only; I uses TRAIN-only typical interfaces.

    diagnostic_entry is an explicitly offline true-interface intervention, never
    a causal planner mode. No files or runtime states are modified.

    Constant contact=[supported,not far,not both far] is the declared flat
    approach entry contract. Estimated histories are never sent to real controls.
    """
    pose,velocity=native_to_ground(qpos,qvel,schema)
    state=approach.initialize(pose,velocity,velocity,initial_history[:,:381])
    state.elapsed_steps=torch.zeros(len(pose),device=pose.device,dtype=torch.long)
    found=pose[:,0]>=jump_x;poses=[pose];velocities=[velocity];domains=[]
    for _ in range(ground_steps):
        prediction=approach.rollout_state(state,path_s,path_xy,path_heading,path_curvature,1,speed_request=speed)
        domains.append(prediction['domain'][:,0]);proposed=prediction['next_state']
        state=GroundPredictionState(**{name:torch.where(found[:,None],getattr(state,name),getattr(proposed,name)) for name in ['pose','velocity','previous_velocity','hidden']},elapsed_steps=torch.where(found,state.elapsed_steps,proposed.elapsed_steps))
        found=found|(state.pose[:,0]>=jump_x);poses.append(state.pose);velocities.append(state.velocity)
    domains.append(prediction['domain'][:,1])
    history=bridge(bridge_features(state,initial_history))
    entry_q,entry_v=ground_to_native(state.pose,state.velocity,schema)
    if typical is not None:
        entry_q,entry_v,history=typical['entry_qpos'],typical['entry_qvel'],typical['entry_history']
    if diagnostic_entry is not None:
        entry_q=diagnostic_entry.get('qpos',entry_q)
        entry_v=diagnostic_entry.get('qvel',entry_v)
        history=diagnostic_entry.get('history',history)
    contact=entry_q.new_tensor([1.,0.,0.]).expand(len(pose),-1)
    phase=entry_q.new_tensor([0.,1.,0.,0.,0.]).expand(len(pose),-1)
    rb_state=jump.initialize(entry_q,entry_v,history[:,:381],history[:,381:],contact,phase,context)
    rb=jump.rollout_until(rb_state,context,jump_steps,stop_guard=handoff_guard)
    terminal=rb['next_state'];tail_pose,tail_velocity=native_to_ground(terminal.qpos,terminal.qvel,schema)
    lane=torch.arange(len(pose),device=pose.device);index=rb['event_step']-1
    previous_q=torch.where((index>0)[:,None],rb['qpos'][lane,(index-1).clamp_min(0)],entry_q)
    previous_v=torch.where((index>0)[:,None],rb['qvel'][lane,(index-1).clamp_min(0)],entry_v)
    _,previous_velocity=native_to_ground(previous_q,previous_v,schema)
    tail_history=terminal.history[:,:381]
    if handoff_reset is not None:tail_history=handoff_reset(terminal.qpos,terminal.qvel,tail_history,speed)
    if typical is not None:
        tail_pose,tail_velocity=native_to_ground(typical['departure_qpos'],typical['departure_qvel'],schema)
        _,previous_velocity=native_to_ground(typical['departure_previous_qpos'],typical['departure_previous_qvel'],schema)
        tail_history=typical['departure_history'][:,:381]
    end=departure.initialize(tail_pose,tail_velocity,previous_velocity,tail_history)
    tail=departure.rollout_state(end,path_s,path_xy,path_heading,path_curvature,departure_steps,speed_request=speed)
    return dict(approach_domain=torch.stack(domains,1),approach_pose=torch.stack(poses,1),approach_velocity=torch.stack(velocities,1),entry_found=found,entry_time_s=state.elapsed_steps*.02,entry_qpos=entry_q,entry_qvel=entry_v,entry_history=history,
                jump=rb,departure=tail,departure_history=tail_history,handoff_time_s=state.elapsed_steps*.02+rb['duration_s'],supported=found&rb['event_found'])
