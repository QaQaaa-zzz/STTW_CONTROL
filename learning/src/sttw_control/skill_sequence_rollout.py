"""Causal ground->RB->ground predictions from one real initial state."""
import torch
from .playground_history_model import GroundPredictionState
from .playground_prediction import rotate
from .skill_history_bridge import bridge_features


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
                           jump_x,context,ground_steps=250,jump_steps=125,departure_steps=250):
    """No trajectory/status/future data accepted by this interface.

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
    contact=entry_q.new_tensor([1.,0.,0.]).expand(len(pose),-1)
    phase=entry_q.new_tensor([0.,1.,0.,0.,0.]).expand(len(pose),-1)
    rb_state=jump.initialize(entry_q,entry_v,history[:,:381],history[:,381:],contact,phase,context)
    rb=jump.rollout_until(rb_state,context,jump_steps)
    terminal=rb['next_state'];tail_pose,tail_velocity=native_to_ground(terminal.qpos,terminal.qvel,schema)
    lane=torch.arange(len(pose),device=pose.device);index=rb['event_step']-1
    previous_q=torch.where((index>0)[:,None],rb['qpos'][lane,(index-1).clamp_min(0)],entry_q)
    previous_v=torch.where((index>0)[:,None],rb['qvel'][lane,(index-1).clamp_min(0)],entry_v)
    _,previous_velocity=native_to_ground(previous_q,previous_v,schema)
    end=departure.initialize(tail_pose,tail_velocity,previous_velocity,terminal.history[:,:381])
    tail=departure.rollout_state(end,path_s,path_xy,path_heading,path_curvature,departure_steps,speed_request=speed)
    return dict(approach_domain=torch.stack(domains,1),approach_pose=torch.stack(poses,1),approach_velocity=torch.stack(velocities,1),entry_found=found,entry_time_s=state.elapsed_steps*.02,entry_qpos=entry_q,entry_qvel=entry_v,entry_history=history,
                jump=rb,departure=tail,handoff_time_s=state.elapsed_steps*.02+rb['duration_s'],supported=found&rb['event_found'])
