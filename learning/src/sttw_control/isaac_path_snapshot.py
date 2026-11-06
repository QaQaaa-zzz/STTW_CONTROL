"""Lab closed-loop forks at a shared 50 Hz decision boundary.

Uses official link-pose/COM-velocity getters matching the plant reset writer.
PhysX contact/solver caches are not serialized. Keep an unreset continuation
and a restored no-intervention control when interpreting branch differences.
"""
import torch

FIELDS={
 'rollout':('alpha','speed','alive','task_tick','physical_tick','since_reset_tick','previous_reference','previous_command','measurement','switched','switch_tick','zero_middle','zero_lower','preference_progress'),
 'controller':('eso','disturbance','gains'),
 'lower':('frames','mask','previous_applied','yaw_rate'),
 'policy':('filtered','offsets','execution_mode','hold_channels'),
 'middle':('frames','mask','nominal','rates','pose','yaw_rate','fault','domain_exit','obs','latent'),
}


def capture(env,lane):
    """Capture a frozen-parameter case after at least one completed policy step."""
    plant=env.plant;pose=plant.robot.data.root_link_pose_w[lane].clone();pose[:3]-=plant.origin[lane]
    signal_root=plant.roots[lane].clone();signal_root[:3]-=plant.origin[lane]
    state=dict(signal_root=signal_root,root_pose=pose,root_velocity=plant.robot.data.root_com_vel_w[lane].clone(),joint_position=plant.dofs[lane,:,0].clone(),joint_velocity=plant.dofs[lane,:,1].clone())
    for group,names in FIELDS.items():
        obj=env if group=='rollout' else getattr(env,group)
        for name in names:state[group+'.'+name]=getattr(obj,name)[lane].clone()
    for group in ('projection','follower'):
        for name,value in getattr(env.middle,group).items():state['middle.'+group+'.'+name]=value[lane].clone()
    state['policy_tick']=torch.tensor(env.tick,device=env.device);state['middle_tick']=torch.tensor(env.middle.tick,device=env.device)
    if env.randomization is not None:
        state['randomization.steer_offset']=env.randomization.steer_offset[lane].clone()
        state['randomization.servo_gain']=plant.servos.gain[lane].clone()
        for name,buffer in env.randomization.buffers.items():
            ring=buffer._circular_buffer
            state['delay.'+name+'.history']=ring._buffer[:,lane].clone()
            state['delay.'+name+'.pointer']=torch.tensor(ring._pointer,device=env.device)
            state['delay.'+name+'.pushes']=ring._num_pushes[lane].clone()
            state['delay.'+name+'.lag']=buffer.time_lags[lane].clone()
    return state


@torch.no_grad()
def restore(env,state,lanes):
    """Copy the captured lane into selected siblings without stepping physics."""
    plant=env.plant;ids=torch.as_tensor(lanes,device=env.device,dtype=torch.long)
    plant.reset_idx(ids,state={key:state[key] for key in ('root_pose','root_velocity','joint_position','joint_velocity')})
    # Captured velocity is COM velocity; reset_idx's generic writer takes link velocity.
    plant.robot.write_root_com_velocity_to_sim(state['root_velocity'].expand(len(ids),-1),env_ids=ids)
    plant.refresh()
    plant.roots[ids]=state['signal_root'];plant.roots[ids,:3]+=plant.origin[ids]
    for group,names in FIELDS.items():
        obj=env if group=='rollout' else getattr(env,group)
        for name in names:getattr(obj,name)[ids]=state[group+'.'+name]
    for group in ('projection','follower'):
        for name,value in getattr(env.middle,group).items():value[ids]=state['middle.'+group+'.'+name]
    env.tick=int(state['policy_tick']);env.middle.tick=int(state['middle_tick'])

    if env.randomization is not None:
        env.randomization.steer_offset[ids]=state['randomization.steer_offset']
        plant.servos.gain[ids]=state['randomization.servo_gain']
        for name,buffer in env.randomization.buffers.items():
            ring=buffer._circular_buffer
            # Keep the shared ring clock unchanged for all untouched lanes.
            shift=ring._pointer-int(state['delay.'+name+'.pointer'])
            history=torch.roll(state['delay.'+name+'.history'],shifts=shift,dims=0)
            ring._buffer[:,ids]=history[:,None]
            ring._num_pushes[ids]=state['delay.'+name+'.pushes']
            buffer.set_time_lag(state['delay.'+name+'.lag'].expand(len(ids)),ids)
