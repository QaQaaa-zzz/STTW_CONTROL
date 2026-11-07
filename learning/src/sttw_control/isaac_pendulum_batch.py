"""Batched frozen connection trials on the existing Isaac ground/control stack."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from .pendulum_landing import platform_support_mask


@torch.no_grad()
def run_connection_batch(spec, out, plant, ground, policy, history, geometry):
    from isaaclab.utils.math import euler_xyz_from_quat
    n=plant.count;device=plant.device;dt=plant.physics_dt
    candidates=spec['batch_candidates'];steps=round(spec['seconds']/.005)
    active=torch.zeros(n,device=device,dtype=torch.bool);returned=active.clone();alive=~active
    decided=active.clone();entered=active.clone();consumed=active.clone();ready=active.clone()
    landing_ticks=torch.zeros(n,device=device,dtype=torch.long);transition_speed=torch.zeros(n,device=device)
    last=torch.zeros(n,4,device=device);target=last.clone();acceleration=torch.zeros(n,3,device=device)
    previous_velocity=plant.roots[:,7:10].clone();block_velocity=previous_velocity.clone()
    rows=[];valid=[];actions=[];body_rows=[];contact_rows=[];observations=[]
    ground_commands=[];jump_commands=[];arm_commands=[];phases=[]
    events=[{} for _ in candidates];global_tick=0;physics_tick=0;start=time.monotonic()
    requested_speed=torch.tensor([c['speed_m_s'] for c in candidates],device=device)
    requested_alpha=torch.tensor([c['alpha'] for c in candidates],device=device)
    wheels=[plant.bodies.index(name) for name in ('frontwheel','rearwheel')]
    joints=plant.pendulum_servos.indices;ret=spec['landing_return']
    plant.set_steering_position_modes(active)

    def event(lane,kind,m,root,**extra):
        events[lane][kind]=dict(time=global_tick*.005,root=root[lane:lane+1].cpu().tolist(),
            measurement=m[lane:lane+1].cpu().tolist(),position=plant.dofs[lane:lane+1,:,0].cpu().tolist(),
            velocity=plant.dofs[lane:lane+1,:,1].cpu().tolist(),**extra)

    def before_lower(lower_tick,final=False):
        nonlocal global_tick,last,target,active,returned,alive,entered,consumed,physics_tick,decided,ready
        global_tick=block_tick*4+lower_tick;physics_tick=0
        m,pose,speed=plant.observe();root=plant.roots.clone();root[:,:3]-=plant.origin
        distance=spec['platform_front_x']-(geometry.frontmost_x(plant.rigid)-plant.origin[:,0])
        sampled=active | (global_tick%4==0)
        row=torch.cat((torch.full((n,1),global_tick*.005,device=device),root[:,:3],speed[:,None],m[:,:4],plant.dofs[:,joints,0],distance[:,None]),-1)
        rows.append(row);valid.append(alive&sampled);actions.append(last.clone())
        bodies=plant.rigid.clone();bodies[:,:,:3]-=plant.origin[:,None,:]
        body_rows.append(bodies);contact_rows.append(plant.contact.clone())
        alive &= ~sampled | (torch.isfinite(row).all(-1)&(m[:,0].abs()<torch.pi/4)&(root[:,2]>.05))
        if global_tick%4==0:alive &= active | ground.alive
        if final:return
        if global_tick%4==0:
            crossing=alive&~decided&~active&~returned&(root[:,0]>=spec['connection_candidate']['decision_x_m'])
            for lane in crossing.nonzero().flatten().tolist():
                candidate=dict(spec['connection_candidate'],**candidates[lane])
                event(lane,'connection_decision',m,root,decision_stage='exit',candidate=candidate,
                    previous_alpha_duration_s=global_tick*.005,previous_reference=ground.previous_reference[lane:lane+1].cpu().tolist(),
                    previous_alpha=ground.alpha[lane:lane+1].cpu().tolist(),history='retained; no reset')
            ground.speed.copy_(torch.where(crossing,requested_speed,ground.speed))
            ground.alpha.copy_(torch.where(crossing,requested_alpha,ground.alpha));decided|=crossing
            inside=(root[:,0]>=spec['platform_front_x']-1.1)&(root[:,0]<=spec['platform_front_x']+.4)
            consumed|=entered&~inside;entered|=inside;signal=(inside&~consumed).float()
            obs=torch.cat((history.frames.flatten(1),signal[:,None]),-1) if global_tick==0 and spec.get('zero_initial_history',False) else history.advance(root,plant.dofs[:,:,0],plant.dofs[:,:,1],acceleration,distance,last,signal)
            observations.append(obs.clone())
            switching=alive&~active&~returned&(root[:,0]>=spec['handoff_x'])
            ids=switching.nonzero().flatten()
            if len(ids):
                for lane in ids.tolist():event(lane,'handoff',m,root,distance=distance[lane:lane+1].cpu().tolist())
                if spec.get('stop_at_ground_handoff',False):
                    alive &= ~switching
                else:
                    plant.set_pendulum_locked(False,ids);active|=switching;plant.set_steering_position_modes(active)
            if spec.get('stop_at_ground_handoff',False):return
            action,proposed=policy.infer(obs,plant.dofs[:,plant.joints.index('knee_joint'),0])
            last=torch.where(active[:,None],action,last);target=torch.where(active[:,None],proposed,target)
        _,pitch,_=euler_xyz_from_quat(plant.robot.data.root_link_quat_w)
        pitch=torch.atan2(torch.sin(pitch),torch.cos(pitch))
        wp=plant.rigid[:,wheels,:3]-plant.origin[:,None,:]
        settled=platform_support_mask(wp,plant.contact[:,wheels],m[:,0],pitch,spec['platform_front_x'],spec['platform_back_x'])&active&alive
        landing_ticks.copy_(torch.where(settled,landing_ticks+1,0))
        newly_ready=active&alive&~ready&(landing_ticks*.005>=ret['support_s'])
        ready|=newly_ready;transition_speed.copy_(torch.where(newly_ready,speed,transition_speed))
        for lane in newly_ready.nonzero().flatten().tolist():
            events[lane]['landing_ready']=dict(time=global_tick*.005,speed=speed[lane:lane+1].cpu().tolist())
        if ret.get('minimum_handoff_speed_m_s',0)>0:
            advancing=active&ready&alive
            transition_speed.copy_(torch.where(advancing,(transition_speed+ret['acceleration_m_s2']*.005).clamp_max(ret['speed_m_s']),transition_speed))
            target[:,1]=torch.where(advancing,transition_speed/plant.wheel_radius_m,target[:,1])
        handing=active&ready&alive&(speed>=ret.get('minimum_handoff_speed_m_s',0))&(global_tick%4==0)
        ids=handing.nonzero().flatten()
        if len(ids):
            plant.set_pendulum_locked(True,ids);active&=~handing;returned|=handing
            plant.set_steering_position_modes(active)
            ground.resume_ground_lanes(ids,ret['speed_m_s'],ret.get('alpha',spec['ground_alpha']))
            for lane in ids.tolist():event(lane,'landing_handoff',m,root,controller_state='Selected lane ground controller history initialized; no physics reset',arm='locked at current posture')

    def physics_step(command):
        nonlocal physics_tick,previous_velocity,acceleration
        t=global_tick*.005+physics_tick*dt;physics_tick+=1
        actual=torch.where(active[:,None],torch.stack((torch.zeros(n,device=device),target[:,1]),-1),command)
        actual=torch.where(alive[:,None],actual,0.)
        phase=torch.where(active,torch.where(ready,2,1),torch.where(returned,3,0))
        clock=torch.full((n,1),t,device=device)
        ground_commands.append(torch.cat((clock,actual,alive[:,None],phase[:,None]),-1))
        jump_commands.append(torch.cat((clock,phase[:,None],target.clone()),-1));phases.append(phase)
        arm_before=torch.cat((clock,target[:,2:],plant.dofs[:,joints,0],plant.dofs[:,joints,1]),-1)
        arm_target=torch.where(active[:,None],target[:,2:],plant.pendulum_target)
        contact=plant.step(actual,pendulum_target=arm_target,steering_position_target=target[:,:1])
        velocity=plant.robot.data.root_link_lin_vel_w
        acceleration=(velocity-previous_velocity)/dt;previous_velocity=velocity.clone()
        arm_torque=plant.last_pendulum_torque[:,joints] if not plant.pendulum_is_locked else torch.zeros(n,2,device=device)
        arm_commands.append(torch.cat((arm_before,arm_torque,(alive&active)[:,None]),-1))
        return contact

    ground.before_lower_step=before_lower;ground.physics_step_override=physics_step
    for block_tick in range(round(spec['seconds']/.02)):
        block_velocity=plant.roots[:,7:10].clone()
        ground.step()
        velocity=plant.robot.data.root_link_lin_vel_w
        acceleration=torch.where(active[:,None],acceleration,(velocity-block_velocity)/.02)
        previous_velocity=velocity.clone()
        if not bool(alive.any()):break
    block_tick+=1;before_lower(0,final=True)
    elapsed=time.monotonic()-start
    arrays=dict(trace=torch.stack(rows).cpu().numpy(),valid=torch.stack(valid).cpu().numpy(),action=torch.stack(actions).cpu().numpy(),observations=torch.stack(observations).cpu().numpy())
    body=torch.stack(body_rows).cpu().numpy();contacts=torch.stack(contact_rows).cpu().numpy()
    commands=torch.stack(ground_commands).cpu().numpy();jump=torch.stack(jump_commands).cpu().numpy();arms=torch.stack(arm_commands).cpu().numpy();phase=torch.stack(phases).cpu().numpy()
    parent=json.loads((out/'protocol.json').read_text());controller=json.loads((out/'ground_controller.json').read_text());cases=[]
    for lane,candidate in enumerate(candidates):
        folder=out/f'lane_{lane:02d}';folder.mkdir()
        lane_spec=dict(spec);lane_spec.pop('batch_candidates');lane_spec.update(output=str(folder),skills=[spec['skills'][lane]],connection_candidate=dict(spec['connection_candidate'],**candidate))
        protocol=dict(parent,spec=lane_spec,policies=[parent['policies'][lane]],batch_execution=dict(parent_output=str(out),lane=lane,count=n,shared_elapsed_s=elapsed,scope='Experimental mixed-phase GPU batch; requires paired validation before adoption'))
        (folder/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(folder/'ground_controller.json').write_text(json.dumps(controller,indent=2)+'\n')
        (folder/'config.json').write_text(json.dumps(lane_spec,indent=2)+'\n')
        np.savez_compressed(folder/'ground_route.npz',**dict(np.load(out/'ground_route.npz')))
        # Preserve per-lane recorded terminal sample and original phase-dependent cadence.
        selected=arrays['valid'][:,lane]
        trace=arrays['trace'][selected,lane:lane+1];mask=np.ones((len(trace),1),dtype=bool)
        np.savez_compressed(folder/'trajectory.npz',trace=trace,valid=mask,action=arrays['action'][selected,lane:lane+1],observations=arrays['observations'][:,lane:lane+1],columns=['time','x','y','z','speed','roll','roll_rate','steer','steer_rate','hip','knee','obstacle_relative_x'])
        np.savez_compressed(folder/'body_contact.npz',time=trace[:,0,0],body_state=body[selected,lane:lane+1],contact=contacts[selected,lane:lane+1],body_names=plant.bodies,valid=mask)
        g=(phase[:,lane]==0)|(phase[:,lane]==3);j=~g
        np.savez_compressed(folder/'executed_ground_commands.npz',trace=commands[g,lane:lane+1],columns=['time','steer_rate_rad_s','rear_axis_speed_rad_s','alive_before_step','phase_0_ground_3_return'])
        np.savez_compressed(folder/'executed_jump_targets.npz',trace=jump[j,lane:lane+1],columns=['time','phase_1_jump_2_landing_transition','steering_position_rad','rear_axis_speed_rad_s','hip_position_rad','knee_position_rad'])
        np.savez_compressed(folder/'arm_commands.npz',trace=arms[j,lane:lane+1],columns=['time','hip_target','knee_target','hip_position','knee_position','hip_velocity','knee_velocity','hip_command_torque','knee_command_torque','alive_before_step'])
        for name,value in events[lane].items():(folder/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
        tr=trace[:,0];case=dict(skill=spec['skills'][lane],duration_s=float(tr[-1,0]),physical_termination=not bool(alive[lane]) and not (spec.get('stop_at_ground_handoff',False) and 'handoff' in events[lane]),ground_handoff_reached='handoff' in events[lane],peak_root_z=float(tr[:,3].max()),final_xyz=tr[-1,1:4].tolist(),peak_abs_roll_deg=float(np.rad2deg(abs(tr[:,5])).max()),full_jump_success='requires original landing and chain analysis')
        result=dict(status='complete',elapsed_s=elapsed,elapsed_scope='Shared whole-batch elapsed, not additive per lane',physics_lane_steps=block_tick*round(.02/dt),landing_return_executed=bool(returned[lane]),ground_controller='H75+ECBC (zero lower residual)',cases=[case])
        (folder/'results.json').write_text(json.dumps(result,indent=2)+'\n');cases.append(dict(lane=lane,candidate=candidate,output=str(folder)))
    summary=dict(status='complete',elapsed_s=elapsed,physics_lane_steps=block_tick*round(.02/dt)*n,optimizer_updates=0,cases=cases)
    (out/'batch_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary
