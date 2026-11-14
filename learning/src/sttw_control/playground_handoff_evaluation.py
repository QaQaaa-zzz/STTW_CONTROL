"""GPU-resident batched STTW/RB/STTW inference on the native Playground chassis."""
import json
import shutil
import sys
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jp
import torch
import mujoco
from mujoco import mjx
from mujoco.mjx.warp.types import DATA_NON_VMAP
from mujoco_playground._src import mjx_env
from jit_dvgc.action_mapping import map_action
from jit_dvgc.geometry import build_geometry_contract, extract_geometry
from jit_dvgc.observation import initial_history, advance_history, observable_frame, actor_observation, ObservableGeometry
from .torch_controller import TorchController
from .torch_path_policy import TorchPathPolicy
from .torch_path_reference import TorchPathReference, wrap
from .torch_middle_control import TorchMiddleControl
from .playground_adapters import GroundAdapter


def evaluate_handoff(model, mm, idx, cfg, bundle, hcfg, identity, controller_config, skills, args, brake_ids):
    source=args.output/'source';source.mkdir()
    for name in ['playground_handoff_evaluation.py','torch_controller.py','torch_path_reference.py','torch_middle_control.py','playground_adapters.py']:
        shutil.copy2(Path(__file__).with_name(name),source/name)
    shutil.copy2(sys.argv[0],source/Path(sys.argv[0]).name)
    started=time.perf_counter();device='cuda:0';hand=cfg['ground_handoff'];terrain=cfg['ditch_terrain']
    to_torch=torch.utils.dlpack.from_dlpack
    to_jax=lambda x:jax.dlpack.from_dlpack(x.detach().contiguous())
    tensor=lambda x:torch.as_tensor(x,device=device)
    variants=hand.get('handoff_variants',[{}])
    lanes=[(r,m,i) for i in range(len(variants)) for r in args.routes for m in args.modes];n=len(lanes)
    event=hand.get('handoff_mechanism')=='landing_state'
    landing_height=tensor([variants[i].get('landing_height_m',.35) for r,m,i in lanes]).float()
    swing_scale=jp.asarray([variants[i].get('swing_scale',1.) for r,m,i in lanes],dtype=jp.float32)
    jump_start=tensor([variants[i].get('jump_start_x_m',hand['approach']['jump_start_x_m']) for r,m,i in lanes]).float()
    recovery_wc=tensor([variants[i].get('recovery_wc',controller_config['wc']) for r,m,i in lanes]).float()
    body_feedback=tensor([variants[i].get('body_speed_feedback',False) for r,m,i in lanes])
    recovery_eso=tensor([variants[i].get('recovery_eso_scale',1.) for r,m,i in lanes]).float()
    recovery_eso_delay=tensor([variants[i].get('recovery_eso_delay_s',0.) for r,m,i in lanes]).float()
    recovery_look=tensor([variants[i].get('lookahead_scale',1.) for r,m,i in lanes]).float()
    landing_damping=tensor([variants[i].get('steering_damping',.1) for r,m,i in lanes]).float()
    landing_pitch_k=tensor([variants[i].get('landing_pitch_k',0.) for r,m,i in lanes]).float()
    landing_pitch_d=tensor([variants[i].get('landing_pitch_d',0.) for r,m,i in lanes]).float()
    height_offset=jp.asarray([variants[i].get('rb_far_height_offset_m',0.) for r,m,i in lanes],dtype=jp.float32)
    staged_native=hand.get('staged_native_handoff',False)
    gate_roll=tensor([variants[i].get('gate_roll_rad',.08) for r,m,i in lanes]).float()
    gate_rate=tensor([variants[i].get('gate_roll_rate',.3) for r,m,i in lanes]).float()
    gate_hold=tensor([variants[i].get('gate_hold_s',0.) for r,m,i in lanes]).float()
    settle_joints=hand.get('require_settled_pendulum',False)
    home_compensation=hand.get('home_bias_compensation',False)
    require_both_wheels=hand.get('require_both_far_wheels',False)
    lock_first=hand.get('sttw_only_after_lock',False)
    native_landing=hand.get('landing_control')=='native'
    rb_guidance=tensor([variants[i].get('rb_guidance_bound_rad',0.) for r,m,i in lanes]).float()
    gate_upper=hand.get('gate_recovery_upper',False)
    gate_eso=hand.get('gate_recovery_eso',False)
    align_landing=hand.get('landing_control') in ('align','home')
    home_landing=hand.get('landing_control')=='home'
    saved=np.load(hand['reference_path']);route=TorchPathReference(dict(saved),hcfg,device,route_ids=[r for r,m,i in lanes])
    geometry=build_geometry_contract(model);brake_qpos=np.array([idx.hip_qpos_address,idx.knee_qpos_address])
    home_pose=tensor(model.key_qpos[idx.keyframe_id,brake_qpos]).float()
    joint_kp=tensor(model.actuator_gainprm[2:,0]).float()
    terrain_geoms=jp.asarray(model.geom_type==mujoco.mjtGeom.mjGEOM_HFIELD)
    rb_model=mm.replace(opt=mm.opt.replace(timestep=jp.asarray(.005,dtype=mm.opt.timestep.dtype),integrator=type(mm.opt.integrator)(int(mujoco.mjtIntegrator.mjINT_RK4)))) if cfg.get('rb_physics') else mm
    rb_substeps=1 if cfg.get('rb_physics') else args.physics_substeps

    def contacts(d):
        c=d._impl;g=c.contact__geom
        valid=(jp.arange(g.shape[0])<c.nacon[0])&(c.contact__dist<=0)
        valid &= jp.any((g==idx.frontwheel_geom_id)|(g==idx.rearwheel_geom_id),-1)
        valid &= jp.any(terrain_geoms[jp.clip(g,0,model.ngeom-1)]&(g>=0),-1)
        worlds=c.contact__worldid[None,:]==jp.arange(n)[:,None]
        far=valid&(c.contact__pos[:,0]>terrain['front_x_m']+terrain['width_m'])
        front=jp.any(worlds&far&jp.any(g==idx.frontwheel_geom_id,-1),1)
        rear=jp.any(worlds&far&jp.any(g==idx.rearwheel_geom_id,-1),1)
        return jp.stack((jp.any(worlds&valid,1),front|rear,front&rear),1)

    def ground_one(d,c,lock,damping):
        plant=mm.replace(eq_data=mm.eq_data.at[brake_ids,0].set(lock),actuator_biasprm=mm.actuator_biasprm.at[0,2].set(-damping))
        return mjx_env.step(plant,d,c,args.physics_substeps)
    ground=jax.vmap(ground_one)
    jump=jax.vmap(lambda d,c:mjx_env.step(rb_model,d,c,rb_substeps))

    @jax.jit
    def advance(data,ctrl,locks,damping,rb,locked,alive):
        data=data.replace(eq_active=data.eq_active.at[:,brake_ids].set(locked[:,None]))
        # Homogeneous phases execute only one kernel; mixed phases select each world's state.
        def mixed(_):
            gd=ground(data,ctrl,locks,damping);rd=jump(data,ctrl)
            flags=jp.where(rb[:,None],contacts(rd),contacts(gd))
            def choose(path,r,g):
                if path[-1].name in DATA_NON_VMAP:return g
                return jp.where(rb.reshape((n,)+(1,)*(r.ndim-1)),r,g)
            return jax.tree_util.tree_map_with_path(choose,rd,gd),flags
        def only_jump(_):
            d=jump(data,ctrl);return d,contacts(d)
        def only_ground(_):
            d=ground(data,ctrl,locks,damping);return d,contacts(d)
        def step(_):
            d,flags=jax.lax.cond(jp.all(rb),only_jump,lambda _:jax.lax.cond(jp.any(rb),mixed,only_ground,None),None)
            def freeze(path,new,old):
                if path[-1].name in DATA_NON_VMAP:return new
                return jp.where(alive.reshape((n,)+(1,)*(new.ndim-1)),new,old)
            return jax.tree_util.tree_map_with_path(freeze,d,data),flags
        return jax.lax.cond(jp.any(alive),step,lambda _:(data,jp.zeros((n,3),dtype=jp.bool_)),None)

    @jax.jit
    def observe(data):
        q=data.qpos;v=data.qvel;w,x,y,z=q[:,3:7].T
        roll=jp.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=jp.arcsin(jp.clip(2*(w*y-z*x),-1,1));yaw=jp.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        gyro=data.sensordata[:,idx.sensor_addresses[5]:idx.sensor_addresses[5]+3]
        m=jp.stack((-roll,-gyro[:,0],q[:,idx.steering_qpos_address],v[:,idx.steering_dof_address],gyro[:,2],v[:,idx.rearwheel_dof_address],v[:,idx.frontwheel_dof_address]),1)
        return q,v,m,jp.stack((q[:,0],q[:,1],yaw),1),v[:,0]*jp.cos(yaw)+v[:,1]*jp.sin(yaw),jp.stack((roll,pitch,yaw),1)

    def frame_one(d,history,last,offset):
        g=extract_geometry(d,geometry)
        far_fraction=jp.clip((d.qpos[0]-terrain['front_x_m'])/terrain['width_m'],0.,1.)
        geom=ObservableGeometry(g.obstacle_relative_x,d.qpos[2]+offset*far_fraction,g.roll,g.pitch,g.yaw)
        history=advance_history(history,observable_frame(d,idx,geom,last,jp.asarray(1.)))
        return history,actor_observation(history,(d.qpos[0]>=9.5)&(d.qpos[0]<=11.))
    frame=jax.jit(jax.vmap(frame_one))
    results=[];all_timing=[]
    for name,skill in skills.items():
        route=TorchPathReference(dict(saved),hcfg,device,route_ids=[r for r,m,i in lanes])
        route.config['follower']['lookahead_scale']=1.
        init_start=time.perf_counter()
        qpos=np.tile(model.key_qpos[idx.keyframe_id],(n,1));qvel=np.tile(model.key_qvel[idx.keyframe_id],(n,1))
        qpos[:,0]=hand.get('start_x_m',1.5);qvel[:,0]=hand.get('start_vx_m_s',2.)
        for lane,(_,_,i) in enumerate(lanes):
            case=variants[i];yaw=case.get('start_yaw_rad',0.);roll=case.get('start_roll_rad',0.)
            qpos[lane,1]=case.get('start_y_m',0.)
            qpos[lane,3:7]=[np.cos(yaw/2)*np.cos(roll/2),np.cos(yaw/2)*np.sin(roll/2),np.sin(yaw/2)*np.sin(roll/2),np.sin(yaw/2)*np.cos(roll/2)]
            qvel[lane,:2]=hand.get('start_vx_m_s',2.)*np.array([np.cos(yaw),np.sin(yaw)])
        def make(q,v):
            d=mjx_env.make_data(model,qpos=q,qvel=v,impl='warp',naconmax=4096*n,naccdmax=512,njmax=256)
            plant=mm.replace(eq_data=mm.eq_data.at[brake_ids,0].set(q[brake_qpos]))
            return mjx.forward(plant,d.replace(eq_active=d.eq_active.at[brake_ids].set(True)))
        data=jax.jit(jax.vmap(make))(jp.asarray(qpos,dtype=jp.float32),jp.asarray(qvel,dtype=jp.float32))
        def infer_one(obs,knee,scale):
            action,_=skill.infer(obs,knee)
            action=action.at[2:].multiply(scale)
            return action,map_action(action,knee,skill.bundle.action_mapping)
        infer=jax.jit(jax.vmap(infer_one))
        history=jax.tree.map(lambda x:jp.broadcast_to(x,(n,)+x.shape),initial_history())
        last=jp.zeros((n,4));obs=jax.vmap(lambda h:actor_observation(h,jp.asarray(False)))(history)
        policy=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],hcfg,n,device)
        ctl=TorchController(controller_config,n,device)
        middle=TorchMiddleControl(policy,route,identity['frame_scales'],torch.full((n,),hand['approach']['alpha'],device=device),torch.full((n,),hand['approach']['speed_m_s'],device=device))
        q,v,m,pose,speed,angles=map(to_torch,observe(data));raw=torch.stack((speed,m[:,2]),1);previous=torch.stack((m[:,3],m[:,5]),1)
        nearest=(route.xy[route.route_ids]-pose[:,None,:2]).square().sum(-1).argmin(-1)
        middle.reset(m,pose,speed,raw,raw,previous,ctl,initial_progress=route.s[nearest])
        middle.lane_ticks=torch.zeros(n,device=device,dtype=torch.long);middle.latent=torch.zeros_like(raw)
        ground_adapter=GroundAdapter(middle,ctl)
        updates={round(u['time_s']/.005):(u,TorchPathReference(dict(np.load(u['reference_path'])),hcfg,device,route_ids=[r for r,m,i in lanes]))
                 for u in cfg.get('plan_updates',[]) if name in u['skills']}
        plan_receipts=[]
        phase=torch.zeros(n,device=device,dtype=torch.long);alive=torch.ones(n,device=device,dtype=torch.bool)
        recovering=torch.zeros_like(alive)
        stable_time=torch.zeros(n,device=device);upper_ready_time=torch.zeros(n,device=device)
        airborne=torch.zeros_like(alive);flags=torch.zeros(n,3,device=device,dtype=torch.bool)
        far_time=torch.full((n,),-1.,device=device);entry_time=far_time.clone();handoff_time=far_time.clone();hold=torch.zeros(n,device=device)
        lock=q[:,brake_qpos].clone();target=torch.zeros(n,4,device=device);target[:,2:]=lock
        zero=tensor([mode=='zero_middle' for _,mode,i in lanes]);lo=tensor(model.actuator_ctrlrange[:,0]).float();hi=tensor(model.actuator_ctrlrange[:,1]).float()
        codes=torch.zeros(n,device=device,dtype=torch.long);durations=torch.full((n,),args.seconds,device=device)
        rows=[];poses=[];velocities=[];contact_rows=[];valid_rows=[];goal=hand['goal'];data.qpos.block_until_ready();init_s=time.perf_counter()-init_start;rollout_start=time.perf_counter()
        for tick in range(round(args.seconds/.005)+1):
            t=tick*.005
            if tick in updates:
                update,new_route=updates[tick]
                before=ground_adapter.snapshot()
                receipt=ground_adapter.update_plan(new_route,torch.full((n,),update['committed_until_m'],device=device),t,update['expected_previous_version'])
                route=middle.route
                receipt['state_preserved']=bool(torch.equal(before['middle']['frames'],middle.frames)
                    and torch.equal(before['middle']['mask'],middle.mask)
                    and torch.equal(before['middle']['lane_ticks'],middle.lane_ticks)
                    and torch.equal(before['policy']['filtered'],policy.filtered)
                    and torch.equal(before['policy']['offsets'],policy.offsets)
                    and torch.equal(before['controller']['eso'],ctl.eso)
                    and torch.equal(before['controller']['filtered_roll_rate'],ctl.filtered_roll_rate))
                plan_receipts.append(receipt)
            enter=alive&(phase==0)&(q[:,0]>=jump_start)&(tick%4==0)&(not args.ground_only)
            phase=torch.where(enter,1,phase);entry_time=torch.where(enter,t,entry_time)
            airborne |= alive&((phase==1)|(phase==3))&~flags[:,0]&(q[:,0]>terrain['front_x_m']-1.5)
            far=alive&airborne&((phase==1)|(phase==3))&flags[:,1]&(far_time<0)
            far_time=torch.where(far,t,far_time)
            if event:
                descending=(v[:,2]<-.1)&(q[:,2]-terrain['far_bank_height_m']<=landing_height)&(q[:,0]>terrain['front_x_m']+terrain['width_m']-.15)
                preparing=alive&(phase==1)&airborne&(descending|flags[:,1])
                phase=torch.where(preparing,3,phase)
                landed=flags[:,2] if require_both_wheels else flags[:,1]
                ready=landed&(angles[:,0].abs()<.35)&(angles[:,1].abs()<.45)&(v[:,2].abs()<1.5)
                if staged_native:
                    body_ready=landed&(angles[:,0].abs()<gate_roll)&(angles[:,1].abs()<.2)&(v[:,2].abs()<.5)&(m[:,1].abs()<gate_rate)&(speed>=1.5)
                    if settle_joints:
                        body_ready &= (angles[:,1].abs()<.03)&(v[:,4].abs()<.3)&((q[:,brake_qpos]-home_pose).abs().amax(-1)<.15)&(v[:,[idx.hip_dof_address,idx.knee_dof_address]].abs().amax(-1)<.5)
                    stable_time=torch.where(alive&(phase==3)&body_ready,stable_time+.005,0.)
                    taking_wheels=alive&(phase==3)&body_ready&(stable_time>=gate_hold)
                    phase=torch.where(taking_wheels,4,phase)
                if native_landing:ready &= (angles[:,0].abs()<.08)&(angles[:,1].abs()<.15)&(v[:,2].abs()<.4)&(m[:,1].abs()<.3)
                ready &= (v[:,5].abs()<hand.get('handoff_yaw_rate_max_rad_s',float('inf')))&(m[:,3].abs()<hand.get('handoff_steer_rate_max_rad_s',float('inf')))
                ready &= (speed-hand.get('continuation',{}).get('speed_m_s',1.5)).abs()<=hand.get('handoff_speed_error_max_m_s',float('inf'))
                if home_landing or native_landing:ready &= ((q[:,brake_qpos]-home_pose).abs().amax(-1)<hand.get('lock_position_tolerance_rad',.08))&(v[:,[idx.hip_dof_address,idx.knee_dof_address]].abs().amax(-1)<hand.get('lock_velocity_tolerance_rad_s',.5))&(m[:,1].abs()<1.)
                returning=alive&(phase==(4 if staged_native else 3))&ready
                reset_control=returning if lock_first else taking_wheels if staged_native else returning if native_landing else preparing|far|returning
            else:
                returning=alive&(phase==1)&(far_time>=0)&(t-far_time>=hand.get('handoff_delay_s',0.)-1e-7)
                reset_control=returning
            phase=torch.where(returning,2,phase);handoff_time=torch.where(returning,t,handoff_time)
            lock=torch.where(returning[:,None],q[:,brake_qpos],lock)
            middle.v_user=torch.where(reset_control,hand.get('continuation',{}).get('speed_m_s',1.5),middle.v_user)
            middle.alpha=torch.where(reset_control,hand.get('continuation',{}).get('alpha',.5),middle.alpha)
            route.config['follower']['lookahead_scale']=torch.where(phase>=2,recovery_look,1.)
            middle.reset_mask(reset_control,m,pose,speed,ctl)
            recovering |= reset_control
            target[:,0]=torch.where(reset_control,q[:,idx.steering_qpos_address],target[:,0]);target[:,2:]=torch.where(returning[:,None],lock,target[:,2:])
            ey=(route.xy[route.route_ids]-q[:,None,:2]).square().sum(-1).min(-1).values.sqrt()
            eh=wrap(middle.projection['heading']-pose[:,2]);ground_mode=((phase==0)|(phase==2)|(phase==4)) if native_landing else (phase!=1)
            physical=~torch.isfinite(q).all(-1)|(angles[:,0].abs()>.7)|(angles[:,1].abs()>1.309)
            outside=ground_mode&((ey>hand['road_half_width_m'])|middle.domain_exit)
            at_goal=((phase==2)|args.ground_only)&(q[:,0]>=goal['x_min_m'])&(q[:,0]<=goal['x_max_m'])&(q[:,1].abs()<=goal['abs_y_max_m'])&(v[:,0]>=goal['min_vx_m_s'])
            at_goal &= (ey<=goal.get('path_tolerance_m',float('inf')))&(eh.abs()<=goal.get('heading_tolerance_rad',float('inf')))&((speed-middle.v_user).abs()<=goal.get('speed_tolerance_m_s',float('inf')))
            hold=torch.where(alive,torch.where(at_goal,hold+.005,0.),hold)
            reached=at_goal&(hold>=goal.get('hold_s',0.))
            stop=alive&(physical|outside|reached);codes=torch.where(stop,torch.where(physical,1,torch.where(outside,2,3)),codes);durations=torch.where(stop,t,durations)
            rows.append(torch.cat((torch.full((n,1),t,device=device),q[:,:3],angles,v[:,:3],(phase==2)[:,None],ey[:,None],target,(phase==0)[:,None],m[:,5:6],middle.v_user[:,None],phase[:,None]),1))
            poses.append(q);velocities.append(v);contact_rows.append(flags.clone());valid_rows.append(alive.clone());alive &= ~stop
            if tick==round(args.seconds/.005):break
            upper_ready=(phase==2)&(m[:,0].abs()<.05)&(m[:,1].abs()<.2)&(eh.abs()<.15)&(ey<.2)
            upper_ready &= ((speed-middle.v_user).abs()<hand.get('upper_ready_speed_error_m_s',float('inf')))&(v[:,5].abs()<hand.get('upper_ready_yaw_rate_rad_s',float('inf')))&(m[:,3].abs()<hand.get('upper_ready_steer_rate_rad_s',float('inf')))
            upper_ready_time=torch.where(recovering&upper_ready,upper_ready_time+.005,0.)
            recovering &= ~(upper_ready&(upper_ready_time>=hand.get('upper_ready_hold_s',0.)))
            governed=middle.before(pose,zero_mask=zero|(recovering if gate_upper else False))
            ctl.config['wc']=torch.where(phase>=2,recovery_wc,controller_config['wc'])
            feedback_speed=torch.where((phase>=2)&body_feedback,speed,m[:,5]*.07)
            enable_eso=torch.full((n,),t>3.,device=device)
            enable_eso &= (phase<2)|(middle.lane_ticks*.005>=recovery_eso_delay)
            if gate_eso:enable_eso &= (phase<2)|((m[:,0].abs()<.08)&(m[:,1].abs()<.4))
            command=ctl.step(torch.stack((feedback_speed,m[:,2],m[:,3],m[:,0],m[:,1],governed[:,1]),1),enable_eso,torch.where(phase>=2,recovery_eso,1.))
            steer=command[:,0].clamp(-3,3);steer=torch.maximum(((-.8-m[:,2])/.005).clamp_min(-3),torch.minimum(((.8-m[:,2])/.005).clamp_max(3),steer))
            ground_target=torch.stack(((target[:,0]+.005*steer).clamp(-.8,.8),governed[:,0]/.07,lock[:,0],lock[:,1]),1).clamp(lo,hi)
            if tick%4==0:
                action,rb_ctrl=infer(obs,data.qpos[:,idx.knee_qpos_address],swing_scale)
                rb_target=to_torch(rb_ctrl).clone()
                correction=(-.5*wrap(pose[:,2]-middle.projection['heading'])-.4*middle.projection['cross_track']).clamp(-rb_guidance,rb_guidance)
                guide=(far_time>=0)&(m[:,0].abs()<.15)&(m[:,1].abs()<.6)
                rb_target[:,0]=(rb_target[:,0]+torch.where(guide,correction,0.)).clamp(lo[0],hi[0])
                delivered=to_torch(action).clone();delivered[:,0]=rb_target[:,0]/hi[0]
                rb_controls=(phase==1)|(phase==3)|((phase==4)&lock_first)
                last=jp.where(to_jax(rb_controls&alive)[:,None],to_jax(delivered),last)
                target=torch.where(rb_controls[:,None],rb_target,target)
            if align_landing:
                aligned=target[:,0]+(middle.nominal[:,1]-target[:,0]).clamp(-.015,.015)
                ground_target[:,0]=torch.where((phase==3)&(~flags[:,0] if home_landing else True),aligned,ground_target[:,0])
            home_targets=home_pose.expand(n,2).clone()
            pitch_assist=landing_pitch_k*angles[:,1]+landing_pitch_d*v[:,4]
            home_targets[:,0]=(home_targets[:,0]+torch.where(far_time<0,pitch_assist,0.)).clamp(lo[2],hi[2])
            ground_target[:,2:]=torch.where((phase==3)[:,None],home_targets if home_landing else target[:,2:],ground_target[:,2:])
            ground_target[:,2:]=torch.where((phase==4)[:,None],home_pose,ground_target[:,2:])
            if home_compensation:
                compensated=(home_pose+to_torch(data.qfrc_bias[:,[idx.hip_dof_address,idx.knee_dof_address]])/joint_kp).clamp(lo[2:],hi[2:])
                ground_target[:,2:]=torch.where((phase==4)[:,None],compensated,ground_target[:,2:])
            if lock_first:
                # RB owns wheels throughout physical homing; STTW starts only after locking.
                ground_target[:,:2]=torch.where((phase==4)[:,None],target[:,:2],ground_target[:,:2])
            target=torch.where(ground_mode[:,None],ground_target,target)
            if lock_first:
                delta=target[:,2:]-home_pose
                span=torch.where(delta>=0,hi[2:]-home_pose,home_pose-lo[2:]).clamp_min(1e-6)
                actual=to_torch(last).clone();actual[:,2:]=(delta/span).clamp(-1,1)
                last=jp.where(to_jax((phase==4)&alive)[:,None],to_jax(actual),last)
            data,jflags=advance(data,to_jax(target),to_jax(lock),to_jax(torch.where(phase>=2,landing_damping,.1)),to_jax(phase==1 if align_landing else ((phase==1)|(phase==3))),to_jax((phase==0)|(phase==2)),to_jax(alive));flags=to_torch(jflags)
            q,v,m,pose,speed,angles=map(to_torch,observe(data))
            middle.after(m,pose,speed,governed,torch.stack((steer,target[:,1]),1),torch.zeros(n,2,device=device),ctl)
            if (tick+1)%4==0:history,obs=frame(data,history,last,height_offset)
        torch.cuda.synchronize();rollout_s=time.perf_counter()-rollout_start
        traces=torch.stack(rows).cpu().numpy();positions=torch.stack(poses).cpu().numpy();speeds=torch.stack(velocities).cpu().numpy();valid=torch.stack(valid_rows).cpu().numpy();contact_trace=torch.stack(contact_rows).cpu().numpy()
        codes,durations,entry_time,far_time,handoff_time,hold,airborne=[x.cpu().numpy() for x in (codes,durations,entry_time,far_time,handoff_time,hold,airborne)]
        for lane,(route_id,mode,variant) in enumerate(lanes):
            case_dir=f'{name}_route{route_id}_{mode}'+(f'_v{variant}' if len(variants)>1 else '')
            target_dir=args.output/case_dir;target_dir.mkdir();keep=valid[:,lane];trace=traces[keep,lane];qsave=positions[keep,lane]
            np.savez_compressed(target_dir/'trajectory.npz',trace=trace,qpos=qsave,qvel=speeds[keep,lane],wheel_contact_flags=contact_trace[keep,lane],wheel_contact_columns=['any_terrain','any_far','both_far'],reference_xy=saved['xy'][route_id],published_reference_xy=route.xy[route_id].cpu().numpy(),columns=['time','x','y','z','roll','pitch','yaw','vx','vy','vz','ground_active','path_distance','steer_target','rear_target','hip_target','knee_target','approaching','rear_rate_rad_s','requested_speed_m_s','phase'])
            row=dict(case_dir=case_dir,variant=variants[variant],skill=name,route=route_id,continuation=mode,termination=['timeout','physical_failure','corridor_or_domain_exit','goal'][codes[lane]],goal=bool(codes[lane]==3),rb_entry_s=None if entry_time[lane]<0 else float(entry_time[lane]),goal_hold_s=float(hold[lane]),airborne=bool(airborne[lane]),far_contact=bool(far_time[lane]>=0),first_far_contact_s=None if far_time[lane]<0 else float(far_time[lane]),handoff_s=None if handoff_time[lane]<0 else float(handoff_time[lane]),duration_s=float(durations[lane]),final_xy=qsave[-1,:2].tolist())
            results.append(row)
        all_timing.append(dict(skill=name,lanes=n,initialization_s=init_s,rollout_s=rollout_s,simulated_ticks_per_lane=tick))
        (args.output/f'{name}_plan_receipts.json').write_text(json.dumps(plan_receipts,indent=2))
    (args.output/'status.json').write_text(json.dumps(dict(stage='complete',completed=len(results),total=len(results),elapsed_s=time.perf_counter()-started,results=results),indent=2))
    (args.output/'timing.json').write_text(json.dumps(dict(total_s=time.perf_counter()-started,batches=all_timing,execution='GPU MJX-Warp vmap, GPU H75/ECBC/ESO and RB inference, DLPack, terminal-only trajectory download'),indent=2))
