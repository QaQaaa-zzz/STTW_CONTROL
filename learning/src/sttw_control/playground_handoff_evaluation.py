"""GPU-resident batched STTW/RB/STTW inference on the native Playground chassis."""
import json
import time
import numpy as np
import jax
import jax.numpy as jp
import torch
import mujoco
from mujoco import mjx
from mujoco.mjx.warp.types import DATA_NON_VMAP
from mujoco_playground._src import mjx_env
from jit_dvgc.geometry import build_geometry_contract, extract_geometry
from jit_dvgc.observation import initial_history, advance_history, observable_frame, actor_observation, ObservableGeometry
from .torch_controller import TorchController
from .torch_path_policy import TorchPathPolicy
from .torch_path_reference import TorchPathReference, wrap
from .torch_middle_control import TorchMiddleControl


def evaluate_handoff(model, mm, idx, cfg, bundle, hcfg, identity, controller_config, skills, args, brake_ids):
    started=time.perf_counter();device='cuda:0';hand=cfg['ground_handoff'];terrain=cfg['ditch_terrain']
    to_torch=torch.utils.dlpack.from_dlpack
    to_jax=lambda x:jax.dlpack.from_dlpack(x.detach().contiguous())
    tensor=lambda x:torch.as_tensor(x,device=device)
    lanes=[(r,m) for r in args.routes for m in args.modes];n=len(lanes)
    saved=np.load(hand['reference_path']);route=TorchPathReference(dict(saved),hcfg,device,route_ids=[r for r,m in lanes])
    geometry=build_geometry_contract(model);brake_qpos=np.array([idx.hip_qpos_address,idx.knee_qpos_address])
    terrain_geoms=jp.asarray(model.geom_type==mujoco.mjtGeom.mjGEOM_HFIELD)
    rb_model=mm.replace(opt=mm.opt.replace(timestep=jp.asarray(.005,dtype=mm.opt.timestep.dtype),integrator=type(mm.opt.integrator)(int(mujoco.mjtIntegrator.mjINT_RK4)))) if cfg.get('rb_physics') else mm
    rb_substeps=1 if cfg.get('rb_physics') else args.physics_substeps

    def contacts(d):
        c=d._impl;g=c.contact__geom
        valid=(jp.arange(g.shape[0])<c.nacon[0])&(c.contact__dist<=0)
        valid &= jp.any((g==idx.frontwheel_geom_id)|(g==idx.rearwheel_geom_id),-1)
        valid &= jp.any(terrain_geoms[jp.clip(g,0,model.ngeom-1)]&(g>=0),-1)
        worlds=c.contact__worldid[None,:]==jp.arange(n)[:,None]
        return jp.stack((jp.any(worlds&valid,1),jp.any(worlds&valid&(c.contact__pos[:,0]>terrain['front_x_m']+terrain['width_m']),1)),1)

    def ground_one(d,c,lock):
        plant=mm.replace(eq_data=mm.eq_data.at[brake_ids,0].set(lock))
        return mjx_env.step(plant,d,c,args.physics_substeps)
    ground=jax.vmap(ground_one)
    jump=jax.vmap(lambda d,c:mjx_env.step(rb_model,d,c,rb_substeps))

    @jax.jit
    def advance(data,ctrl,locks,rb,alive):
        data=data.replace(eq_active=data.eq_active.at[:,brake_ids].set(~rb[:,None]))
        # Homogeneous phases execute only one kernel; mixed phases select each world's state.
        def mixed(_):
            gd=ground(data,ctrl,locks);rd=jump(data,ctrl)
            flags=jp.where(rb[:,None],contacts(rd),contacts(gd))
            def choose(path,r,g):
                if path[-1].name in DATA_NON_VMAP:return g
                return jp.where(rb.reshape((n,)+(1,)*(r.ndim-1)),r,g)
            return jax.tree_util.tree_map_with_path(choose,rd,gd),flags
        def only_jump(_):
            d=jump(data,ctrl);return d,contacts(d)
        def only_ground(_):
            d=ground(data,ctrl,locks);return d,contacts(d)
        def step(_):
            d,flags=jax.lax.cond(jp.all(rb),only_jump,lambda _:jax.lax.cond(jp.any(rb),mixed,only_ground,None),None)
            def freeze(path,new,old):
                if path[-1].name in DATA_NON_VMAP:return new
                return jp.where(alive.reshape((n,)+(1,)*(new.ndim-1)),new,old)
            return jax.tree_util.tree_map_with_path(freeze,d,data),flags
        return jax.lax.cond(jp.any(alive),step,lambda _:(data,jp.zeros((n,2),dtype=jp.bool_)),None)

    @jax.jit
    def observe(data):
        q=data.qpos;v=data.qvel;w,x,y,z=q[:,3:7].T
        roll=jp.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=jp.arcsin(jp.clip(2*(w*y-z*x),-1,1));yaw=jp.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        gyro=data.sensordata[:,idx.sensor_addresses[5]:idx.sensor_addresses[5]+3]
        m=jp.stack((-roll,-gyro[:,0],q[:,idx.steering_qpos_address],v[:,idx.steering_dof_address],gyro[:,2],v[:,idx.rearwheel_dof_address],v[:,idx.frontwheel_dof_address]),1)
        return q,v,m,jp.stack((q[:,0],q[:,1],yaw),1),v[:,0]*jp.cos(yaw)+v[:,1]*jp.sin(yaw),jp.stack((roll,pitch,yaw),1)

    def frame_one(d,history,last):
        g=extract_geometry(d,geometry);geom=ObservableGeometry(g.obstacle_relative_x,d.qpos[2],g.roll,g.pitch,g.yaw)
        history=advance_history(history,observable_frame(d,idx,geom,last,jp.asarray(1.)))
        return history,actor_observation(history,(d.qpos[0]>=9.5)&(d.qpos[0]<=11.))
    frame=jax.jit(jax.vmap(frame_one))
    results=[];all_timing=[]
    for name,skill in skills.items():
        init_start=time.perf_counter()
        qpos=np.tile(model.key_qpos[idx.keyframe_id],(n,1));qvel=np.tile(model.key_qvel[idx.keyframe_id],(n,1))
        qpos[:,0]=hand.get('start_x_m',1.5);qvel[:,0]=hand.get('start_vx_m_s',2.)
        def make(q,v):
            d=mjx_env.make_data(model,qpos=q,qvel=v,impl='warp',naconmax=4096*n,naccdmax=512,njmax=256)
            plant=mm.replace(eq_data=mm.eq_data.at[brake_ids,0].set(q[brake_qpos]))
            return mjx.forward(plant,d.replace(eq_active=d.eq_active.at[brake_ids].set(True)))
        data=jax.jit(jax.vmap(make))(jp.asarray(qpos,dtype=jp.float32),jp.asarray(qvel,dtype=jp.float32))
        infer=jax.jit(jax.vmap(skill.infer))
        history=jax.tree.map(lambda x:jp.broadcast_to(x,(n,)+x.shape),initial_history())
        last=jp.zeros((n,4));obs=jax.vmap(lambda h:actor_observation(h,jp.asarray(False)))(history)
        policy=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],hcfg,n,device)
        ctl=TorchController(controller_config,n,device)
        middle=TorchMiddleControl(policy,route,identity['frame_scales'],torch.full((n,),hand['approach']['alpha'],device=device),torch.full((n,),hand['approach']['speed_m_s'],device=device))
        q,v,m,pose,speed,angles=map(to_torch,observe(data));raw=torch.stack((speed,m[:,2]),1);previous=torch.stack((m[:,3],m[:,5]),1)
        nearest=(route.xy[route.route_ids]-pose[:,None,:2]).square().sum(-1).argmin(-1)
        middle.reset(m,pose,speed,raw,raw,previous,ctl,initial_progress=route.s[nearest])
        middle.lane_ticks=torch.zeros(n,device=device,dtype=torch.long);middle.latent=torch.zeros_like(raw)
        phase=torch.zeros(n,device=device,dtype=torch.long);alive=torch.ones(n,device=device,dtype=torch.bool)
        airborne=torch.zeros_like(alive);flags=torch.zeros(n,2,device=device,dtype=torch.bool)
        far_time=torch.full((n,),-1.,device=device);entry_time=far_time.clone();handoff_time=far_time.clone();hold=torch.zeros(n,device=device)
        lock=q[:,brake_qpos].clone();target=torch.zeros(n,4,device=device);target[:,2:]=lock
        zero=tensor([mode=='zero_middle' for _,mode in lanes]);lo=tensor(model.actuator_ctrlrange[:,0]).float();hi=tensor(model.actuator_ctrlrange[:,1]).float()
        codes=torch.zeros(n,device=device,dtype=torch.long);durations=torch.full((n,),args.seconds,device=device)
        rows=[];poses=[];valid_rows=[];goal=hand['goal'];data.qpos.block_until_ready();init_s=time.perf_counter()-init_start;rollout_start=time.perf_counter()
        for tick in range(round(args.seconds/.005)+1):
            t=tick*.005
            enter=alive&(phase==0)&(q[:,0]>=hand['approach']['jump_start_x_m'])&(tick%4==0)&(not args.ground_only)
            phase=torch.where(enter,1,phase);entry_time=torch.where(enter,t,entry_time)
            airborne |= alive&(phase==1)&~flags[:,0]&(q[:,0]>terrain['front_x_m']-1.5)
            far=alive&airborne&(phase==1)&flags[:,1]&(far_time<0)
            far_time=torch.where(far,t,far_time)
            returning=alive&(phase==1)&(far_time>=0)&(t-far_time>=hand.get('handoff_delay_s',0.)-1e-7)
            phase=torch.where(returning,2,phase);handoff_time=torch.where(returning,t,handoff_time)
            lock=torch.where(returning[:,None],q[:,brake_qpos],lock)
            middle.v_user=torch.where(returning,hand.get('continuation',{}).get('speed_m_s',1.5),middle.v_user)
            middle.alpha=torch.where(returning,hand.get('continuation',{}).get('alpha',.5),middle.alpha)
            middle.reset_mask(returning,m,pose,speed,ctl)
            target[:,0]=torch.where(returning,q[:,idx.steering_qpos_address],target[:,0]);target[:,2:]=torch.where(returning[:,None],lock,target[:,2:])
            ey=(route.xy[route.route_ids]-q[:,None,:2]).square().sum(-1).min(-1).values.sqrt()
            eh=wrap(middle.projection['heading']-pose[:,2]);ground_mode=phase!=1
            physical=~torch.isfinite(q).all(-1)|(angles[:,0].abs()>.7)|(angles[:,1].abs()>1.309)
            outside=ground_mode&((ey>hand['road_half_width_m'])|middle.domain_exit)
            at_goal=((phase==2)|args.ground_only)&(q[:,0]>=goal['x_min_m'])&(q[:,0]<=goal['x_max_m'])&(q[:,1].abs()<=goal['abs_y_max_m'])&(v[:,0]>=goal['min_vx_m_s'])
            at_goal &= (ey<=goal.get('path_tolerance_m',float('inf')))&(eh.abs()<=goal.get('heading_tolerance_rad',float('inf')))&((speed-middle.v_user).abs()<=goal.get('speed_tolerance_m_s',float('inf')))
            hold=torch.where(alive,torch.where(at_goal,hold+.005,0.),hold)
            reached=at_goal&(hold>=goal.get('hold_s',0.))
            stop=alive&(physical|outside|reached);codes=torch.where(stop,torch.where(physical,1,torch.where(outside,2,3)),codes);durations=torch.where(stop,t,durations)
            rows.append(torch.cat((torch.full((n,1),t,device=device),q[:,:3],angles,v[:,:3],(phase==2)[:,None],ey[:,None],target,(phase==0)[:,None],m[:,5:6],middle.v_user[:,None]),1))
            poses.append(q);valid_rows.append(alive.clone());alive &= ~stop
            if tick==round(args.seconds/.005):break
            governed=middle.before(pose,zero_mask=zero)
            command=ctl.step(torch.stack((m[:,5]*.07,m[:,2],m[:,3],m[:,0],m[:,1],governed[:,1]),1),torch.full((n,),t>3.,device=device))
            steer=command[:,0].clamp(-3,3);steer=torch.maximum(((-.8-m[:,2])/.005).clamp_min(-3),torch.minimum(((.8-m[:,2])/.005).clamp_max(3),steer))
            ground_target=torch.stack(((target[:,0]+.005*steer).clamp(-.8,.8),governed[:,0]/.07,lock[:,0],lock[:,1]),1).clamp(lo,hi)
            if tick%4==0:
                action,rb_ctrl=infer(obs,data.qpos[:,idx.knee_qpos_address])
                last=jp.where(to_jax((phase==1)&alive)[:,None],action,last)
                target=torch.where((phase==1)[:,None],to_torch(rb_ctrl),target)
            target=torch.where(ground_mode[:,None],ground_target,target)
            data,jflags=advance(data,to_jax(target),to_jax(lock),to_jax(phase==1),to_jax(alive));flags=to_torch(jflags)
            q,v,m,pose,speed,angles=map(to_torch,observe(data))
            middle.after(m,pose,speed,governed,torch.stack((steer,target[:,1]),1),torch.zeros(n,2,device=device),ctl)
            if (tick+1)%4==0:history,obs=frame(data,history,last)
        torch.cuda.synchronize();rollout_s=time.perf_counter()-rollout_start
        traces=torch.stack(rows).cpu().numpy();positions=torch.stack(poses).cpu().numpy();valid=torch.stack(valid_rows).cpu().numpy()
        codes,durations,entry_time,far_time,handoff_time,hold,airborne=[x.cpu().numpy() for x in (codes,durations,entry_time,far_time,handoff_time,hold,airborne)]
        for lane,(route_id,mode) in enumerate(lanes):
            target_dir=args.output/f'{name}_route{route_id}_{mode}';target_dir.mkdir();keep=valid[:,lane];trace=traces[keep,lane];qsave=positions[keep,lane]
            np.savez_compressed(target_dir/'trajectory.npz',trace=trace,qpos=qsave,reference_xy=saved['xy'][route_id],columns=['time','x','y','z','roll','pitch','yaw','vx','vy','vz','ground_active','path_distance','steer_target','rear_target','hip_target','knee_target','approaching','rear_rate_rad_s','requested_speed_m_s'])
            row=dict(skill=name,route=route_id,continuation=mode,termination=['timeout','physical_failure','corridor_or_domain_exit','goal'][codes[lane]],goal=bool(codes[lane]==3),rb_entry_s=None if entry_time[lane]<0 else float(entry_time[lane]),goal_hold_s=float(hold[lane]),airborne=bool(airborne[lane]),far_contact=bool(far_time[lane]>=0),first_far_contact_s=None if far_time[lane]<0 else float(far_time[lane]),handoff_s=None if handoff_time[lane]<0 else float(handoff_time[lane]),duration_s=float(durations[lane]),final_xy=qsave[-1,:2].tolist())
            results.append(row)
        all_timing.append(dict(skill=name,lanes=n,initialization_s=init_s,rollout_s=rollout_s,simulated_ticks_per_lane=tick))
    (args.output/'status.json').write_text(json.dumps(dict(stage='complete',completed=len(results),total=len(results),elapsed_s=time.perf_counter()-started,results=results),indent=2))
    (args.output/'timing.json').write_text(json.dumps(dict(total_s=time.perf_counter()-started,batches=all_timing,execution='GPU MJX-Warp vmap, GPU H75/ECBC/ESO and RB inference, DLPack, terminal-only trajectory download'),indent=2))
