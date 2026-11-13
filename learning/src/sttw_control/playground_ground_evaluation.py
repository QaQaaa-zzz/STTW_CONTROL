"""Batched STTW capability evaluation on one unchanged Playground terrain."""
import json,time
from pathlib import Path
import numpy as np
import torch
from .torch_controller import TorchController
from .torch_path_policy import TorchPathPolicy
from .torch_path_reference import TorchPathReference,wrap
from .torch_middle_control import TorchMiddleControl


def evaluate_ground_suite(model,mm,idx,cfg,bundle,hcfg,identity,controller_config,output,substeps,brake_ids,simulation=None):
    torch.set_default_device('cuda:0')
    device='cuda:0'
    suite=cfg['ground_suite'];cases=suite['cases'];methods=[(f'alpha{v:g}',v,False) for v in suite['alphas']]+[('zero_middle',.5,True)]
    lanes=[dict(case=i,method=name,alpha=alpha,zero_middle=zero) for i in range(len(cases)) for name,alpha,zero in methods]
    n=len(lanes);tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device=device)
    case_ids=np.array([v['case'] for v in lanes]);alpha=tensor([v['alpha'] for v in lanes]);zero=torch.tensor([v['zero_middle'] for v in lanes]);speed=tensor([cases[i]['speed_m_s'] for i in case_ids])
    saved=np.load(suite['reference_path']);route=TorchPathReference(dict(saved),hcfg,device,route_ids=case_ids)
    policy=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],hcfg,n,device);ctl=TorchController(controller_config,n,device);middle=TorchMiddleControl(policy,route,identity['frame_scales'],alpha,speed)
    if simulation is None:
        import jax
        import jax.numpy as jp
        from mujoco import mjx
        from mujoco_playground._src import mjx_env
        qpos=np.tile(model.key_qpos[idx.keyframe_id],(n,1));qvel=np.tile(model.key_qvel[idx.keyframe_id],(n,1))
        for lane,case_id in enumerate(case_ids):
            case=cases[case_id];qpos[lane,:2]=case['start_xy'];yaw=case['start_yaw'];qpos[lane,3:7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)];qvel[lane,:2]=case['speed_m_s']*np.array([np.cos(yaw),np.sin(yaw)])
        mm=mm.replace(eq_data=mm.eq_data.at[brake_ids,0].set(jp.asarray(qpos[0,[idx.hip_qpos_address,idx.knee_qpos_address]])))
        def make(q,v):
            d=mjx_env.make_data(model,qpos=q,qvel=v,impl='warp',naconmax=4096,naccdmax=512,njmax=256)
            return d.replace(eq_active=d.eq_active.at[brake_ids].set(True))
        data=jax.vmap(make)(jp.asarray(qpos,dtype=jp.float32),jp.asarray(qvel,dtype=jp.float32));data=jax.jit(jax.vmap(lambda d:mjx.forward(mm,d)))(data)
        advance=jax.jit(jax.vmap(lambda d,c:mjx_env.step(mm,d,c,substeps)))
        def observe(d):
            q=np.asarray(d.qpos);v=np.asarray(d.qvel);w,x,y,z=q[:,3:7].T
            roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1));yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
            gyro=np.asarray(d.sensordata)[:,idx.sensor_addresses[5]:idx.sensor_addresses[5]+3]
            m=tensor(np.stack((-roll,-gyro[:,0],q[:,idx.steering_qpos_address],v[:,idx.steering_dof_address],gyro[:,2],v[:,idx.rearwheel_dof_address],v[:,idx.frontwheel_dof_address]),-1))
            pose=tensor(np.stack((q[:,0],q[:,1],yaw),-1));speed=tensor(v[:,0]*np.cos(yaw)+v[:,1]*np.sin(yaw))
            return q,m,pose,speed,tensor(pitch)
    else:
        simulation.reset(cases,case_ids)
        data=None
        observe=lambda data:simulation.observe()
    q,m,pose,v,pitch=observe(data);raw=torch.stack((v,m[:,2]),-1);previous=torch.stack((m[:,3],m[:,5]),-1)
    middle.reset(m,pose,v,raw,raw,previous,ctl)
    target=tensor(np.zeros((n,4)));target[:,2:]=tensor(q[:,[idx.hip_qpos_address,idx.knee_qpos_address]]) if simulation is None else simulation.plant.pendulum_target
    alive=torch.ones(n,dtype=torch.bool);hold=torch.zeros(n);goal=tensor([cases[i]['goal_s'] for i in case_ids]);turn_end=tensor([cases[i]['turn_end_s'] for i in case_ids]);reason=['timeout']*n;duration=np.full(n,suite['seconds']);rows=[];initial_qpos=q.copy();started=time.monotonic()
    declaration=dict(schema='playground_ground_suite_v1',role='nominal DEV, no training',source_config=cfg,model=identity,controller=controller_config,network_config=hcfg,lanes=lanes,physics_dt=float(model.opt.timestep) if simulation is None else simulation.plant.physics_dt,lower_dt=.005,middle_dt=.02,backend='MJX-Warp vmap; shared model; independent worlds; batched Torch controller' if simulation is None else 'IsaacLab PhysX; original velocity steering; shared controller and evaluation loop',pendulum='hip/knee equality brakes active for all STTW steps' if simulation is None else 'hip/knee PhysX joint brakes active for all STTW steps',budget=dict(episodes=n,seconds_each=suite['seconds'],maximum_physics_steps=round(suite['seconds']/.005)*substeps*n,optimizer_updates=0))
    (output/'suite_declaration.json').write_text(json.dumps(declaration,indent=2))
    for tick in range(round(suite['seconds']/.005)):
        valid=alive.clone();governed=middle.before(pose,zero_mask=zero)
        command=ctl.step(torch.stack((m[:,5]*.07,m[:,2],m[:,3],m[:,0],m[:,1],governed[:,1]),-1),torch.full((n,),tick*.005>3.))
        steer=command[:,0].clamp(-3,3);steer=torch.maximum(((-.8-m[:,2])/.005).clamp_min(-3),torch.minimum(((.8-m[:,2])/.005).clamp_max(3),steer))
        target[:,0]=(target[:,0]+.005*steer).clamp(-.8,.8);target[:,1]=(governed[:,0]/.07).clamp(-5,40)
        applied=torch.where(alive[:,None],target,torch.zeros_like(target));applied[:,2:]=target[:,2:]
        if simulation is None:data=advance(data,jp.asarray(applied.cpu().numpy()))
        else:simulation.advance(applied,torch.where(alive,steer,0.))
        q,m,pose,v,pitch=observe(data)
        middle.after(m,pose,v,governed,torch.stack((steer,target[:,1]),-1),torch.zeros(n,2),ctl)
        projection=middle.projection;ey=projection['cross_track'];eh=wrap(projection['heading']-pose[:,2]);ev=v-speed
        finite=torch.isfinite(m).all(-1)&torch.isfinite(pose).all(-1)&torch.isfinite(v)
        physical=~finite|(m[:,0].abs()>.7)|(pitch.abs()>1.309)
        corridor=(ey.abs()>suite['road_half_width_m'])|middle.domain_exit|(pose[:,1].abs()>suite['ground_lateral_limit_m'])
        settled=(ey.abs()<=suite['final_path_m'])&(eh.abs()<=suite['final_heading_rad'])&(ev.abs()<=suite['final_speed_m_s'])&(projection['progress']>=turn_end)
        hold=torch.where(settled,hold+.005,0.)
        reached=(projection['progress']>=goal)&(hold>=suite['final_hold_s'])
        stop=valid&(physical|corridor|reached);failed=valid&(physical|corridor)
        for lane in stop.nonzero().flatten().tolist():
            reason[lane]='physical_failure' if physical[lane] else 'corridor_or_domain_exit' if corridor[lane] else 'goal_hold';duration[lane]=(tick+1)*.005
        rows.append(dict(qpos=q.copy(),measurement=m.cpu().numpy().copy(),pose=pose.cpu().numpy().copy(),speed=v.cpu().numpy().copy(),ey=ey.cpu().numpy().copy(),eh=eh.cpu().numpy().copy(),reference=governed.cpu().numpy().copy(),nominal=middle.nominal.cpu().numpy().copy(),offsets=policy.offsets.cpu().numpy().copy(),command=target.cpu().numpy().copy(),valid=valid.cpu().numpy().copy(),failed=failed.cpu().numpy().copy(),progress=projection['progress'].cpu().numpy().copy(),curvature=route.at(projection['progress'])[2].cpu().numpy().copy(),hold=hold.cpu().numpy().copy()))
        alive &= ~stop
        if (tick+1)%100==0 or not alive.any():
            status=dict(stage='running',step=tick+1,active=int(alive.sum()),total=n,elapsed_s=time.monotonic()-started);(output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
        if not alive.any():break
    data_arrays={key:np.stack([row[key] for row in rows]) for key in rows[0]};np.savez_compressed(output/'trajectory.npz',time=(np.arange(len(rows))+1)*.005,initial_qpos=initial_qpos,**data_arrays)
    results=[dict(lane=i,**lane,case_name=cases[lane['case']]['name'],termination=reason[i],duration_s=float(duration[i]),success=reason[i]=='goal_hold') for i,lane in enumerate(lanes)]
    status=dict(stage='complete',steps=len(rows),elapsed_s=time.monotonic()-started,total=n,success=sum(v['success'] for v in results),results=results)
    (output/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
