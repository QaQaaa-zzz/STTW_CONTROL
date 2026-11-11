"""Frozen native RB -> STTW trial using Playground MJX stepping; no learning."""
import argparse
import json
import os
from pathlib import Path
import time
import xml.etree.ElementTree as ET
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx
# Increase per-pair heightfield contact workspace, retaining the 25mm surface.
import warp as wp
from mujoco.mjx.third_party.mujoco_warp._src import collision_convex
collision_convex.MJ_MAXCONPAIR = 512
collision_convex.vec_maxconpair = wp.types.vector(length=512, dtype=float)
collision_convex.mat_maxconpair = wp.types.matrix(shape=(512, 3), dtype=float)
from mujoco_playground._src import mjx_env
from scipy.interpolate import LinearNDInterpolator
import torch
from sttw_control.ditch_terrain import build_ditch_terrain
from sttw_control.pendulum_skill import PendulumSkill
from sttw_control.torch_controller import TorchController
from sttw_control.torch_path_policy import TorchPathPolicy
from sttw_control.torch_path_reference import TorchPathReference
from sttw_control.torch_middle_control import TorchMiddleControl
from jit_dvgc.geometry import build_geometry_contract, extract_geometry
from jit_dvgc.model import _build_index
from jit_dvgc.observation import initial_history, advance_history, observable_frame, actor_observation, ObservableGeometry


def transplant(bundle, terrain, output):
    """Two disjoint heightfields, preserving the sampled rough bank surface."""
    root = ET.parse(bundle.xml_path).getroot()
    root.find('compiler').set('meshdir', str(bundle.xml_path.parent/'meshes'))
    root.find('option').set('timestep', '.005')
    root.find('option').set('ccd_iterations', '100')
    world, assets = root.find('worldbody'), root.find('asset')
    for name in ('floor', 'step'):
        geom = world.find(f"geom[@name='{name}']")
        geom.set('contype', '0'); geom.set('conaffinity', '0'); geom.set('rgba', '0 0 0 0')
    world.find("geom[@name='step']").set('pos', f"{terrain['front_x_m']+2} 0 0")
    vertices, triangles = build_ditch_terrain(terrain)
    heights = []
    front, far = terrain['front_x_m'], terrain['front_x_m']+terrain['width_m']
    for i,(lo,hi) in enumerate(((-5.,front),(far,terrain['ground_end_x_m']))):
        v = vertices[(vertices[:,0]>=lo-1e-5)&(vertices[:,0]<=hi+1e-5)]
        points, indices = np.unique(v[:,:2], axis=0, return_index=True)
        xs=np.linspace(lo,hi,round((hi-lo)/.025)+1); ys=np.linspace(-4,4,321)
        xx, yy=np.meshgrid(xs,ys)
        zz=LinearNDInterpolator(points,v[indices,2])(xx,yy)
        # float32 source endpoints differ slightly from decimal task endpoints.
        from scipy.interpolate import NearestNDInterpolator
        missing=~np.isfinite(zz)
        zz[missing]=NearestNDInterpolator(points,v[indices,2])(xx[missing],yy[missing])
        bottom=float(zz.min()); scale=max(float(zz.max()-bottom),1e-6)
        ET.SubElement(assets,'hfield',name=f'bank{i}',nrow=str(len(ys)),ncol=str(len(xs)),size=f'{(hi-lo)/2} 4 {scale} .1')
        ET.SubElement(world,'geom',name=f'bank{i}',type='hfield',hfield=f'bank{i}',pos=f'{(lo+hi)/2} 0 {bottom}',friction='.8 .005 .0001',contype='1',conaffinity='15',rgba='.45 .38 .25 1')
        heights.append(((zz-bottom)/scale).astype(np.float32))
    lo,hi=terrain['land_bridge_y_interval_m']
    ET.SubElement(world,'geom',name='land_bridge',type='box',pos=f'{(front+far)/2} {(lo+hi)/2} -.05',size=f'{(far-front)/2} {(hi-lo)/2} .05',contype='1',conaffinity='15')
    path=output/'terrain.xml'; ET.ElementTree(root).write(path)
    model=mujoco.MjModel.from_xml_path(str(path))
    for i,z in enumerate(heights):
        adr=model.hfield_adr[i]; model.hfield_data[adr:adr+z.size]=z.ravel()
    mujoco.mj_saveModel(model,str(output/'terrain.mjb'),None)
    np.savez_compressed(output/'terrain_source.npz',vertices=vertices,triangles=triangles,**{f'hfield{i}':z for i,z in enumerate(heights)})
    return model


def measurement(data,idx):
    q=np.asarray(data.qpos); v=np.asarray(data.qvel); w,x,y,z=q[3:7]
    roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)); yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
    gyro=np.asarray(data.sensordata)[idx.sensor_addresses[5]:idx.sensor_addresses[5]+3]
    m=[-roll,-gyro[0],q[idx.steering_qpos_address],v[idx.steering_dof_address],gyro[2],v[idx.rearwheel_dof_address],v[idx.frontwheel_dof_address]]
    tensor=lambda a:torch.tensor(a,dtype=torch.float32)
    return tensor([m]),tensor([[q[0],q[1],yaw]]),tensor([v[0]*np.cos(yaw)+v[1]*np.sin(yaw)]),(roll,pitch,yaw)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry',type=Path,default=Path('learning/configs/playground_rb_skill_sources.json'))
    p.add_argument('--config',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skills',nargs='+',default=['source','rb_run1','rb_run2','rb_run3'])
    p.add_argument('--modes',nargs='+',choices=['h75','zero_middle'],default=['h75','zero_middle'])
    p.add_argument('--model',type=Path,help='Reuse an existing terrain MJB from this task')
    p.add_argument('--approach', choices=['native','h75'], default='h75')
    p.add_argument('--routes',nargs='+',type=int,default=[60,61,62]); p.add_argument('--seconds',type=float,default=10)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False); torch.set_num_threads(1)
    cfg=json.loads(a.config.read_text()); hand=cfg['ground_handoff']; terrain=cfg['ditch_terrain']
    bundle=Path(hand['bundle']); identity=json.loads((bundle/'identity.json').read_text()); hcfg=json.loads((bundle/'config.json').read_text())
    controller=json.loads(Path(json.loads(Path(hand['settings']).read_text())['controller_config']).read_text())
    skills={name:PendulumSkill(a.registry,name) for name in a.skills}
    original=next(iter(skills.values())).bundle; idx=original.model_index
    model=mujoco.MjModel.from_binary_path(str(a.model)) if a.model else transplant(original,terrain,a.output); idx=_build_index(model); geometry=build_geometry_contract(model); mm=mjx.put_model(model,impl='warp')
    roads=np.load(hand['reference_path']); n=len(a.routes)*len(skills)*len(a.modes)
    contract=dict(terrain_model=str(a.model) if a.model else str(a.output/'terrain.mjb'),role='bounded native Playground transfer DEV, no training',skills={k:s.record for k,s in skills.items()},terrain=terrain,source_config=str(a.config),routes=a.routes,episode_s=a.seconds,episodes=n,physics_steps_max=round(a.seconds/.005)*n,backend='MuJoCo Playground mjx_env.step, MJX-Warp',heightfield_contact_workspace=512,physics_dt=.005,lower_dt=.005,upper_dt=.02,ground_model=identity,goal=hand['goal'],road_half_width_m=hand['road_half_width_m'],start_x_m=1.5,start_vx_m_s=2.,handoff='first wheel contact on far bank after genuine airborne interval; next 5ms lower-control boundary',approach=a.approach,jump_start_x_m=hand['approach']['jump_start_x_m'],scope='Declared approach controller then native RB jump then H75 or zero-middle STTW. No planning or training.',terrain_conversion='Existing seeded terrain vertices resampled to 25mm heightfields, open trench and bridge retained; native vehicle and contacts retained, not a controlled simulator-only comparison.',pendulum='finite native position servos hold measured landing joint angles; no joint lock',observer='measured ESO initialization, empty H75 history at handoff')
    (a.output/'declaration.json').write_text(json.dumps(contract,indent=2))
    @jax.jit
    def step(data,ctrl):return mjx_env.step(mm,data,ctrl,1)
    @jax.jit
    def frame(data,history,last,signal):
        g=extract_geometry(data,geometry)
        geom=ObservableGeometry(g.obstacle_relative_x,data.qpos[2],g.roll,g.pitch,g.yaw)
        history=advance_history(history,observable_frame(data,idx,geom,last,jp.asarray(1.)))
        return history,actor_observation(history,signal)
    results=[]; start=time.time()
    for name,skill in skills.items():
      for route_id in a.routes:
       for mode in a.modes:
        label=f'{name}_route{route_id}_{mode}'; target=a.output/label; target.mkdir()
        qpos=np.array(model.key_qpos[idx.keyframe_id]); qvel=np.array(model.key_qvel[idx.keyframe_id]); qpos[0]=1.5; qvel[0]=2.
        data=mjx_env.make_data(model,qpos=jp.asarray(qpos),qvel=jp.asarray(qvel),impl='warp',naconmax=4096,njmax=256)
        data=jax.jit(mjx.forward)(mm,data)
        history=initial_history(); last=jp.zeros(4); obs=actor_observation(history,jp.asarray(False)); active=False; airborne=False; handoff=None; rows=[]; poses=[]; reason='timeout'; clearance=False
        policy=TorchPathPolicy(bundle/'actor.npz',identity['files']['actor.npz'],hcfg,1,'cpu'); ctl=TorchController(controller,1,'cpu')
        route=TorchPathReference({k:roads[k][route_id] if roads[k].ndim>({'xy':2}.get(k,1)) else roads[k] for k in ('s','xy','heading','curvature')},hcfg,'cpu')
        middle=TorchMiddleControl(policy,route,identity['frame_scales'],torch.tensor([.5]),torch.tensor([1.5]))
        target_ctrl=np.zeros(4,dtype=np.float32)
        approaching=a.approach=='h75'
        if approaching:
            m,pose,speed,_=measurement(data,idx); q=np.asarray(data.qpos)
            target_ctrl[2:]=q[[idx.hip_qpos_address,idx.knee_qpos_address]]
            middle.v_user.fill_(hand['approach']['speed_m_s']);middle.alpha.fill_(hand['approach']['alpha'])
            raw=torch.stack((speed,m[:,2]),-1);previous=torch.stack((m[:,3],m[:,5]),-1)
            nearest=((route.xy-pose[0,:2])**2).sum(-1).argmin().reshape(1)
            middle.reset(m,pose,speed,raw,raw,previous,ctl,initial_progress=route.s[nearest])
        for tick in range(round(a.seconds/.005)+1):
            m,pose,speed,angles=measurement(data,idx); q=np.asarray(data.qpos); v=np.asarray(data.qvel)
            if approaching and q[0]>=hand['approach']['jump_start_x_m'] and tick%4==0:
                approaching=False
                (target/'jump_entry.json').write_text(json.dumps(dict(time_s=tick*.005,qpos=q.tolist(),qvel=v.tolist(),physical_reset=False)))
            # Download actual solver contacts, not a height-only touchdown proxy.
            host=mjx.get_data(model,data); wheel_ids=(idx.frontwheel_geom_id,idx.rearwheel_geom_id)
            contact=[c for c in host.contact[:host.ncon] if c.dist<=0 and any(g in wheel_ids for g in c.geom)]
            airborne |= len(contact)==0 and q[0]>terrain['front_x_m']-1.5
            far_contact=any(c.pos[0]>terrain['front_x_m']+terrain['width_m'] for c in contact)
            if airborne and far_contact and not active and not approaching:
                active=True; clearance=True; middle.v_user.fill_(1.5);middle.alpha.fill_(.5); target_ctrl[2:]=q[[idx.hip_qpos_address,idx.knee_qpos_address]]
                target_ctrl=np.clip(target_ctrl,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
                ctl.reset(slice(None)); target_ctrl[0]=q[idx.steering_qpos_address]
                ctl.eso[0,:3]=m[0,[2,0,1]]; ctl.filtered_roll_rate[0]=m[0,1]
                raw=torch.stack((speed,m[:,2]),-1); previous=torch.stack((m[:,3],m[:,5]),-1)
                nearest=((route.xy-pose[0,:2])**2).sum(-1).argmin().reshape(1)
                middle.reset(m,pose,speed,raw,raw,previous,ctl,initial_progress=route.s[nearest])
                handoff=dict(time_s=tick*.005,qpos=q.tolist(),qvel=v.tolist(),physical_reset=False)
                (target/'handoff.json').write_text(json.dumps(handoff,indent=2))
            ey=float(np.min(np.linalg.norm(np.asarray(route.xy)-q[:2],axis=1)))
            poses.append(q.copy())
            rows.append([tick*.005,*q[:3],*angles,*v[:3],float(active),ey,*target_ctrl,float(approaching)])
            if not np.isfinite(q).all() or abs(angles[0])>.7 or abs(angles[1])>1.309:reason='physical_failure';break
            if (active or approaching) and (ey>hand['road_half_width_m'] or bool(middle.domain_exit[0])):reason='corridor_or_domain_exit';break
            goal=hand['goal']
            if active and goal['x_min_m']<=q[0]<=goal['x_max_m'] and abs(q[1])<=goal['abs_y_max_m'] and v[0]>=goal['min_vx_m_s']:reason='goal';break
            if tick==round(a.seconds/.005):break
            if active or approaching:
                governed=middle.before(pose,zero_middle=(mode=='zero_middle' and active))
                command=ctl.step(torch.stack((m[:,5]*.07,m[:,2],m[:,3],m[:,0],m[:,1],governed[:,1]),-1),torch.tensor([tick*.005>3.]))
                steer=float(command[0,0].clamp(-3,3)); target_ctrl[0]=np.clip(target_ctrl[0]+.005*steer,-.8,.8)
                target_ctrl[1]=float((governed[0,0]/.07).clamp(-60,60))
                target_ctrl=np.clip(target_ctrl,model.actuator_ctrlrange[:,0],model.actuator_ctrlrange[:,1])
            elif tick%4==0:
                last,ctrl=skill.infer(obs,q[idx.knee_qpos_address]); target_ctrl=np.asarray(ctrl).copy()
            data=step(data,jp.asarray(target_ctrl,dtype=jp.float32))
            if active or approaching:
                m2,pose2,speed2,_=measurement(data,idx)
                delivered=torch.tensor([[steer,target_ctrl[1]]],dtype=torch.float32)
                middle.after(m2,pose2,speed2,governed,delivered,torch.zeros(1,2),ctl)
            if (tick+1)%4==0:
                signal=jp.logical_and(data.qpos[0]>=9.5,data.qpos[0]<=11.)
                history,obs=frame(data,history,last,signal)
        np.savez_compressed(target/'trajectory.npz',trace=np.asarray(rows),qpos=np.asarray(poses),columns=['time','x','y','z','roll','pitch','yaw','vx','vy','vz','ground_active','path_distance','steer_target','rear_target','hip_target','knee_target','approaching'],reference_xy=np.asarray(route.xy))
        row=dict(skill=name,route=route_id,continuation=mode,termination=reason,goal=reason=='goal',airborne=bool(airborne),far_contact=bool(clearance),handoff_s=None if handoff is None else handoff['time_s'],duration_s=tick*.005,final_xy=q[:2].tolist())
        results.append(row); print(json.dumps(row),flush=True)
        (a.output/'status.json').write_text(json.dumps(dict(stage='running',completed=len(results),total=n,elapsed_s=time.time()-start,results=results),indent=2))
    (a.output/'status.json').write_text(json.dumps(dict(stage='complete',completed=len(results),total=n,elapsed_s=time.time()-start,results=results),indent=2))


if __name__=='__main__':main()
