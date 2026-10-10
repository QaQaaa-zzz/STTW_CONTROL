"""Geometric path environment sharing unchanged plant and frozen local adapter.

Projection commits once after each actual lower tick. Observation is pure.
The legacy plant's time-integrated reference diagnostics are discarded here;
route geometry is immutable and only geometric errors enter this task.
"""
import jax
import jax.numpy as j
from flax import struct
from .teleop_env import TeleopEnv
from .closed_loop_kernel import controls,preview_controls
from .controller import _system
from .direct_command_policy import initial_history,make_frame,push_history,q_ratio
from .geometric_path import project,at,wrap
from .path_reference import pursuit,publish
from .path_command_policy import correction_tick,assemble_observation
from .path_command_reward import costs,failure_reward

def guard_lower_action(action,finite):
    """Preserve lower_fault while preventing invalid inference entering physics."""
    return j.where(finite,action,j.zeros_like(action))


@struct.dataclass
class PathState:
    physical: object
    lower: object
    history: object
    projection: object
    follower: object
    offsets: object
    filtered: object
    nominal_rates: object
    previous_bounded: object
    yaw_rate: object
    tick: object
    alpha: object
    fault: object
    domain_exit: object

class PathCommandEnv:
    task_mode='geometric_path'
    schema='sttw_geometric_path_actor351_v1'
    def __init__(self,config,plant_spec,lower):
        self.config=config;self.plant_spec=plant_spec;self.lower=lower
        self.physics=TeleopEnv(config={**plant_spec,'lower_internal_diagnostics':True},backend='mjx')
        self.cc=self.physics.cc;self.ac=self.physics.ac
        t=config['timing']
        actual=(float(self.physics.model.opt.timestep),self.physics.substeps,self.cc.dt,plant_spec['plant']['policy_dt_s'])
        expected=(t['physics_dt_s_expected'],t['physics_substeps_per_control_expected'],t['lower_dt_s'],t['upper_dt_s'])
        if actual!=expected or t['path_follower_dt_s']!=t['upper_dt_s']:
            raise ValueError(f'loaded timing {actual} != {expected}; model={self.physics.bundle.identity}')
        if abs(actual[0]*actual[1]-actual[2])>1e-12:raise ValueError('substeps mismatch')
        if not plant_spec['lower_reference_centered']:raise ValueError('lower must be reference centered')
        if self.cc.wheelbase!=config['follower']['wheelbase_m_expected'] or abs(self.cc.caster-float(j.deg2rad(config['follower']['caster_deg_expected'])))>1e-7:raise ValueError('geometry mismatch')
        self.frame_scales=j.array([f['scale'] for f in plant_spec['network']['frame_fields']])

    def reset(self,snapshot,path,v_user,alpha=0.):
        m,pose,v=self.physics.observe(snapshot.data)
        projection=project(path,pose[:2],j.array(0.),j.array(0.),self.config)
        follower=pursuit(path,projection.progress,pose,v,j.asarray(v_user),self.config)
        nominal,rates=publish(snapshot.raw,follower.command,self.config)
        p=snapshot.replace(raw=nominal,reference_pose=pose,yaw_wrapped=pose[2],yaw_unwrapped=pose[2])
        s=PathState(p,self.lower.initial(pose),initial_history(self.plant_spec),projection,follower,j.zeros(2),j.zeros(2),rates,j.zeros(2),j.array(0.),j.int32(0),j.asarray(alpha),j.bool_(False),follower.invalid)
        return self.record_frame(s)

    def record_frame(self,s):
        m,_,v=self.physics.observe(s.physical.data);_,_,a4,_,_=_system(m[5]*.1,s.physical.controller.gains,self.cc)
        frame=make_frame(measurement=m.at[4].set(s.yaw_rate),forward_speed=v,previous_governed=s.physical.governor.current_reference,previous_final_command=s.physical.actuator.previous,previous_bounded_residual=s.previous_bounded,raw=s.physical.raw,raw_rates=s.nominal_rates,eso_equilibrium_shift=-s.physical.controller.disturbance/a4,cc=self.cc)
        return s.replace(history=push_history(s.history,frame),fault=s.fault|~j.all(j.isfinite(frame)))

    def observation(self,s,path,v_user,horizon):
        _,pose,_=self.physics.observe(s.physical.data);p=s.projection
        _,_,k0=at(path,p.progress);_,_,k1=at(path,p.progress+.5);_,_,k2=at(path,p.progress+1.)
        eh=wrap(p.heading-pose[2]);b=s.follower.preview_body
        context=j.stack([s.alpha,p.cross_track,j.sin(eh),j.cos(eh),k0,k1,k2,b[0],b[1],j.asarray(v_user),s.offsets[0],s.offsets[1],s.filtered[0],s.filtered[1],s.follower.lookahead])
        obs,fault=assemble_observation(s.history.frames,s.history.mask,context,self.frame_scales,self.config)
        critic=j.concatenate([obs,j.array([(horizon-s.tick*self.cc.dt)/horizon])])
        return obs,critic,fault|s.fault

    def tick(self,s,z,path,v_user):
        p=s.physical;m,pose,v=self.physics.observe(p.data)
        filtered,offsets,governed,flags=correction_tick(s.filtered,s.offsets,z,p.raw,self.config)
        lower,action,obs,lf=self.lower.prepare_local(s.lower,m,pose,v,governed,p,self.cc)
        preview=preview_controls(p.controller,m,governed,governed,p.physical_tick*self.cc.dt>3.,self.cc)
        override=controls(p.controller,p.actuator,m,governed,governed,p.physical_tick*self.cc.dt>3.,False,self.cc,self.ac,lower_action=guard_lower_action(action,lf['finite']),previewed=preview)
        p2,log=self.physics._step(p,governed,False,exact_governed=True,control_override=override)
        _,post,_=self.physics.observe(p2.data);yaw_rate=(p2.yaw_unwrapped-p.yaw_unwrapped)/self.cc.dt
        projection=project(path,post[:2],s.projection.progress,j.linalg.norm(post[:2]-pose[:2]),self.config)
        eh=wrap(projection.heading-post[2]);_,_,k=at(path,projection.progress)
        ff=j.arctan(self.cc.wheelbase*k/j.cos(self.cc.caster));chi=j.clip((j.abs(ff/q_ratio(v_user,self.cc))-.18)/.12,0,1)
        c=costs(alpha=s.alpha,chi=chi,ev=log['actual_forward_speed']-v_user,ey=projection.cross_track,eh=eh,peak_roll=log['peak_roll'],roll_rate=log['phi_dot'],speed=log['actual_forward_speed'],offsets=offsets,offset_rate=j.zeros(2),config=self.config)
        lower=self.lower.finish_local(lower,log['applied_residual'],yaw_rate)
        goal_point,goal_heading,_=at(path,path.goal)
        goal_distance=j.dot(post[:2]-goal_point,j.array([j.cos(goal_heading),j.sin(goal_heading)]))
        # Keep plant bookkeeping anchor unchanged; it is never the task path.
        p2=p2.replace(reference_pose=p.reference_pose)
        next_tick=s.tick+1
        follower=jax.lax.cond(next_tick%4==0,lambda _:pursuit(path,projection.progress,post,log['actual_forward_speed'],v_user,self.config),lambda _:s.follower,None)
        nominal,rates=publish(p.raw,follower.command,self.config)
        log.update(time=s.tick*self.cc.dt,active_tick=j.bool_(True),v_user=v_user,path_cross_track=projection.cross_track,path_heading_error=eh,path_progress=projection.progress,path_projection_unclamped=projection.unclamped,path_curvature=k,
            reference_xy=projection.point,reference_yaw_unwrapped=projection.heading,e_psi_unwrapped=eh,along=j.array(0.),lateral=projection.cross_track,large_heading_debt=j.abs(eh)>j.pi,
            nominal=p.raw,nominal_rates=s.nominal_rates,pp_target=s.follower.command,preview_body=s.follower.preview_body,preview_world=s.follower.preview_world,lookahead=s.follower.lookahead,pp_curvature=s.follower.kappa,
            target_offset=flags['target'],filtered_offset=filtered,offsets=offsets,latent_z=z,lower_error=j.stack([log['actual_forward_speed'],log['actual_delta']])-governed,lower_observation=obs,lower_action=action,lower_fault=~lf['finite'],policy_fault=flags['policy_fault']|s.fault,domain_exit=follower.invalid,goal_section_signed_distance=goal_distance,
            reference_clip_channels=flags['reference_clip_channels'],tick_reward=c['reward'],raw_components=c['raw'],effective_components=c['effective'],component_capped=c['capped'],chi=chi)
        end=s.replace(physical=p2.replace(raw=nominal),lower=lower,projection=projection,follower=follower,filtered=filtered,offsets=offsets,nominal_rates=rates,previous_bounded=log['applied_residual'],yaw_rate=yaw_rate,tick=next_tick,fault=s.fault|~lf['finite']|flags['policy_fault'],domain_exit=s.domain_exit|follower.invalid)
        return end,log

    def set_log_template(self,s,path,v_user):
        template=jax.eval_shape(lambda x:self.tick(x,j.zeros(2),path,v_user),s)[1]
        self.zero_log=jax.tree.map(lambda x:j.zeros(x.shape,x.dtype),template)

    def policy_step(self,s,z,path,v_user,horizon):
        s=s.replace(fault=s.fault|~j.all(j.isfinite(z)))
        def step(st,_):
            stopped=st.physical.failed|st.fault|st.domain_exit
            return jax.lax.cond(stopped,lambda st:(st,self.zero_log),lambda st:self.tick(st,z,path,v_user),st)
        end,logs=jax.lax.scan(step,s,None,length=4)
        end=self.record_frame(end)
        rate=(end.offsets-s.offsets)/self.config['timing']['upper_dt_s'];c=self.config['path_reward']
        rate_cost=c['offset_rate_weight']*j.sum((rate/j.array(c['offset_rate_scales']))**2);effective=j.minimum(rate_cost,c['component_caps']['offset_rate'])
        logs['raw_components']['offset_rate']=j.full(4,rate_cost);logs['effective_components']['offset_rate']=j.full(4,effective);logs['component_capped']['offset_rate']=j.full(4,rate_cost>effective)
        logs['tick_reward']-=c['scale']*self.cc.dt*effective
        n=j.sum(logs['active_tick']);last=j.maximum(n-1,0);normal=j.sum(j.where(logs['active_tick'],logs['tick_reward'],0))
        reward=j.where(end.physical.failed,failure_reward(j.ceil(horizon/.02)-s.tick//4,self.config),normal)
        logs['scored_tick_reward']=j.where(end.physical.failed,j.zeros(4).at[last].set(reward),j.where(logs['active_tick'],logs['tick_reward'],0))
        return end,logs
