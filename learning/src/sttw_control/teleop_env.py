"""Oracle full-state teleop physics; no old task stepping, rewards or policies."""
from dataclasses import replace
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from .env import RecoveryEnv, TaskConfig, load_config
from .teleop_commands import load_teleop_config, CONFIG_PATH
from .controller import initial_controller
from .actuator import initial_actuator
from .governor_state import GovernorState,Snapshot
from .closed_loop_kernel import controls
from .teleop_reference import integrate,unwrap

class TeleopEnv:
    observation_mode='oracle_full_state'
    def __init__(self,config=None,backend='mjx'):
        self.spec=load_teleop_config() if config is None else config
        legacy=load_config(CONFIG_PATH.parent/'fixed_endpoint_directional_alpha0.json')
        self.cc=replace(legacy.controller,fixed_roll_reference=None)
        self.ac=legacy.actuator
        self.backend=backend
        self.helpers=RecoveryEnv(TaskConfig(controller=self.cc,actuator=self.ac,
            speed_reference=2.3,eso_start=3.,roll_failure=.7),backend=backend)
        self.model=self.helpers.model;self.bundle=self.helpers.bundle
        self.substeps=self.helpers.substeps
        expected=(.005,1.5,10.,3.,60.,.8,1.,1.,False,'additive',.408,25*np.pi/180)
        actual=(self.cc.dt,self.ac.steer_residual_scale,self.ac.rear_residual_scale,
            self.ac.steer_rate_limit,self.ac.rear_rate_limit,self.ac.steer_limit,
            self.ac.strength,self.ac.base_output_scale,self.ac.project_base,self.ac.composition,
            self.cc.wheelbase,self.cc.caster)
        if actual!=expected:raise ValueError(f'frozen parameter mismatch: {actual} != {expected}')
        self._controls=jax.jit(lambda cs,ac,m,c,g,e,b:controls(cs,ac,m,c,g,e,b,self.cc,self.ac))
        self.step=jax.jit(self._step) if backend=='mjx' else self._step

    def initial(self):
        # Fresh physics only; never RecoveryEnv.reset/_prepare (which updates ESO).
        m=self.model;b=self.bundle;raw=jp.array([2.3,0.])
        qpos=jp.asarray(m.qpos0,dtype=raw.dtype)
        qvel=jp.zeros(m.nv).at[0].set(2.3).at[b.rear_dof].set(-23.).at[b.front_dof].set(-23.)
        ctrl=jp.zeros(m.nu).at[1].set(-23.)
        if self.backend=='cpu':
            data=mujoco.MjData(m);data.qpos[:]=qpos;data.qvel[:]=qvel;data.ctrl[:]=ctrl
            mujoco.mj_forward(m,data)
        else:
            data=self.helpers._mjx.make_data(m,impl='jax').replace(qpos=qpos,qvel=qvel,ctrl=ctrl)
            data=jax.jit(lambda d:self.helpers._mjx.forward(self.helpers.mjx_model,d))(data)
        p=self.helpers.pose(data)
        governor=GovernorState(raw,raw,jp.int32(0),jp.asarray(0.,dtype=raw.dtype),jp.int32(0),jp.int32(0))
        return Snapshot(data,initial_controller(self.cc),initial_actuator(self.ac,23.),
            jp.int32(0),raw,p,p[2],p[2],governor,jp.bool_(False))

    def observe(self,data):
        m=self.helpers.measure(data);p=self.helpers.pose(data)
        speed=jp.dot(jp.asarray(data.qvel[:2]),jp.array([jp.cos(p[2]),jp.sin(p[2])]))
        return m,p,speed

    def _step(self,s,goal,bypass,exact_governed=False,control_override=None):
        h=self.helpers;dt=self.cc.dt
        m,_,_=self.observe(s.data)
        current=s.governor.current_reference
        governed=jp.where(bypass,s.raw,current+jp.clip(goal-current,-jp.array([1.,.6])*dt,jp.array([1.,.6])*dt))
        governed=jp.where(exact_governed,goal,governed)
        if control_override is None:
            cs,act,log=self._controls(s.controller,s.actuator,m,s.raw,governed,
                s.physical_tick*dt>3.,bypass)
        else:cs,act,log=control_override
        final=act.previous;ctrl=jp.array([0.,-final[1],final[0],final[0]])
        if self.backend=='cpu':
            d=mujoco.MjData(self.model);mujoco.mj_copyData(d,self.model,s.data)
            d.ctrl[:]=np.asarray(ctrl);d.xfrc_applied[:]=0;d.qfrc_applied[:]=0
            contact=False;peak_roll=0.;peak_rate=0.;finite=True
            for _ in range(self.substeps):
                mujoco.mj_step(self.model,d)
                contact=contact or bool(h._contact_failure(d))
                mm=np.asarray(h.measure(d));peak_roll=max(peak_roll,abs(mm[0]));peak_rate=max(peak_rate,abs(mm[1]))
                finite=finite and np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() and np.isfinite(d.act).all()
            mujoco.mj_forward(self.model,d)
        else:
            d=s.data.replace(ctrl=ctrl,xfrc_applied=jp.zeros_like(s.data.xfrc_applied),qfrc_applied=jp.zeros_like(s.data.qfrc_applied))
            def substep(_,carry):
                d,contact,pr,pd,finite=carry
                d=h._mjx.step(h.mjx_model,d);mm=h.measure(d)
                finite=finite&jp.all(jp.isfinite(d.qpos))&jp.all(jp.isfinite(d.qvel))&jp.all(jp.isfinite(d.act))
                return d,contact|h._contact_failure(d),jp.maximum(pr,jp.abs(mm[0])),jp.maximum(pd,jp.abs(mm[1])),finite
            initial=(d,jp.bool_(False),jp.asarray(0.,dtype=d.qpos.dtype),jp.asarray(0.,dtype=d.qpos.dtype),jp.bool_(True))
            if self.spec.get('lower_internal_diagnostics',False):
                limits=jp.asarray(self.model.actuator_forcerange);limited=jp.asarray(self.model.actuator_forcelimited)
                def diagnostic_substep(i,carry):
                    physical,lo,hi,hits=carry;physical=substep(i,physical);force=physical[0].actuator_force
                    at_limit=limited&((force<=limits[:,0]+1e-5)|(force>=limits[:,1]-1e-5))
                    return physical,jp.minimum(lo,force),jp.maximum(hi,force),hits+at_limit.astype(jp.int32)
                result,force_min,force_max,force_hits=jax.lax.fori_loop(0,self.substeps,diagnostic_substep,
                    (initial,jp.full(self.model.nu,jp.inf),jp.full(self.model.nu,-jp.inf),jp.zeros(self.model.nu,jp.int32)))
                d,contact,peak_roll,peak_rate,finite=result
                log.update(actuator_force_min=force_min,actuator_force_max=force_max,actuator_force_limit_substeps=force_hits)
            else:
                d,contact,peak_roll,peak_rate,finite=jax.lax.fori_loop(0,self.substeps,substep,initial)
            d=h._mjx.forward(h.mjx_model,d)
        if self.spec.get('lower_internal_diagnostics',False):
            log.update(actuator_force_end=d.actuator_force,actuator_ctrl=ctrl)
        m,p,speed=self.observe(d)
        peak_roll=jp.maximum(peak_roll,jp.abs(m[0]));peak_rate=jp.maximum(peak_rate,jp.abs(m[1]))
        finite=finite&jp.all(jp.isfinite(m))&jp.all(jp.isfinite(cs.eso))&jp.isfinite(cs.disturbance)&jp.all(jp.isfinite(cs.gains))
        failed=s.failed|~finite|contact|h._contact_failure(d)|(peak_roll>.7)
        ref=integrate(s.reference_pose,s.raw,dt,self.cc.wheelbase,self.cc.caster)
        yaw=unwrap(s.yaw_unwrapped,s.yaw_wrapped,p[2]);error=ref[2]-yaw
        delta_xy=p[:2]-ref[:2]
        along=jp.dot(delta_xy,jp.array([jp.cos(ref[2]),jp.sin(ref[2])]))
        lateral=jp.dot(delta_xy,jp.array([jp.sin(ref[2]),-jp.cos(ref[2])]))
        gov=s.governor.replace(current_reference=governed,last_goal=goal)
        state=s.replace(data=d,controller=cs,actuator=act,physical_tick=s.physical_tick+1,
            reference_pose=ref,yaw_wrapped=p[2],yaw_unwrapped=yaw,governor=gov,failed=failed)
        log.update(actual_forward_speed=speed,wheel_speed_proxy=m[5]*.1,slip_proxy=m[5]*.1-speed,
            actual_delta=m[2],actual_delta_rate=m[3],phi=m[0],phi_dot=m[1],
            yaw_wrapped=p[2],yaw_unwrapped=yaw,reference_yaw_unwrapped=ref[2],e_psi_unwrapped=error,
            actual_xy=p[:2],reference_xy=ref[:2],along=along,lateral=lateral,
            limited_command=s.raw,governed=governed,goal=goal,final_ctrl=ctrl,
            omega_c=s.raw[0]*jp.cos(self.cc.caster)*jp.tan(s.raw[1])/self.cc.wheelbase,
            physical_failure=failed,nonfinite=~finite,contact_failure=jp.asarray(contact),
            peak_roll=peak_roll,peak_roll_rate=peak_rate,large_heading_debt=jp.abs(error)>jp.pi)
        return state,log

    def from_mjx(self,s,mjx_env):
        if self.backend!='cpu':raise ValueError('CPU audit destination required')
        d=mjx_env.helpers._mjx.get_data(self.model,s.data)
        return s.replace(data=d)
