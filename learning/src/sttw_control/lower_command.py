"""Random speed/steer command tracking with bounded ECBC residuals, 200 Hz.

This task is independent of old geometric Actors and upper reference policies.
Body forward velocity is simulation-assisted; deployment requires an estimator.
"""
import jax
import jax.numpy as jp
from flax import struct
from .teleop_env import TeleopEnv
from .closed_loop_kernel import controls
from .direct_command_policy import make_frame
from .direct_command_scenarios import publish_command
from .controller import _system

SCALES=jp.array([.3,1.5,.35,3.,3.,2.,3.,3.,3.,.35,.8,.45,3.,.35,3.,60.,1.5,10.,.3,.3])

def map_action(z):
    """Normalized front steering-rate / rear axle-rate residual, not references."""
    return jp.tanh(z)

def huber(x):
    a=jp.abs(x);return jp.where(a<=1.,x*x,2*a-1.)

def reward_terms(ev,ed,roll,action,previous,failed,spec):
    r=spec['reward'];dt=spec['dt']
    terms=dict(speed=r['speed_weight']*jp.exp(-(ev/r['speed_sigma'])**2)-r['error_weight']*huber(ev/r['speed_huber_scale']),
        steer=r['steer_weight']*jp.exp(-(ed/r['steer_sigma'])**2)-r['error_weight']*huber(ed/r['steer_huber_scale']),
        roll=-r['roll_weight']*huber(jp.maximum(jp.abs(roll)-r['roll_start'],0)/r['roll_scale']),
        action=-r['action_weight']*jp.sum(action**2),
        action_change=-r['action_change_weight']*jp.sum((action-previous)**2))
    terms={k:v*dt for k,v in terms.items()};terms['failure']=-r['failure_penalty']*failed
    terms['reward']=sum(terms.values());return terms

def command_rows(env_id,episode,speed,spec):
    key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(spec['command_seed']),env_id),episode)
    kt,kv,kd,km,ks=jax.random.split(key,5)
    times=jp.concatenate([jp.zeros(1),jp.cumsum(jax.random.uniform(kt,(12,),minval=spec['hold_s'][0],maxval=spec['hold_s'][1]))])
    v=jax.random.uniform(kv,(13,),minval=spec['command_speed'][0],maxval=spec['command_speed'][1]).at[0].set(speed)
    modes=jax.random.uniform(km,(13,));p=spec['command_probabilities']
    amplitude=jp.where(modes<p[0],spec['command_steer_narrow'],jp.where(modes<p[0]+p[1],spec['command_steer_wide'],0.))
    d=(jax.random.uniform(kd,(13,),minval=-1.,maxval=1.)*amplitude).at[0].set(0.)
    rows=jp.stack([jp.where(times<spec['tail_start_s'],times,99.),v,d],axis=1)
    # Fixed final row comes after every active random target in publication order.
    rows=jp.concatenate([rows,jp.array([[99.,0.,0.],[99.,0.,0.],[spec['tail_start_s'],v[-1],0.]])],axis=0)
    ks1,ks2=jax.random.split(ks)
    slew=jp.stack([jax.random.uniform(ks1,(),minval=spec['speed_slew'][0],maxval=spec['speed_slew'][1]),jax.random.uniform(ks2,(),minval=spec['steer_slew'][0],maxval=spec['steer_slew'][1])])
    return rows,slew

def observation_from_history(frames,mask,tick,spec):
    obs=jp.concatenate([jp.clip(frames/SCALES,-5.,5.).reshape(-1),mask])
    return obs,jp.concatenate([obs,jp.array([1.-tick/(spec['episode_seconds']/spec['dt'])])])

@struct.dataclass
class LowerState:
    physical:object
    frames:object
    mask:object
    rows:object
    slew:object
    rates:object
    previous_action:object
    previous_applied:object
    yaw_rate:object
    tick:object
    env_id:object
    episode:object

class LowerCommandEnv:
    def __init__(self,spec):
        self.spec=spec;self.physics=TeleopEnv();self.cc=self.physics.cc;self.ac=self.physics.ac
    def observation(self,s):return observation_from_history(s.frames,s.mask,s.tick,self.spec)
    def record(self,s):
        m,_,v=self.physics.observe(s.physical.data);m=m.at[4].set(s.yaw_rate)
        _,_,a4,_,_=_system(m[5]*.1,s.physical.controller.gains,self.cc)
        frame=make_frame(measurement=m,forward_speed=v,previous_governed=s.physical.governor.current_reference,
          previous_final_command=s.physical.actuator.previous,previous_bounded_residual=s.previous_applied,
          raw=s.physical.raw,raw_rates=s.rates,eso_equilibrium_shift=-s.physical.controller.disturbance/a4,cc=self.cc)
        return s.replace(frames=jp.concatenate([s.frames[1:],frame[None]]),mask=jp.concatenate([s.mask[1:],jp.ones(1)]))
    def reset(self,snapshot,env_id,episode):
        pose=self.physics.helpers.pose(snapshot.data);rows,slew=command_rows(env_id,episode,snapshot.raw[0],self.spec)
        raw,rates,_=publish_command(snapshot.raw,rows,0,slew,self.cc.dt)
        p=snapshot.replace(raw=raw,reference_pose=pose,yaw_unwrapped=pose[2],yaw_wrapped=pose[2],failed=jp.bool_(False))
        s=LowerState(p,jp.zeros((self.spec['history_frames'],20)),jp.zeros(self.spec['history_frames']),rows,slew,rates,jp.zeros(2),jp.zeros(2),jp.asarray(0.),jp.int32(0),env_id,episode)
        return self.record(s)
    def step(self,s,z):
        p=s.physical;raw=p.raw;action=map_action(z);m,_,_=self.physics.observe(p.data)
        override=controls(p.controller,p.actuator,m,raw,raw,p.physical_tick*self.cc.dt>3.,True,self.cc,self.ac,lower_action=action)
        end,log=self.physics._step(p,raw,True,exact_governed=True,control_override=override)
        terms=reward_terms(log['actual_forward_speed']-raw[0],log['actual_delta']-raw[1],log['phi'],action,s.previous_action,end.failed,self.spec)
        tick=s.tick+1;done=end.failed|(tick>=round(self.spec['episode_seconds']/self.spec['dt']))
        next_raw,rates,target=publish_command(raw,s.rows,tick,s.slew,self.cc.dt)
        ns=s.replace(physical=end.replace(raw=next_raw),tick=tick,rates=rates,previous_action=action,previous_applied=log['applied_residual'],yaw_rate=(end.yaw_unwrapped-p.yaw_unwrapped)/self.cc.dt)
        ns=self.record(ns)
        log.update(reward=terms['reward'],reward_parts=terms,time=s.tick*self.cc.dt,done=done,action=action,target=target,raw_rates=s.rates)
        return ns,terms['reward'],done,log
