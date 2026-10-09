"""Random speed/steer command tracking with bounded ECBC residuals, 200 Hz.

This task is independent of old geometric Actors and upper reference policies.
Body forward velocity is simulation-assisted; deployment requires an estimator.
"""
import math
import jax
import jax.numpy as jp
from flax import struct
from .teleop_env import TeleopEnv
from .closed_loop_kernel import controls
from .direct_command_policy import make_frame
from .direct_command_scenarios import publish_command
from .controller import _system,ControllerConfig

SCALES=jp.array([.3,1.5,.35,3.,3.,2.,3.,3.,3.,.35,.8,.45,3.,.35,3.,60.,1.5,10.,.3,.3])

def map_action(z):
    """Normalized front steering-rate / rear axle-rate residual, not references."""
    return jp.tanh(z)

def huber(x):
    a=jp.abs(x);return jp.where(a<=1.,x*x,2*a-1.)

def reward_terms(ev,ed,roll,action,previous,failed,spec,*,roll_rate=0.,remaining_ticks=None,peak_roll=None):
    if spec.get("candidate_local_interface"):
        return local_reward_terms(ev,ed,roll,action,previous,failed,spec,roll_rate=roll_rate,remaining_ticks=remaining_ticks,peak_roll=peak_roll)
    r=spec['reward'];dt=spec['dt']
    terms=dict(speed=r['speed_weight']*jp.exp(-(ev/r['speed_sigma'])**2)-r['error_weight']*huber(ev/r['speed_huber_scale']),
        steer=r['steer_weight']*jp.exp(-(ed/r['steer_sigma'])**2)-r['error_weight']*huber(ed/r['steer_huber_scale']),
        roll=-r['roll_weight']*huber(jp.maximum(jp.abs(roll)-r['roll_start'],0)/r['roll_scale']),
        action=-r['action_weight']*jp.sum(action**2),
        action_change=-r['action_change_weight']*jp.sum((action-previous)**2))
    terms={k:v*dt for k,v in terms.items()};terms['failure']=-r['failure_penalty']*failed
    terms['reward']=sum(terms.values());return terms

def command_rows(env_id,episode,speed,spec):
    if spec.get('candidate_local_interface'):
        return local_command_rows(env_id,episode,speed,spec)
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
        raw,rates,_=publish_local_command(snapshot.raw,rows,0,slew,self.spec)
        p=snapshot.replace(raw=raw,reference_pose=pose,yaw_unwrapped=pose[2],yaw_wrapped=pose[2],failed=jp.bool_(False))
        s=LowerState(p,jp.zeros((self.spec['history_frames'],20)),jp.zeros(self.spec['history_frames']),rows,slew,rates,jp.zeros(2),jp.zeros(2),jp.asarray(0.),jp.int32(0),env_id,episode)
        return self.record(s)
    def step(self,s,z):
        p=s.physical;raw=p.raw;action=map_action(z);m,_,_=self.physics.observe(p.data)
        override=controls(p.controller,p.actuator,m,raw,raw,p.physical_tick*self.cc.dt>3.,True,self.cc,self.ac,lower_action=action)
        end,log=self.physics._step(p,raw,True,exact_governed=True,control_override=override)
        terms=reward_terms(log['actual_forward_speed']-raw[0],log['actual_delta']-raw[1],log['phi'],action,s.previous_action,end.failed,self.spec,roll_rate=log['phi_dot'],remaining_ticks=round(self.spec['episode_seconds']/self.spec['dt'])-s.tick,peak_roll=log['peak_roll'])
        tick=s.tick+1;done=end.failed|(tick>=round(self.spec['episode_seconds']/self.spec['dt']))
        next_raw,rates,target=publish_local_command(raw,s.rows,tick,s.slew,self.spec)
        ns=s.replace(physical=end.replace(raw=next_raw),tick=tick,rates=rates,previous_action=action,previous_applied=log['applied_residual'],yaw_rate=(end.yaw_unwrapped-p.yaw_unwrapped)/self.cc.dt)
        ns=self.record(ns)
        log.update(reward=terms['reward'],reward_parts=terms,time=s.tick*self.cc.dt,done=done,action=action,target=target,raw_rates=s.rates)
        return ns,terms['reward'],done,log


# Opt-in local candidate; old lower_command configurations keep their behavior.
def local_roll_reference(speed,steer):
    _,_,_,ratio,_=_system(speed,jp.zeros(3),ControllerConfig())
    return steer/ratio


def local_steer_limit(speed,roll_limit):
    _,_,_,ratio,_=_system(speed,jp.zeros(3),ControllerConfig())
    return jp.minimum(.28,jp.abs(ratio)*roll_limit)


def local_command_rows(env_id,episode,speed,spec,*,case=None):
    key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(spec['command_seed']),env_id),episode)
    keys=jax.random.split(key,6);design=spec['command_design']
    sign=jp.where(jax.random.bernoulli(keys[0]),1.,-1.)
    family=jax.random.randint(keys[1],(),0,3)
    low=jax.random.uniform(keys[2],(),minval=1.7,maxval=2.1)
    high=jax.random.uniform(keys[3],(),minval=2.3,maxval=2.8)
    roll=jp.where(jax.random.uniform(keys[4])<design['rare_fraction'],design['rare_roll_reference'],design['ordinary_roll_reference'])
    small=jax.random.uniform(keys[5],(),minval=.02,maxval=.06)
    if case is not None:
        sign=jp.asarray(1. if case.endswith('_positive') else -1.)
        family=jp.asarray(0 if case.startswith('steady') else 2 if case.startswith('post_return_small') else 1)
        low,high,roll,small=map(jp.asarray,(1.9,2.6,.24,.04))
    turn=sign*local_steer_limit(low,roll)*.98
    rows=jp.zeros((16,3)).at[:,0].set(99.).at[0].set(jp.array([0.,speed,0.]))
    # Full physical turns induce the state deviations seen by return/small-turn phases.
    dynamic=rows.at[1].set(jp.array([.5,low,0.])).at[2].set(jp.array([2.,low,turn])).at[3].set(jp.array([5.,low,0.])).at[4].set(jp.array([7.,high,0.])).at[5].set(jp.array([9.,high,jp.where(family==2,sign*small,0.)]))
    steady=rows.at[1].set(jp.array([1.,high,sign*local_steer_limit(high,roll)*.75]))
    return jp.where(family==0,steady,dynamic),jp.array([.5,.2])


def publish_local_command(raw,rows,tick,slew,spec):
    proposed,rates,target=publish_command(raw,rows,tick,slew,spec['dt'])
    if not spec.get('candidate_local_interface'):return proposed,rates,target
    # Screen every published point, including coupled speed/steer slews.
    # First move steer toward a target safe at both current and target speed;
    # hold speed if its tentative increment would violate the declared envelope.
    bound=spec['command_design']['rare_roll_reference']
    limit=local_steer_limit(jp.maximum(raw[0],target[0]),bound)
    desired_delta=jp.clip(target[1],-limit,limit)
    delta=raw[1]+jp.clip(desired_delta-raw[1],-slew[1]*spec['dt'],slew[1]*spec['dt'])
    safe=jp.abs(local_roll_reference(proposed[0],delta))<=bound+1e-7
    speed=jp.where(safe,proposed[0],raw[0]);command=jp.stack([speed,delta])
    return command,(command-raw)/spec['dt'],target


def local_reward_terms(ev,ed,roll,action,previous,failed,spec,*,roll_rate,remaining_ticks,peak_roll):
    if remaining_ticks is None or peak_roll is None:raise ValueError('local reward needs interval peak and remaining finite ticks')
    r=spec['reward'];caps=r['caps']
    if abs(sum(caps.values())-r['failure_cost_rate_bound'])>1e-9:raise ValueError('component/failure bound mismatch')
    costs=dict(speed=r['speed_weight']*huber(ev/r['speed_huber_scale']),steer=r['steer_weight']*huber(ed/r['steer_huber_scale']),
        roll=r['roll_weight']*huber(jp.maximum(peak_roll-r['roll_start'],0)/r['roll_scale']),
        working_roll=r['working_roll_weight']*huber(jp.maximum(peak_roll-r['working_roll_start'],0)/r['working_roll_scale']),
        roll_rate=r['roll_rate_weight']*huber(jp.maximum(jp.abs(roll_rate)-r['roll_rate_start'],0)/r['roll_rate_scale']),
        action=r['action_weight']*jp.sum(action**2),action_change=r['action_change_weight']*jp.sum((action-previous)**2))
    terms={k:jp.where(failed,0.,-r['scale']*spec['dt']*jp.minimum(v,caps[k])) for k,v in costs.items()}
    gamma=spec['gamma'];n=jp.maximum(remaining_ticks,0)
    tail=n if gamma==1 else -jp.expm1(n*math.log(gamma))/(1-gamma)
    terms['failure']=jp.where(failed,-r['failure_extra']-r['scale']*spec['dt']*r['failure_cost_rate_bound']*tail,0.)
    terms['reward']=sum(terms.values());return terms
