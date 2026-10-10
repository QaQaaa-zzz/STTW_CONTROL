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

def command_rows(env_id,episode,speed,spec,cc=None):
    if spec.get('candidate_local_interface'):
        return local_command_rows(env_id,episode,speed,spec,cc=cc)
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
    issued_stream:object=None
    task_labels:object=None
    episode_totals:object=None

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
        pose=self.physics.helpers.pose(snapshot.data);rows,slew=command_rows(env_id,episode,snapshot.raw[0],self.spec,self.cc)
        raw,rates,_=publish_local_command(snapshot.raw,rows,0,slew,self.spec,self.cc)
        p=snapshot.replace(raw=raw,reference_pose=pose,yaw_unwrapped=pose[2],yaw_wrapped=pose[2],failed=jp.bool_(False))
        s=LowerState(p,jp.zeros((self.spec['history_frames'],20)),jp.zeros(self.spec['history_frames']),rows,slew,rates,jp.zeros(2),jp.zeros(2),jp.asarray(0.),jp.int32(0),env_id,episode)
        if self.spec.get('command_design',{}).get('version')=='four_local_families_v2':
            _,_,labels=sample_local_v2(env_id,episode,snapshot.raw[0],self.spec,self.cc)
            s=s.replace(task_labels=labels,episode_totals=jp.zeros(6))
        return self.record(s)
    def step(self,s,z):
        p=s.physical;raw=p.raw;action=map_action(z);m,_,_=self.physics.observe(p.data)
        override=controls(p.controller,p.actuator,m,raw,raw,p.physical_tick*self.cc.dt>3.,True,self.cc,self.ac,lower_action=action)
        end,log=self.physics._step(p,raw,True,exact_governed=True,control_override=override)
        terms=reward_terms(log['actual_forward_speed']-raw[0],log['actual_delta']-raw[1],log['phi'],action,s.previous_action,end.failed,self.spec,roll_rate=log['phi_dot'],remaining_ticks=round(self.spec['episode_seconds']/self.spec['dt'])-s.tick,peak_roll=log['peak_roll'])
        tick=s.tick+1;done=end.failed|(tick>=round(self.spec['episode_seconds']/self.spec['dt']))
        from .direct_command_scenarios import publish_issued_reference
        _,_,target=publish_issued_reference(raw,s.issued_stream,s.tick,s.slew,self.spec['dt'],True) if s.issued_stream is not None else publish_local_command(raw,s.rows,s.tick,s.slew,self.spec,self.cc)
        next_raw,rates,_=publish_issued_reference(raw,s.issued_stream,tick,s.slew,self.spec['dt'],True) if s.issued_stream is not None else publish_local_command(raw,s.rows,tick,s.slew,self.spec,self.cc)
        ns=s.replace(physical=end.replace(raw=next_raw),tick=tick,rates=rates,previous_action=action,previous_applied=log['applied_residual'],yaw_rate=(end.yaw_unwrapped-p.yaw_unwrapped)/self.cc.dt)
        if s.episode_totals is not None:
            ev=log['actual_forward_speed']-raw[0];ed=log['actual_delta']-raw[1]
            totals=s.episode_totals+jp.array([terms['reward'],1.,ev**2,ed**2,log['peak_roll']>.302,0.])
            totals=totals.at[5].set(jp.maximum(s.episode_totals[5],log['peak_roll']))
            ns=ns.replace(episode_totals=totals)
            log['completed_episode_totals']=jp.where(done,totals,jp.zeros(6))
        ns=self.record(ns)
        log.update(reward=terms['reward'],reward_parts=terms,time=s.tick*self.cc.dt,done=done,action=action,target=target,raw_rates=s.rates)
        if s.task_labels is not None:log['task_labels']=s.task_labels
        return ns,terms['reward'],done,log


# Opt-in local candidate; old lower_command configurations keep their behavior.
def local_roll_reference(speed,steer,cc=None):
    _,_,_,ratio,_=_system(speed,jp.zeros(3),ControllerConfig() if cc is None else cc)
    return steer/ratio


def local_steer_limit(speed,roll_limit,cc=None,abs_limit=.28):
    _,_,_,ratio,_=_system(speed,jp.zeros(3),ControllerConfig() if cc is None else cc)
    return jp.minimum(abs_limit,jp.abs(ratio)*roll_limit)


def local_command_rows(env_id,episode,speed,spec,*,case=None,cc=None):
    if spec['command_design'].get('version')=='four_local_families_v2':
        rows,slew,_=sample_local_v2(env_id,episode,speed,spec,cc,case=case)
        return rows,slew
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


def publish_local_command(raw,rows,tick,slew,spec,cc=None):
    proposed,rates,target=publish_command(raw,rows,tick,slew,spec['dt'])
    if not spec.get('candidate_local_interface'):return proposed,rates,target
    # Screen every published point, including coupled speed/steer slews.
    # First move steer toward a target safe at both current and target speed;
    # hold speed if its tentative increment would violate the declared envelope.
    bound=spec['command_design']['rare_roll_reference']
    limit=local_steer_limit(jp.maximum(raw[0],target[0]),bound,cc,spec['command_design'].get('abs_steer_max_edge',.28))
    desired_delta=jp.clip(target[1],-limit,limit)
    delta=raw[1]+jp.clip(desired_delta-raw[1],-slew[1]*spec['dt'],slew[1]*spec['dt'])
    safe=jp.abs(local_roll_reference(proposed[0],delta,cc))<=bound+1e-7
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


def sample_local_v2(env_id,episode,speed,spec,cc,*,case=None):
    """Four declared families; labels are diagnostics only, never observations."""
    d=spec['command_design'];key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(spec['command_seed']),env_id),episode)
    keys=jax.random.split(key,20)
    uniform=lambda i,a,b,shape=():jax.random.uniform(keys[i],shape,minval=a,maxval=b)
    sign=jp.where(jax.random.bernoulli(keys[0]),1.,-1.)
    small_sign=jp.where(jax.random.bernoulli(keys[1]),1.,-1.)
    probabilities=jp.array(list(d['family_probabilities'].values()))
    family=jp.sum(uniform(2,0.,1.)>jp.cumsum(probabilities)).astype(jp.int32)
    edge=uniform(3,0.,1.)<d['edge_fraction'];dynamic=uniform(4,0.,1.)<d['dynamic_rate_fraction']
    vrange=jp.where(edge,jp.array(d['command_speed_edge']),jp.array(d['command_speed_core']))
    max_delta=jp.where(edge,d['abs_steer_max_edge'],d['abs_steer_max_core'])
    roll=jp.where(uniform(5,0.,1.)<d['rare_fraction'],uniform(6,d['ordinary_roll_reference'],d['rare_roll_reference']),uniform(6,*d['sample_target_roll_reference_abs_range']))
    low=uniform(7,jp.maximum(1.7,vrange[0]),jp.minimum(2.1,vrange[1]));high=uniform(8,jp.maximum(2.3,vrange[0]),jp.minimum(2.8,vrange[1]))
    small=uniform(9,*d['post_return_small_magnitude'])
    times=jp.array(list(d['base_event_times'].values()))+uniform(10,-d['time_jitter_abs_s'],d['time_jitter_abs_s'],(5,))
    times=times.at[4].set(jp.minimum(times[4],d['post_return_small_latest_request_s']))
    sv=jp.where(dynamic,jp.array(d['speed_slew_dynamic']),jp.array(d['speed_slew_normal']));sd=jp.where(dynamic,jp.array(d['steer_slew_dynamic']),jp.array(d['steer_slew_normal']))
    slew=jp.stack([uniform(11,sv[0],sv[1]),uniform(12,sd[0],sd[1])])
    if case is not None:
        family=jp.asarray(0 if case.startswith('steady') or case=='straight_speed_change' else 2 if case.startswith('post_return_small') else 1)
        small_sign=jp.asarray(1. if case.endswith('_positive') else -1.)
        sign=jp.where(family==2,-small_sign,small_sign)
        low,high,roll,small=map(jp.asarray,(1.9,2.6,.24,spec['validation']['post_small_target']))
        max_delta=jp.asarray(.28);times=jp.array([.5,2.,4.5,6.,spec['validation']['post_small_target_start_s']]);slew=jp.array([.5,.3])
    turn=sign*local_steer_limit(low,roll,cc,max_delta)
    rows=jp.zeros((16,3)).at[:,0].set(99.).at[0].set(jp.array([0.,speed,0.]))
    seq=rows.at[1].set(jp.array([times[0],low,0.])).at[2].set(jp.array([times[1],low,turn])).at[3].set(jp.array([times[2],low,jp.where(family==3,-turn,0.)])).at[4].set(jp.array([times[3],high,0.])).at[5].set(jp.array([times[4],high,jp.where(family==2,small_sign*small,0.)]))
    seq=seq.at[4].set(jp.where(family==3,jp.array([7.,low,0.]),seq[4]))
    hold=uniform(13,*d['ordinary_target_hold_s'],(13,));otimes=jp.concatenate([jp.zeros(1),jp.cumsum(hold)])
    ov=uniform(14,vrange[0],vrange[1],(14,));od=uniform(15,-1.,1.,(14,))*jp.minimum(d['ordinary_abs_steer_max'],jax.vmap(lambda v:local_steer_limit(v,roll,cc,max_delta))(ov))
    od=jp.where(uniform(16,0.,1.,(14,))<d['ordinary_straight_target_probability'],0.,od)
    ordinary=rows.at[:14].set(jp.stack([otimes,ov,od],axis=1)).at[0].set(jp.array([0.,speed,0.]))
    if case is not None:
        ordinary=rows.at[1].set(jp.array([1.,high,sign*local_steer_limit(high,roll,cc,max_delta)*.75]))
        if case=='straight_speed_change':ordinary=rows.at[1].set(jp.array([.5,low,0.])).at[2].set(jp.array([6.,high,0.]))
    labels=jp.array([family,edge,dynamic,sign,small_sign,roll,max_delta],dtype=jp.float32)
    return jp.where(family==0,ordinary,seq),slew,labels


def fixed_reference_stream(raw,spec,case,cc):
    """No physics: one immutable 5ms issued stream including exact derivatives."""
    rows,slew=local_command_rows(jp.int32(88001),jp.int32(0),raw[0],spec,case=case,cc=cc)
    def step(previous,tick):
        command,rates,_=publish_local_command(previous,rows,tick,slew,spec,cc)
        return command,jp.concatenate([jp.array([tick*spec['dt']]),command,rates])
    return jax.lax.scan(step,raw,jp.arange(round(spec['validation']['seconds']/spec['dt'])+1))[1]
