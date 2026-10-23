"""Real prefix-bank recovery snippets with original task clocks and timeout masks."""
import jax
import jax.numpy as j
from flax import struct
from .path_command_training import Episode
from .path_command_reward import configured_cost_keys
from .tolerance_pulse import FORCES,force_probabilities


def segment_flags(flags, elapsed_ticks, maximum_ticks):
    timeout=(elapsed_ticks>=maximum_ticks)&~flags['done']
    return dict(flags,done=flags['done']|timeout,segment_timeout=timeout)


def bootstrap_segment_rewards(reward, final_value, timeouts, gamma):
    import torch
    return reward+gamma*torch.where(timeouts,final_value.reshape_as(reward),torch.zeros_like(reward))


@struct.dataclass
class RecoveryEpisode:
    core: object
    origin_tick: object
    bank_index: object
    force: object
    force_index: object
    pulse: object
    fallback: object
    impulse: object

    @property
    def state(self):return self.core.state
    @property
    def route(self):return self.core.route
    @property
    def episode(self):return self.core.episode
    @property
    def env_id(self):return self.core.env_id


class ShortRecoveryBatch:
    def __init__(self,base,bank,*,guided,seed,reserve_config,duration_s=4.):
        self.base=base;self.guided=guided;self.seed=seed;self.reserve_config=reserve_config
        self.maximum_ticks=round(duration_s/.005)
        if base.horizon!=20. or self.maximum_ticks!=800:
            raise ValueError('short recovery preserves original20s task, with4s snippets')
        import numpy as np
        valid=np.flatnonzero(bank['valid'])
        if not len(valid):raise ValueError('no valid prefix states')
        self.original_indices=j.asarray(valid)
        self.states=jax.tree.map(lambda x:j.asarray(x)[valid],bank['states'])
        self.routes=jax.tree.map(lambda x:j.asarray(x)[valid],bank['routes'])
        self.size=len(valid)

    def reset(self,eid,episode):
        from .tolerance_shaping import measurement_reserve
        key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(self.seed),eid),episode)
        choose,enable,amplitude=jax.random.split(key,3)
        idx=jax.random.randint(choose,(),0,self.size)
        state=jax.tree.map(lambda x:x[idx],self.states)
        route=jax.tree.map(lambda x:x[idx],self.routes)
        m,pose,_=self.base.env.physics.observe(state.physical.data)
        reserve=measurement_reserve(m,state.physical.controller,self.base.env.cc,
            remaining=20.-state.tick*.005,config=self.reserve_config)
        probabilities,fallback=force_probabilities(reserve['plus_N'],reserve['minus_N'],reserve['valid'],guided=self.guided)
        force_id=jax.random.categorical(amplitude,j.log(probabilities))
        enabled=jax.random.bernoulli(enable,.8)
        force=j.where(enabled,FORCES[force_id],0.)
        vector=force*j.array([-j.sin(pose[2]),j.cos(pose[2]),0.])
        start=state.physical.physical_tick*.005
        pulse=j.concatenate([j.array([start,start+.1]),vector])
        core=Episode(state,route,eid,episode,j.zeros(6),j.zeros(len(configured_cost_keys(self.base.config))))
        return RecoveryEpisode(core,state.tick,self.original_indices[idx],force,
            j.where(enabled,force_id+1,0),pulse,fallback&enabled,j.zeros(3))

    def observation(self,ep):return self.base.observation(ep.core)

    def advance_one(self,ep,z):
        core,_,reward,flags,final,stats=self.base.advance_one(ep.core,z,auto_reset=False,force_pulse=ep.pulse)
        elapsed=core.state.tick-ep.origin_tick
        flags=segment_flags(flags,elapsed,self.maximum_ticks)
        ended=ep.replace(core=core,impulse=ep.impulse+stats['applied_force_impulse'])
        started=ep.state.tick==ep.origin_tick
        stats['pulse']=dict(fired_now=started&(ep.force!=0),force=ep.force,
            fallback_now=started&ep.fallback,completed=flags['done'],enabled=ep.force!=0,
            completed_impulse=j.where(flags['done'],ended.impulse,0.),
            completed_expected_impulse=j.where(flags['done'],.1*ep.pulse[2:],0.))
        stats['recovery']=dict(started=started,
            force=ep.force,force_index=ep.force_index,bank_index=ep.bank_index,
            fallback=ep.fallback,elapsed_ticks=elapsed,segment_timeout=flags['segment_timeout'],
            completed=flags['done'],impulse=ended.impulse,expected_impulse=.1*ep.pulse[2:],
            family=ep.route.family,active_ticks=stats['active_ticks'])
        nxt=jax.lax.cond(flags['done']&~flags['engineering_fault'],
            lambda _:self.reset(ep.env_id,ep.episode+1),lambda _:ended,None)
        from .path_command_statistics import merge_statistics,reset_statistics
        stats['diagnostics']=merge_statistics(stats['diagnostics'],reset_statistics(nxt.route,flags['done']&~flags['engineering_fault']))
        return nxt,self.observation(nxt),reward,flags,final,stats
