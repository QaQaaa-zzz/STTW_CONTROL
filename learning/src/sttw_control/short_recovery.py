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


def reset_finished_batch(batch,ended,final,flags):
    """Scalar guard outside vmap: skip bank gathers on nonterminal batches.

    Mixed terminal batches still generate candidates for every lane. Stateless
    fold_in(eid, episode) keys and masked selection preserve the original resets.
    Engineering faults are never reset, and final observations precede reset.
    """
    mask=flags['done']&~flags['engineering_fault']
    def install(_):
        candidates=jax.vmap(batch.reset)(ended.env_id,ended.episode+1)
        initial=jax.vmap(batch.observation)(candidates)
        def select(new,old):
            return j.where(mask.reshape(mask.shape+(1,)*(old.ndim-1)),new,old)
        return jax.tree.map(select,candidates,ended),jax.tree.map(select,initial,final)
    return jax.lax.cond(j.any(mask),install,lambda _:(ended,final),None)


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
        self._install_bank(bank)

    def _install_bank(self,bank):
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

    def advance_one(self,ep,z,*,auto_reset=True):
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
        if not auto_reset:return ended,final,reward,flags,final,stats
        def install(_):
            fresh=self.reset(ep.env_id,ep.episode+1)
            return fresh,self.observation(fresh)
        nxt,obs=jax.lax.cond(flags['done']&~flags['engineering_fault'],
            install,lambda _:(ended,final),None)
        from .path_command_statistics import merge_statistics,reset_statistics
        stats['diagnostics']=merge_statistics(stats['diagnostics'],reset_statistics(nxt.route,flags['done']&~flags['engineering_fault']))
        return nxt,obs,reward,flags,final,stats

    def advance_batch(self,episodes,z):
        """Use this directly, rather than vmap(advance_one), in the trainer."""
        from .path_command_statistics import merge_statistics,reset_statistics
        end,_,reward,flags,final,stats=jax.vmap(
            lambda ep,action:self.advance_one(ep,action,auto_reset=False))(episodes,z)
        nxt,obs=reset_finished_batch(self,end,final,flags)
        counts=jax.vmap(reset_statistics)(nxt.route,flags['done']&~flags['engineering_fault'])
        stats['diagnostics']=merge_statistics(stats['diagnostics'],counts)
        return nxt,obs,reward,flags,final,stats


class PreferenceRecoveryBatch(ShortRecoveryBatch):
    """Exact-alpha matched state selection with independently drawn forces.

    Reuses the existing scalar-reset/short-timeout mechanics; never modifies
    captured alpha, physical state, history or original task clocks.
    """
    def __init__(self,base,bank,*,arm,seed,forces,effort_budget,gamma,
                 duration_s=4.,strength=.7,bandwidth=.2,shuffle_seed=918204):
        import numpy as np
        from .margin_state_sampling import exact_alpha_groups,bank_risks,state_probabilities,stratified_permutation
        if arm not in ('uniform','guided','shuffled'):raise ValueError('unknown matched state sampler arm')
        self.base=base;self.seed=seed;self.arm=arm;self.maximum_ticks=round(duration_s/.005)
        if not 0<duration_s<=base.horizon:raise ValueError('invalid recovery snippet')
        forces=np.asarray(forces,np.float32)
        if forces.ndim!=1 or not len(forces) or not np.isfinite(forces).all() or np.any(forces==0) or len(np.unique(forces))!=len(forces):raise ValueError('unique finite nonzero force levels required')
        self.forces=j.asarray(forces);self._install_bank(bank)
        values,self.groups=exact_alpha_groups(np.asarray(self.states.alpha));self.alpha_values=values
        self.risks=bank_risks(self.states,base.env.cc,self.forces,effort_budget=effort_budget,gamma=gamma,duration=.1,horizon=base.horizon)
        self.permutation=stratified_permutation(np.asarray(self.groups),shuffle_seed)
        risk=self.risks[self.permutation] if arm=='shuffled' else self.risks
        self.probabilities=state_probabilities(risk,self.groups,strength=0. if arm=='uniform' else strength,bandwidth=bandwidth,num_groups=len(values))

    def reset(self,eid,episode):
        from .margin_state_sampling import draw
        key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(self.seed),eid),episode)
        picked=draw(key,self.probabilities,self.groups)
        idx=picked['state_index'];state=jax.tree.map(lambda x:x[idx],self.states)
        route=jax.tree.map(lambda x:x[idx],self.routes)
        force=j.where(picked['enabled'],self.forces[picked['force_index']-1],0.)
        pose=self.base.env.physics.helpers.pose(state.physical.data)
        vector=force*j.array([-j.sin(pose[2]),j.cos(pose[2]),0.])
        start=state.physical.physical_tick*.005
        pulse=j.concatenate([j.array([start,start+.1]),vector])
        core=Episode(state,route,eid,episode,j.zeros(6),j.zeros(len(configured_cost_keys(self.base.config))))
        fallback=picked['enabled']&~j.isfinite(self.risks[idx,picked['force_index']-1])
        return RecoveryEpisode(core,state.tick,self.original_indices[idx],force,picked['force_index'],pulse,fallback,j.zeros(3))
