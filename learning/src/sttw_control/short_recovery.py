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
        if hasattr(self,'alpha_values'):
            stats['recovery']['alpha_index']=j.searchsorted(j.asarray(self.alpha_values),ep.state.alpha)
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


def check_preference_batch(plan_path,output):
    """One real-size512-lane eight-step engineering run, without optimization."""
    import json,pickle,time,os
    from pathlib import Path
    import numpy as np
    from .preference_best import atomic_json
    from .path_command_training import PathBatch,load_environment
    from .path_command_policy import make_path_actor
    from .tolerance_training import SPEC,validate_preference_training_plan
    from .runtime import configure_compilation_cache
    out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();ticks=0
    status=dict(state='running',phase='running',pid=os.getpid(),training_steps=0,maximum_lower_ticks=16384)
    def update(**kw):
        status.update(kw);status.update(checked_unix=time.time(),elapsed_s=time.monotonic()-started,active_lower_ticks=ticks)
        atomic_json(out/'status.json',status)
    update(stage='loading')
    try:
        plan=json.loads(Path(plan_path).read_text());cfg=json.loads(Path(plan['config']).read_text())
        parent=json.loads(Path(plan['parent_manifest']).read_text());bankdir=Path(plan['bank_dir'])
        bm=json.loads((bankdir/'manifest.json').read_text());validate_preference_training_plan(plan,cfg,parent,bm)
        with (bankdir/'bank.pkl').open('rb') as stream:bank=pickle.load(stream)
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,snapshot=load_environment(cfg,parent['source'],'terminal_contract_v2')
        base=PathBatch(env,snapshot,cfg,plan['seed'],0.)
        batch=PreferenceRecoveryBatch(base,bank,arm=plan['arm'],seed=plan['pulse_seed'],forces=plan['forces_N'],effort_budget=SPEC['effort_budget'],gamma=SPEC['gamma'])
        initial=batch.reset(j.int32(0),j.int32(0));env.set_log_template(initial.state,initial.route.path,initial.route.speed)
        episodes=jax.jit(jax.vmap(batch.reset))(j.arange(512,dtype=j.int32),j.zeros(512,j.int32))
        with Path(plan['parent_record']['actor']).open('rb') as stream:params=jax.tree.map(j.asarray,pickle.load(stream))
        actor=make_path_actor(cfg)
        def step(ep):
            obs=jax.vmap(batch.observation)(ep);action=actor.apply(params,obs[0])
            end,ob,reward,flags,final,stats=batch.advance_batch(ep,action)
            checks=j.array([j.any(obs[2])|j.any(flags['engineering_fault'])|j.any(ob[2]),j.sum(flags['physical_failure']),j.sum(flags['tracking_domain_failure'])])
            return end,checks,j.sum(stats['active_ticks']),ob[0].shape[-1],ob[1].shape[-1]
        advance=jax.jit(step);physical=domain=0
        for index in range(8):
            update(stage='stepping',completed_policy_steps=index)
            episodes,checks,count,actor_dim,critic_dim=advance(episodes);checks=jax.device_get(checks);ticks+=int(count)
            if checks[0]:raise RuntimeError('engineering fault in conditional batch')
            physical+=int(checks[1]);domain+=int(checks[2])
        ep=jax.device_get(episodes);complete=(np.asarray(ep.episode)==0)&(np.asarray(ep.force)!=0)
        if not complete.any():raise RuntimeError('no complete pulse in engineering run')
        error=np.linalg.norm(np.asarray(ep.impulse)[complete]-.1*np.asarray(ep.pulse)[complete,2:],axis=-1)
        if ticks>16384 or int(actor_dim)!=351 or int(critic_dim)!=352:raise RuntimeError('batch contract changed')
        atomic_json(out/'receipt.json',dict(batch=512,steps=8,active_lower_ticks=ticks,actor_dim=int(actor_dim),critic_dim=int(critic_dim),original_task_horizon_s=base.horizon,snippet_s=4.,physical_failures=physical,domain_failures=domain,complete_nonzero_pulses=int(complete.sum()),max_impulse_error_Ns=float(error.max()),training_steps=0,arm=plan['arm']))
        update(state='completed',phase='completed',stage='engineering_ready',completed_policy_steps=8)
    except BaseException as exc:
        update(state='error',phase='error',error=f'{type(exc).__name__}: {exc}');raise
