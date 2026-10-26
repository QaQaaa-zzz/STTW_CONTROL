"""Post-pulse common-state diagnostic, not a deployable recovery selector.

All native history/filter/ESO/actuator/relay state is retained. Only the existing
neural reference-channel mask changes for one fixed short interval. Prefix
capture uses declared experimental timing; it is deliberately privileged.
"""
import copy

import jax
import jax.numpy as j

ROLES=('full_current', 'pause_neural_steer', 'pause_neural_both')
MASKS=((True,True), (True,False), (False,False))
PAUSE_LOWER_TICKS=80


def inventory(original):
    """All120 source endpoint events, not only selected gains or failures."""
    prefixes=[copy.deepcopy(c) for c in original if c['role']=='goal_full_feedback' and c['alpha'] in (0.,1.)]
    if len(prefixes)!=120 or len({c['case_id'] for c in prefixes})!=120:
        raise ValueError('all120 unique full-current endpoint prefixes required')
    contexts={c['context'] for c in prefixes}
    if len(contexts)!=12:
        raise ValueError('original twelve complete entry contexts required')
    for context in contexts:
        events=[c for c in prefixes if c['context']==context]
        if {(c['alpha'],c['force_N']) for c in events}!={(a,f) for a in (0.,1.) for f in (0.,-16.,16.,-32.,32.)}:
            raise ValueError('complete nominal and four-force endpoint panel required')
    suffixes=[]
    for prefix_index,c in enumerate(prefixes):
        if c['source_keep_channels']!=[True,True] or c['feedback_gain_rad']!=.1 or not c['goal_preview'] or not c['causal_support']:
            raise ValueError('unchanged original full-current control stack required')
        if c['pulse_duration_s']!=.1 or c['pulse_onset_task_tick']%4:
            raise ValueError('original .1s upper-grid pulse timing required')
        fork=c['pulse_onset_task_tick']+20
        if fork%4 or fork+PAUSE_LOWER_TICKS>=2000:
            raise ValueError('full intervention and original remaining horizon required')
        c['fork_task_tick']=fork
        for role,mask in zip(ROLES,MASKS):
            new=copy.deepcopy(c)
            new.update(case_id=c['case_id'].rsplit('/',1)[0]+'/'+role,role=role,source_case_id=c['case_id'],
                       prefix_index=prefix_index,pause_mask=list(mask),pause_lower_ticks=PAUSE_LOWER_TICKS,
                       common_state_branch=True)
            suffixes.append(new)
    return prefixes,suffixes


def clone_complete_carry(carry, branches=3):
    """Literal whole-pytree copies, with no field reconstruction or clock reset."""
    if branches!=3:
        raise ValueError('only the three predeclared branches are supported')
    leaves=jax.tree.leaves(carry)
    if not leaves or any(x.ndim<1 or x.shape[0]!=leaves[0].shape[0] for x in leaves):
        raise ValueError('entire carry must have a common leading prefix dimension')
    return jax.tree.map(lambda x:j.repeat(x,branches,axis=0),carry)


def prefix_guard(advance, zero_log):
    """Freeze each real prefix at its own upper-step boundary; no fake tick."""
    def step(carry,parameters,path,speed,pulse,mode,fork_tick):
        ready=(carry.state.tick>=fork_tick)|carry.state.physical.failed|carry.state.fault
        return jax.lax.cond(ready,lambda _:(carry,zero_log),
                            lambda _:advance(carry,parameters,path,speed,pulse,mode),None)
    return step


def pause_step(advance):
    """Mask existing neural latent channels only, retaining all filters and gates."""
    def step(carry,parameters,path,speed,pulse,mode):
        tick=carry.state.tick
        inside=(tick>=parameters['fork_task_tick'])&(tick<parameters['fork_task_tick']+PAUSE_LOWER_TICKS)
        keep=j.where(inside,parameters['pause_mask'],j.array([True,True]))
        supplied=dict(parameters,actor=dict(native=parameters['actor']['native'],keep_channels=keep))
        end,logs=advance(carry,supplied,path,speed,pulse,mode)
        logs.update(branch_fork_task_tick=j.repeat(parameters['fork_task_tick'],4),
                    branch_pause_active=j.repeat(inside,4))
        return end,logs
    return step


def expected_mask(ticks,fork_tick,pause_mask):
    import numpy as np
    ticks=np.asarray(ticks)
    inside=(ticks>=fork_tick)&(ticks<fork_tick+PAUSE_LOWER_TICKS)
    return np.where(inside[:,None],np.asarray(pause_mask,bool),np.ones(2,bool))
