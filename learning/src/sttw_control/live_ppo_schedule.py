"""Paired complete-episode exposure and explicit weights-only initialization."""
import copy
import numpy as np


def make_schedule(templates, *, seed, updates):
    if updates < 1 or len(templates) != 2 or {x['context'] for x in templates} != {'left','right'}:
        raise ValueError('positive updates and one complete template per left/right required')
    rng=np.random.default_rng(seed);schedule=[]
    for iteration in range(updates):
        cases=[];nonce=0
        for t in templates:
            for alpha in [0.,.25,.5,.75,1.]:
                for force in [0.,-32.,32.]:
                    for replicate in range(2):
                        case=dict(copy.deepcopy(t),alpha=alpha,force_N=force,pulse_duration_s=.1,
                            pulse_onset_task_tick=int(rng.integers(320,681)),rng_fold_in=nonce,
                            replicate=replicate,collection_index=iteration)
                        for name,profile in [('plain',0),('support',7)]:
                            cases.append(dict(case,arm=name,profile=profile))
                        nonce+=1
        schedule.append(cases)
    return schedule


def make_evaluation_schedule(templates, *, seed):
    """Four deterministic roles, one shared zero branch per route/alpha."""
    result=[]
    for c in make_schedule(templates,seed=seed,updates=1)[0]:
        if c['arm']!='plain' or (c['force_N']==0 and c['replicate']!=0):continue
        base={k:v for k,v in c.items() if k not in ['arm','profile']}
        for role,profile in [('parent_plain',0),('parent_support',7),('trained_plain',0),('trained_support',7)]:
            result.append(dict(base,actor_role=role,profile=profile))
    return result


def fresh_learners(config, source, *, seed):
    """Copy native heads/std, discard inherited Adam and all update counters."""
    from .survival_online_learner import OnlineSurvivalLearner
    from .preference_best import config_hash
    if source['schema']!='sttw_dual_survival_boundary_v1' or source['config_sha256']!=config_hash(config):
        raise ValueError('native source/config identity mismatch')
    policy={'actor.'+k:v for k,v in source['actor'].items()}
    policy.update({'critic.'+k:v for k,v in source['task_critic'].items()})
    policy['log_std']=source['log_std']
    result={}
    for name in ['plain','support']:
        obj=OnlineSurvivalLearner(config,policy,beta=0.,seed=seed,epochs=4,
            fixed_denominator=30000,actor_lr=1e-4,critic_lr=3e-4,clip=.2,entropy=.001,grad_clip=1.,entropy_normalization='fixed_slots')
        obj.physical_critic.load_state_dict(copy.deepcopy(source['physical_critic']),strict=True)
        result[name]=obj
    return result
