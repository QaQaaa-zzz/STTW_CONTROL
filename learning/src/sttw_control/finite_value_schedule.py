"""Three-way paired exposure using the established complete-episode sampler."""
import copy

from .finite_value_online import FiniteValueLearner, ROLES
from .live_ppo_schedule import make_schedule as native_schedule
from .preference_best import config_hash


def make_schedule(templates, *, seed, updates):
    result = []
    for cases in native_schedule(templates, seed=seed, updates=updates):
        result.append([dict(copy.deepcopy(c), arm=role, profile=0)
                       for c in cases if c['arm'] == 'plain' for role in ROLES])
    return result


def make_evaluation_schedule(templates, *, seed):
    result = []
    for c in make_schedule(templates, seed=seed, updates=1)[0]:
        if c['arm'] != 'free_GAE' or (c['force_N'] == 0. and c['replicate'] != 0):
            continue
        base = {k: v for k, v in c.items() if k not in ['arm', 'profile']}
        for role in ['parent_plain', 'parent_causal', 'parent_support', *ROLES]:
            result.append(dict(base, actor_role=role,
                               profile=7 if role == 'parent_support' else 0,
                               causal_support=role == 'parent_causal'))
    return result


def fresh_learners(config, source, *, seed):
    if source['schema'] != 'sttw_dual_survival_boundary_v1' or source['config_sha256'] != config_hash(config):
        raise ValueError('native source/config identity mismatch')
    policy = {'actor.' + k: v for k, v in source['actor'].items()}
    policy.update({'critic.' + k: v for k, v in source['task_critic'].items()})
    policy['log_std'] = source['log_std']
    result = {}
    for role in ROLES:
        obj = FiniteValueLearner(config, policy, role=role, rate=.16, beta=0., seed=seed,
                                 epochs=4, fixed_denominator=30000, actor_lr=1e-4,
                                 critic_lr=3e-4, clip=.2, entropy=.001, grad_clip=1.,
                                 entropy_normalization='fixed_slots')
        obj.physical_critic.load_state_dict(copy.deepcopy(source['physical_critic']), strict=True)
        result[role] = obj
    return result
