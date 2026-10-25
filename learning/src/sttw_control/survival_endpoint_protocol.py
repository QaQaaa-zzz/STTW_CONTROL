"""Predeclared endpoint assignments and inference semantics, without physics.

Metadata validation is only one part of authenticating an evaluation: callers
must also verify files, model exports, full source identities and raw receipts.
This module neither selects checkpoints nor launches or resumes training.
"""
import math

ROLES = ('parent', 'task_only', 'physical_credit', 'parent_speed_floor')
ALPHAS = (0., .25, .5, .75, 1.)


def make_endpoint_cases(physical_contexts, nominal_contexts):
    """488 real episodes; zero correction has no fictitious alpha replication.

    Four physical contexts must form two zero/nonzero-force pairs. Routes and
    speeds describe original requested tasks, never governed references. Four
    nominal contexts supply authenticated geometry names and requested speeds.
    """
    if len(physical_contexts) != 4 or len(nominal_contexts) != 4:
        raise ValueError('four physical and four nominal contexts required')
    pairs = {}
    for c in physical_contexts:
        required = ('route', 'speed', 'force_N', 'pulse_onset_task_tick',
                    'pulse_duration_s', 'world_direction')
        if any(k not in c for k in required):
            raise ValueError('complete original pulse context required')
        numeric = [c['speed'], c['force_N'], c['pulse_onset_task_tick'],
                   c['pulse_duration_s'], *c['world_direction']]
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError('finite physical contexts required')
        if (c['speed'] <= 0 or c['pulse_duration_s'] <= 0
                or len(c['world_direction']) != 3
                or c['pulse_onset_task_tick'] < 0
                or int(c['pulse_onset_task_tick']) != c['pulse_onset_task_tick']):
            raise ValueError('positive speed/duration and legal pulse geometry required')
        pairs.setdefault((c['route'], c['speed']), []).append(c)
    if len(pairs) != 2:
        raise ValueError('physical zero/force contexts must be paired by route and speed')
    for group in pairs.values():
        if len(group) != 2 or sum(c['force_N'] == 0 for c in group) != 1:
            raise ValueError('one paired zero and one nonzero force required')
        for key in ('pulse_onset_task_tick', 'pulse_duration_s', 'world_direction'):
            if group[0][key] != group[1][key]:
                raise ValueError('paired pulse timing and direction must match')
    nominal_keys = set()
    for c in nominal_contexts:
        if not math.isfinite(c['speed']) or c['speed'] <= 0 or c.get('force_N', 0.) != 0:
            raise ValueError('finite positive unperturbed nominal context required')
        nominal_keys.add((c['route'], c['speed']))
    if len(nominal_keys) != 4:
        raise ValueError('four unique nominal contexts required')
    route_groups = {key: i for i, key in enumerate(pairs)}
    result = []
    for panel, contexts, stochastic in (
            ('physical_deterministic', physical_contexts, False),
            ('physical_stochastic', physical_contexts, True),
            ('nominal_deterministic', nominal_contexts, False)):
        for ci, context in enumerate(contexts):
            # Whitelist only task/pulse fields; discard inherited training arm,
            # alpha, and RNG labels rather than carrying stale identities.
            task = {k: context[k] for k in ('route', 'speed', 'force_N',
                    'pulse_onset_task_tick', 'pulse_duration_s', 'world_direction') if k in context}
            task.setdefault('force_N', 0.)
            for role in ROLES + (() if stochastic else ('zero_middle',)):
                alphas = (None,) if role == 'zero_middle' else ALPHAS
                for alpha in alphas:
                    for rep in range(4 if stochastic else 1):
                        group = route_groups[(context['route'], context['speed'])] if stochastic else ci
                        result.append(dict(task,
                            case_id=f'{panel}/{ci}/{role}/{alpha}/{rep}', panel=panel,
                            context_index=ci, role=role, alpha=alpha, replicate=rep,
                            inference_mode='zero' if role == 'zero_middle' else ('gaussian' if stochastic else 'mean'),
                            # Same base seed is supplied by the sealed evaluation
                            # declaration. No model, alpha or force fold-in here.
                            rng_fold_in=group*4+rep if stochastic else None,
                            speed_reference_floor_m_s=1.5 if role == 'parent_speed_floor' else None))
    return result


def validate_endpoint_metadata(plan, pair, states, *, plan_sha256, config_sha256):
    """Reject incomplete/selected/wrong-lineage pairs before evaluating them.

    config_sha256 is the canonical semantic config hash, deliberately distinct
    from plan['effective_config_sha256'], which hashes serialized file bytes.
    Actual artifact SHA/export verification remains the caller's responsibility.
    """
    updates = plan['updates']
    if updates != 64 or plan['critic_only_updates'] != 2 or plan['epochs'] != 4:
        raise ValueError('this evaluation requires the declared 64-update endpoint schedule')
    if (pair['completed_updates'] != updates or pair['next_noise_update_index'] != updates
            or pair['plan_sha256'] != plan_sha256
            or pair['base_noise_seed'] != plan['seed']
            or pair['source_initialization_sha256'] != plan['model']['checkpoint_sha256']):
        raise ValueError('endpoint completion or training lineage mismatch')
    expected_arms = {'task_only': 0., 'physical_credit': 1.}
    if plan['arms'] != expected_arms or set(states) != set(expected_arms):
        raise ValueError('both declared matched arms required')
    records = pair['models']
    if len(records) != 2 or {r['arm'] for r in records} != set(expected_arms):
        raise ValueError('exactly one authenticated record for each arm required')
    actor_steps = (updates-plan['critic_only_updates'])*plan['epochs']
    critic_steps = updates*plan['epochs']
    for record in records:
        arm = record['arm']; state = states[arm]
        expected_settings = {k: plan[k] for k in ('epochs', 'actor_lr', 'critic_lr',
                             'clip', 'entropy', 'grad_clip', 'fixed_denominator')}
        expected_settings.update(beta=expected_arms[arm], seed=plan['learner_seed'])
        if (record['next_collection'] != updates
                or state['schema'] != 'sttw_dual_survival_boundary_v1'
                or not state['exact_boundary_resume']
                or state['next_collection'] != updates
                or state['actor_steps'] != actor_steps or state['critic_steps'] != critic_steps
                or state['optimizer_steps'] != [actor_steps, critic_steps, critic_steps]
                or state['config_sha256'] != config_sha256
                or state['settings'] != expected_settings):
            raise ValueError(f'{arm}: endpoint optimizer/config/settings mismatch')
    return dict(completed_updates=updates, actor_steps_per_arm=actor_steps,
                critic_steps_per_head_per_arm=critic_steps, scope='metadata only; authenticate artifacts separately')


def choose_evaluation_action(mean, log_std, key, *, mode):
    """Explicit inference mode; deterministic outputs have no Gaussian density.

    Does not govern references. The speed-floor comparator must constrain the
    reference at the declared interface, never the requested task or raw action.
    """
    import jax.numpy as j
    from .survival_collection import sample_gaussian_action
    if mode == 'gaussian':
        action, logp, next_key = sample_gaussian_action(mean, log_std, key)
        return action, logp, j.bool_(True), next_key
    if mode not in ('mean', 'zero'):
        raise ValueError('explicit mean, zero, or gaussian inference mode required')
    action = mean if mode == 'mean' else j.zeros_like(mean)
    return action, j.asarray(j.nan, dtype=mean.dtype), j.bool_(False), key
