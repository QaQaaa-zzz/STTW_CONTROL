"""Final64 authentication and shared measured-sensor six-role evaluation."""
import copy
import numpy as np

from .finite_value_online import ROLES as TRAIN_ROLES

ROLES = ('parent_plain', 'parent_causal', 'parent_support', *TRAIN_ROLES)
CATEGORIES = ('safe_task', 'safe_task_failed', 'contact', 'unknown')


def confirmation_inventory(protocol, contexts):
    """Balanced new-geometry events; all three controls share every event."""
    rng = np.random.default_rng(protocol['seed'])
    cases = []
    for context, spec in contexts.items():
        for alpha in protocol['alphas']:
            for force in protocol['forces_N']:
                tick = int(rng.integers(*protocol['onset_upper_tick_range'])) * 4
                for role, mode in [('parent_plain', 0), ('parent_causal', 1), ('parent_support', 2)]:
                    event = f'F{force:g}_r0'
                    cases.append(dict(case_id=f'{context}/a{alpha:g}/{event}/{role}',
                        context=context, route=context, alpha=alpha, force_N=force,
                        pulse_onset_task_tick=tick, pulse_duration_s=protocol['pulse_duration_s'],
                        world_direction=spec['world_direction'], requested_speed_m_s=spec['speed_m_s'],
                        role=role, actor_role=role, mode_index=mode, profile=7 if mode == 2 else 0,
                        causal_support=mode == 1, event_id=event, replicate=0))
    return cases


def candidate_extension_cases(source_cases, candidate_roles):
    """Only missing policies; every event/initial context stays identical."""
    if set(candidate_roles) != {'free_GAE','finite_GAE','free_MC'}:
        raise ValueError('all three frozen final64 candidates required')
    events=[c for c in source_cases if c['role']=='parent_plain']
    if len(events)!=30 or len({(c['context'],c['alpha'],c['event_id']) for c in events})!=30:
        raise ValueError('thirty unique declared source events required')
    rows=[]
    for event in events:
        for role in candidate_roles:
            c=copy.deepcopy(event)
            c.update(role=role,actor_role=role,mode_index=0,profile=0,causal_support=False,
                case_id=event['case_id'].rsplit('/',1)[0]+'/'+role)
            rows.append(c)
    return rows



def supported_candidate_cases(source_cases, role):
    """One declared Actor with the existing measured gate, no event changes."""
    if role != 'free_GAE':
        raise ValueError('only predeclared free_GAE complementarity probe supported')
    rows=candidate_extension_cases(source_cases,['free_GAE','finite_GAE','free_MC'])
    rows=[c for c in rows if c['role']==role]
    for case in rows:
        case.update(mode_index=1,causal_support=True)
    return rows


def rebind_path_context(state, path, speed, env):
    """New task geometry only; retain all physical/lower/history/filter state."""
    import jax.numpy as j
    from .geometric_path import project
    from .path_reference import pursuit
    _, pose, actual_speed = env.physics.observe(state.physical.data)
    projection = project(path, pose[:2], j.float32(0), j.float32(0), env.config)
    follower = pursuit(path, projection.progress, pose, actual_speed, speed, env.config)
    return state.replace(projection=projection, follower=follower)


def endpoint_records(plan, status, pair, *, plan_sha256):
    if (status.get('phase') != 'completed' or status.get('completed_updates') != 64
            or plan.get('updates') != 64 or pair.get('completed_updates') != 64
            or pair.get('plan_sha256') != plan_sha256):
        raise ValueError('completed authenticated final64 required; no best/intermediate model')
    models = pair.get('models', [])
    if len(models) != 3 or {m['arm'] for m in models} != set(TRAIN_ROLES):
        raise ValueError('three unique final64 roles required')
    result = {role: copy.deepcopy(plan['model']) for role in ROLES[:3]}
    for model in models:
        role = model['arm']; mode, lam = TRAIN_ROLES[role]
        expected = dict(role=role, mode=mode, rate=.16, gae_lambda=lam,
                        gamma=.997, horizon_upper_steps=500)
        if model.get('profile') != 0 or model.get('value_contract') != expected:
            raise ValueError('final value/estimator/permission contract differs')
        result[role] = copy.deepcopy(model)
    return result


def evaluation_inventory(schedule):
    if len(schedule) != 300:
        raise ValueError('sealed 50 events by six roles required')
    cases = []; groups = {}
    for row in schedule:
        role = row['actor_role']
        if (role not in ROLES or row['profile'] != (7 if role == 'parent_support' else 0)
                or row['causal_support'] != (role == 'parent_causal')):
            raise ValueError('evaluation role/reference permissions mismatch')
        event = f'F{row["force_N"]:g}_r{row["replicate"]}'
        key = (row['context'], row['alpha'], event)
        group = groups.setdefault(key, {})
        if role in group:
            raise ValueError('duplicate matched evaluation cell')
        group[role] = row
        cases.append(dict(row, case_id=f'{key[0]}/a{key[1]:g}/{event}/{role}',
                          role=role, event_id=event, requested_speed_m_s=row['speed'],
                          mode_index={'parent_causal': 1, 'parent_support': 2}.get(role, 0)))
    if len(groups) != 50 or any(set(g) != set(ROLES) for g in groups.values()):
        raise ValueError('complete matched event groups required')
    for group in groups.values():
        common = [{k:v for k,v in row.items() if k not in ['actor_role','profile','causal_support']}
                  for row in group.values()]
        if any(row != common[0] for row in common):
            raise ValueError('paired roles must share full event/initial-context definition')
    return cases


def evaluation_profile(mode, gate_active):
    import jax.numpy as j
    return j.where((mode == 2) | ((mode == 1) & gate_active), 7, 0)


def make_evaluation_step(env, actor, *, horizon_s):
    """Mean Actor with original live permissions; pulse only enters physics.

    The causal parent uses the previously fixed measured-rate .8/.4, .6s gate.
    Neither event parameters nor future measurements enter the gate or Actor.
    """
    import jax.numpy as j
    from .anticipatory_reference import make_reference_step
    from .reference_reserve_gate import gate_step
    advance = make_reference_step(env, actor, horizon_s=horizon_s, allow_reentry=True)
    lower_bound = j.float32(env.config['residual_interface']['speed_reference_bounds_m_s'][0])
    dt = env.config['timing']['upper_dt_s']

    def step(carry, parameters, path, speed, pulse, mode):
        measured, _, _ = env.physics.observe(carry.state.physical.data)
        roll_rate = measured[1]
        relay, _, receipt = gate_step(carry.relay, j.zeros(2), rho=j.float32(0),
                                      roll_rate=roll_rate, mode=2, dt=dt)
        profile = evaluation_profile(mode, receipt['active'])
        s = carry.state
        stopped = s.physical.failed | s.fault | (s.tick >= round(horizon_s / .005))
        fault = ~stopped & (mode == 1) & receipt['fault']
        supplied = carry.replace(state=s.replace(fault=s.fault | fault), relay=relay)
        end, logs = advance(supplied, parameters, path, speed, pulse, profile, lower_bound, j.bool_(False))
        active = logs['active_tick']
        return end, dict(logs, relay_active=active & (mode == 1) & receipt['active'],
                         relay_trigger=active & (mode == 1) & receipt['trigger'],
                         relay_input_roll_rate=j.where(active, roll_rate, j.nan),
                         relay_remaining_s=j.where(active, relay.remaining_s, j.nan),
                         relay_profile=j.repeat(profile, 4),
                         decision_task_tick=j.repeat(s.tick, 4))
    return step


def verify_gate_log(raw, case):
    """Replay the old gate and check measured input precedes the new advance."""
    import jax
    import jax.numpy as j
    from .reference_reserve_gate import initial_gate,gate_step
    active=raw['active_tick'].astype(bool)
    ticks=np.flatnonzero(active & (raw['physical_tick'] % 4 == 0))
    rate=raw['relay_input_roll_rate'][ticks]
    def step(state,value):
        end,_,receipt=gate_step(state,j.zeros(2),rho=j.float32(0),roll_rate=value,mode=2,dt=.02)
        return end,(receipt['active'],receipt['trigger'],end.remaining_s)
    _,(gate,trigger,remaining)=jax.jit(lambda x:jax.lax.scan(step,initial_gate(),x))(j.asarray(rate))
    gate=np.asarray(gate);trigger=np.asarray(trigger);mode=case['mode_index']
    np.testing.assert_array_equal(raw['relay_active'][ticks],(mode==1)&gate)
    np.testing.assert_array_equal(raw['relay_trigger'][ticks],(mode==1)&trigger)
    np.testing.assert_allclose(raw['relay_remaining_s'][ticks],np.asarray(remaining),atol=1e-6)
    later=ticks[ticks>0]
    np.testing.assert_allclose(raw['relay_input_roll_rate'][later],raw['phi_dot'][later-1],atol=2e-6,rtol=2e-5)
    repeated=np.repeat(np.asarray(evaluation_profile(mode,gate)),4)
    expected=np.zeros_like(raw['relay_profile']);expected[:len(repeated)]=repeated
    np.testing.assert_array_equal(raw['relay_profile'][active],expected[active])


def paired_report(rows, *, roles=ROLES):
    if not rows or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('unique nonempty evaluation cases required')
    from .reference_task_feedback import PROBE_ROLES
    from .nominal_reference_feedback import NOMINAL_ROLES
    from .nominal_table_ablation import SHARED_ROLE,PARENT_ROLE
    from .zero_middle_feedback import ZERO_ROLES
    from .middle_channel_ablation import CHANNEL_ROLES
    PROBE_ROLES=PROBE_ROLES+NOMINAL_ROLES+(SHARED_ROLE,PARENT_ROLE)+ZERO_ROLES+CHANNEL_ROLES
    if not roles or not set(roles) <= set(ROLES+PROBE_ROLES):
        raise ValueError('known evaluation roles required')
    groups = {}; counts = {role:{cat:0 for cat in CATEGORIES} for role in roles}
    for row in rows:
        role = row['role']; category = row['category']
        if role not in roles or category not in CATEGORIES:
            raise ValueError('declared role and four-way outcome required')
        force = row.get('force_N',0.)
        key = (row['context'],row['alpha'],row['event_id'],force,
               row.get('pulse_onset_task_tick') if force else None,
               row.get('pulse_duration_s') if force else None,
               tuple(row.get('world_direction',[])) if force else None,
               row.get('requested_speed_m_s'))
        group=groups.setdefault(key,{})
        if role in group:raise ValueError('duplicate paired role')
        group[role]=row; counts[role][category]+=1
    if any(set(group)!=set(roles) for group in groups.values()):
        raise ValueError('all six roles required in each paired event')
    contrasts=[(role,base) for role in (*TRAIN_ROLES,*PROBE_ROLES) for base in ROLES[:3]]
    contrasts += [(role,'free_GAE') for role in PROBE_ROLES]
    contrasts += [('finite_GAE','free_GAE'),('finite_GAE','free_MC'),('free_MC','free_GAE'),
                  ('parent_causal','parent_plain'),('parent_support','parent_plain')]
    contrasts += [('free_GAE','zero_middle'),('feedback_010','zero_middle_feedback'),
                  ('zero_middle_feedback','zero_middle')]
    contrasts += [(a,b) for a,b in [('speed_only_feedback','zero_middle_feedback'),
                   ('steer_only_feedback','zero_middle_feedback'),('feedback_010','speed_only_feedback'),
                   ('feedback_010','steer_only_feedback')]]
    contrasts = [(a,b) for a,b in contrasts if a in roles and b in roles]
    comparisons={}
    for target,baseline in contrasts:
        result=dict(known=0,unknown=0,physical_gains=0,physical_losses=0,
                    task_gains=0,task_losses=0,details=[])
        for key,group in groups.items():
            a,b=group[target],group[baseline];unknown='unknown' in [a['category'],b['category']]
            p=None if unknown else int(a['category']!='contact')-int(b['category']!='contact')
            t=None if unknown else int(a['category']=='safe_task')-int(b['category']=='safe_task')
            result['unknown' if unknown else 'known']+=1
            if not unknown:
                result['physical_gains']+=int(p==1);result['physical_losses']+=int(p==-1)
                result['task_gains']+=int(t==1);result['task_losses']+=int(t==-1)
            result['details'].append(dict(context=key[0],alpha=key[1],event_id=key[2],
                target_case=a['case_id'],baseline_case=b['case_id'],physical_change=p,task_change=t,
                scored_cost_change=None if unknown else a['scored_cost']-b['scored_cost']))
        comparisons[target+'_vs_'+baseline]=result
    return dict(role_counts=counts,comparisons=comparisons)
