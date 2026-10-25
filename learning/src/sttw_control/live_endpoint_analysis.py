"""Explicit paired DEV outcomes and descriptive tracking-band return times."""
import numpy as np


ROLES=('parent_plain','parent_support','trained_plain','trained_support')
CATEGORIES=('safe_task','safe_task_failed','contact','unknown')
COMPARISONS=(('trained_plain','parent_plain'),('trained_support','parent_support'),
             ('trained_support','trained_plain'),('parent_support','parent_plain'))


def block_reward_series(data,*,config):
    """Preserve partial final blocks and explicit terminal replacement reward."""
    ids=np.flatnonzero(np.asarray(data['active_tick'],bool))
    if not len(ids) or not np.array_equal(ids,np.arange(len(ids))):raise ValueError('complete active prefix required')
    components=np.asarray(data['effective_costs'],float)
    reward=np.asarray(data['scored_tick_reward'],float)
    if not np.isfinite(components[ids]).all() or not np.isfinite(reward[ids]).all():raise ValueError('nonfinite active rewards')
    blocks=np.unique(ids//4);dt=config['timing']['lower_dt_s'];scale=config['path_reward']['scale']
    groups=[ids[ids//4==b] for b in blocks]
    scored=np.array([reward[g].sum() for g in groups])
    signed=np.array([(-scale*dt*components[g]).sum(0) for g in groups])
    return dict(time_s=np.array([g[-1]*dt for g in groups]),scored_reward=scored,
        signed_running_components=signed,terminal_replacement_reward=scored-signed.sum(1))


def paired_endpoint_report(rows):
    if not rows or len({r['case_id'] for r in rows})!=len(rows):
        raise ValueError('nonempty unique actual case identities required')
    groups={};counts={role:{cat:0 for cat in CATEGORIES} for role in ROLES}
    for row in rows:
        if row['role'] not in ROLES or row['category'] not in CATEGORIES:raise ValueError('unknown role/category')
        counts[row['role']][row['category']]+=1
        # Pulse metadata at zero force has no physical meaning for pairing.
        key=(row['context'],row['alpha'],row['event_id'],row['force_N'],
             row['pulse_onset_task_tick'] if row['force_N'] else None,row['pulse_duration_s'] if row['force_N'] else None,
             tuple(row.get('world_direction',[])) if row['force_N'] else None,row.get('requested_speed_m_s'))
        group=groups.setdefault(key,{})
        if row['role'] in group:raise ValueError('duplicate matched role')
        group[row['role']]=row
    if any(set(g)!=set(ROLES) for g in groups.values()):raise ValueError('incomplete or mismatched four-role cases')
    comparisons={};pairs=[]
    safe=lambda r:r['category'] in ('safe_task','safe_task_failed')
    for target,base in COMPARISONS:
        name=target+'_vs_'+base
        tally=dict(declared_pairs=len(groups),known_pairs=0,unknown_pairs=0,physical_gains=0,physical_losses=0,task_gains=0,task_losses=0)
        for group in groups.values():
            a,b=group[target],group[base];unknown='unknown' in (a['category'],b['category'])
            physical=None if unknown else int(safe(a))-int(safe(b))
            task=None if unknown else int(a['category']=='safe_task')-int(b['category']=='safe_task')
            tally['unknown_pairs' if unknown else 'known_pairs']+=1
            if not unknown:
                tally['physical_gains']+=int(physical==1);tally['physical_losses']+=int(physical==-1)
                tally['task_gains']+=int(task==1);tally['task_losses']+=int(task==-1)
            cost=None if a.get('scored_cost') is None or b.get('scored_cost') is None else a['scored_cost']-b['scored_cost']
            pairs.append(dict(comparison=name,case_id=a['case_id'],baseline_case_id=b['case_id'],
                context=a['context'],alpha=a['alpha'],event_id=a['event_id'],force_N=a['force_N'],
                physical_change=physical,task_change=task,scored_cost_change=cost,unknown=unknown))
        comparisons[name]=tally
    return dict(role_counts=counts,comparisons=comparisons,pairs=pairs,
        independent_unit='Two old route contexts and one paired training repeat; branches are not independent replicates.')


def tracking_recovery_time(data,*,speed,onset_tick,config):
    """Descriptive return after first post-onset violation, not causal recovery.

    Uses original task bands/hold duration. A return can precede later contact;
    contact and full-task outcomes must therefore remain separate in reports.
    """
    active=np.asarray(data['active_tick'],bool);at=np.flatnonzero(active)
    clock=np.asarray(data['physical_tick'])[active];dt=config['timing']['lower_dt_s']
    if not np.array_equal(at,np.arange(len(at))) or not np.array_equal(clock,np.arange(len(clock))):
        raise ValueError('complete non-reset active clock prefix required')
    band=config['stage_B_baseline'];hold=round(band['hold_s']/dt)
    if hold<1 or not np.isclose(hold*dt,band['hold_s']):raise ValueError('hold must match lower tick')
    values=[np.asarray(data[k],float)[active] for k in ['path_cross_track','path_heading_error','actual_forward_speed']]
    if not all(np.isfinite(v).all() for v in values):raise ValueError('nonfinite tracking observations')
    good=(abs(values[0])<=band['final_y_error_m'])&(abs(values[1])<=band['final_path_heading_error_rad'])&(
        abs(values[2]-speed)<=band['final_speed_error_m_s'])
    bad=np.flatnonzero((clock>=onset_tick)&~good)
    result=dict(status='not_observed' if not np.any(clock>=onset_tick) else 'not_disrupted',
        first_violation_s=None,recovery_start_s=None,confirmed_at_s=None,delay_from_onset_s=None)
    if not len(bad):return result
    start=int(bad[0]);result.update(status='not_recovered',first_violation_s=float(start*dt))
    if len(good)<hold:return result
    total=np.r_[0,np.cumsum(good.astype(int))]
    starts=np.flatnonzero((total[hold:]-total[:-hold])==hold)
    starts=starts[starts>start]
    if len(starts):
        i=int(starts[0]);result.update(status='recovered',recovery_start_s=float(i*dt),
            confirmed_at_s=float((i+hold)*dt),delay_from_onset_s=float((i-onset_tick)*dt))
    return result
