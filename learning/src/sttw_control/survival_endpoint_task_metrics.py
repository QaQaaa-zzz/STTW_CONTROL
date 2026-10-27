"""Offline original task/working-band metrics on an explicit physical episode.

No task errors are inferred during held-reference physics. These descriptive
labels do not impose a new adoption gate or change the survival objective.
"""
import math
import numpy as np


def goal_arrival_metrics(physical, *, goal_xy, radius_m, dt, horizon_s):
    """First entry into the ORIGINAL goal disk before contact/fault.

    No heading/speed/hold or follower-domain condition is imposed. Arrival is
    evaluated from measured sample positions; no interpolation is invented.
    A later fall does not erase arrival and must be reported separately.
    """
    if not all(math.isfinite(x) and x > 0 for x in [radius_m, dt, horizon_s]):
        raise ValueError('positive finite radius, dt and horizon required')
    goal = np.asarray(goal_xy, float)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError('finite original XY goal required')
    active = np.asarray(physical['active_tick'], bool)
    clock = np.asarray(physical['physical_tick'])
    xy = np.asarray(physical['actual_xy'], float)
    if active.ndim != 1 or clock.shape != active.shape or xy.shape != (len(active), 2):
        raise ValueError('aligned physical clock and XY required')
    ticks = clock[active]
    limit = round(horizon_s / dt)
    if not math.isclose(limit * dt, horizon_s, abs_tol=1e-8):
        raise ValueError('horizon must align to physical ticks')
    if not np.array_equal(ticks, np.arange(len(ticks))) or len(ticks) > limit:
        raise ValueError('complete observed clock prefix required')
    contact = active & np.asarray(physical['contact_failure'], bool)
    fault = active & (np.asarray(physical['nonfinite'], bool) |
                      physical['lower_fault'] | physical['policy_fault'])
    # Finite positions before an explicit fault are required for an answer.
    fault = fault | (active & ~np.isfinite(xy).all(axis=1))
    fatal = np.flatnonzero(contact | fault)
    stop = int(fatal[0]) if fatal.size else len(active)
    valid = active & (np.arange(len(active)) < stop)
    distance = np.linalg.norm(xy[valid] - goal, axis=1)
    hits = np.flatnonzero(distance <= radius_m)
    arrived = bool(hits.size)
    if arrived:
        outcome = 'arrived'
    elif fatal.size:
        outcome = 'unknown' if fault[stop] else 'contact_before_arrival'
    else:
        outcome = 'not_arrived' if len(ticks) == limit else 'unknown'
    first = float((ticks[hits[0]] + 1) * dt) if arrived else None
    return dict(goal_arrived=None if outcome == 'unknown' else arrived,
        arrival_outcome=outcome, goal_radius_m=float(radius_m),
        first_arrival_s=first,
        minimum_upright_goal_distance_m=float(distance.min()) if distance.size else None,
        contact_after_arrival=bool(arrived and contact.any()),
        fault_after_arrival=bool(arrived and fault.any()),
        contract='Original XY target disk; enter before contact/fault; no heading, speed or hold requirement.')


def goal_running_cost_metrics(physical, *, goal_xy, radius_m, dt, horizon_s,
                              running_tick_cost, cost_valid):
    """Separate factual running cost through goal entry from later cost.

    ``running_tick_cost`` is an already scaled/integrated cost per lower tick,
    either a vector or a matrix of named-by-caller components. It must exclude
    terminal replacements. No reward formula, alpha, terminal penalty or
    ranking is changed here. Missing task observations make the corresponding
    cost subtotal unknown, while a measured goal entry stays a measured entry.
    The sample reaching the disk is included in the pre-arrival interval.
    This inherits contact/fault-only arrival, not native-domain-stop arrival.
    ``full_observed`` need not cover the horizon; it and post-arrival cost may
    include valid logged samples at/after contact. ``before_fatal`` does not.
    """
    arrival = goal_arrival_metrics(physical, goal_xy=goal_xy, radius_m=radius_m,
                                   dt=dt, horizon_s=horizon_s)
    active = np.asarray(physical['active_tick'], bool)
    tick = np.asarray(physical['physical_tick'])
    valid = np.asarray(cost_valid, bool)
    costs = np.asarray(running_tick_cost, float)
    if valid.shape != active.shape or costs.ndim not in (1, 2) or costs.shape[0] != active.size:
        raise ValueError('running costs and validity must align with physical samples')
    if costs.ndim == 2 and costs.shape[1] == 0:
        raise ValueError('at least one running cost component required')
    eligible = active & valid
    if not np.isfinite(costs[eligible]).all() or np.any(costs[eligible] < 0):
        raise ValueError('valid running costs must be finite nonnegative integrated costs')

    # Use the measured hit and physical tick identity, never rounded arrival
    # seconds, reference positions, path progress or a final-hold predicate.
    xy = np.asarray(physical['actual_xy'], float)
    fatal = active & (np.asarray(physical['contact_failure'], bool) |
                      np.asarray(physical['nonfinite'], bool) |
                      np.asarray(physical['lower_fault'], bool) |
                      np.asarray(physical['policy_fault'], bool) |
                      ~np.isfinite(xy).all(axis=1))
    fatal_rows = np.flatnonzero(fatal)
    before_fatal = active & (np.arange(active.size) <
                             (fatal_rows[0] if fatal_rows.size else active.size))
    hits = np.flatnonzero(before_fatal & (np.linalg.norm(xy - goal_xy, axis=1) <= radius_m))
    hit = int(hits[0]) if hits.size else None
    prefix = before_fatal if hit is None else active & (np.arange(active.size) <= hit)
    after = np.zeros_like(active) if hit is None else active & (np.arange(active.size) > hit)

    def subtotal(mask):
        if not np.all(valid[mask]):
            return None
        value = costs[mask].sum(axis=0, dtype=np.float64)
        return float(value) if costs.ndim == 1 else value.tolist()

    return dict(arrival,
        goal_entry_physical_tick=int(tick[hit]) if hit is not None else None,
        pre_arrival_running_cost=subtotal(prefix) if hit is not None else None,
        post_arrival_observed_running_cost=subtotal(after) if hit is not None else None,
        pre_arrival_ticks=int(prefix.sum()) if hit is not None else None,
        observed_before_fatal_running_cost=subtotal(before_fatal),
        observed_before_fatal_ticks=int(before_fatal.sum()),
        full_observed_running_cost=subtotal(active),
        full_observed_ticks=int(active.sum()),
        cost_contract='Per-lower-tick integrated running costs; entry tick included; '
                      'no terminal replacement, no post-goal erasure, no new policy objective.')


def task_metrics(physical,*,speed,goal_s,config):
    dt=float(config['timing']['lower_dt_s']);horizon=float(config['training_future_phase_C']['episode_s'])
    band=config['stage_B_baseline'];limit=round(horizon/dt);hold=round(band['hold_s']/dt)
    if not math.isclose(limit*dt,horizon,abs_tol=1e-8) or not 0<hold<=limit:
        raise ValueError('task observation window and final hold must match logged clock')
    active=np.asarray(physical['active_tick'],bool);valid=active & np.asarray(physical['task_reward_valid'],bool)
    clock=np.asarray(physical['physical_tick']);seen=clock[active]
    if (seen.size and (seen[0]!=0 or np.any(np.diff(seen)!=1) or seen[-1]>=limit)):
        raise ValueError('missing, duplicated or out-of-window physical clock')
    # A task prefix cannot become task-valid again after irreversible handover.
    if np.any(np.diff(np.asarray(valid[active],np.int8))>0):raise ValueError('task observations resumed after invalid tail')
    fault=active & (np.asarray(physical['nonfinite'],bool)|physical['lower_fault']|physical['policy_fault'])
    domain=active & physical['domain_exit'];contact=active & physical['contact_failure']
    irreversible=np.flatnonzero(domain|contact);faults=np.flatnonzero(fault)
    # An earlier genuine domain exit already settles task failure even if a
    # later held physical tail has an engineering-unknown outcome.
    known_failure=bool(irreversible.size and (not faults.size or irreversible[0]<faults[0]))
    complete=bool(seen.size==limit and valid.sum()==limit and not fault.any())
    finite_fields=['actual_forward_speed','path_cross_track','path_heading_error','path_progress','goal_section_signed_distance','peak_roll']
    if not all(np.isfinite(np.asarray(physical[k])[valid]).all() for k in finite_fields):
        raise ValueError('nonfinite task observations must remain unknown, not scored')
    speed_error=np.asarray(physical['actual_forward_speed'],float)-speed
    ey=np.asarray(physical['path_cross_track'],float);eh=np.asarray(physical['path_heading_error'],float)
    tail=valid & (clock>=limit-hold);tail_complete=bool(tail.sum()==hold and not fault.any())
    within=bool(tail_complete and np.all(abs(ey[tail])<=band['final_y_error_m'])
        and np.all(abs(eh[tail])<=band['final_path_heading_error_rad'])
        and np.all(abs(speed_error[tail])<=band['final_speed_error_m_s']))
    goal=bool(np.any(valid & (physical['goal_section_signed_distance']>=0) & (physical['path_progress']>=goal_s)))
    recovered=False if known_failure else bool(goal and within) if complete else None
    peak=np.asarray(physical['peak_roll'],float)
    working=bool(np.any(peak[valid]>band['working_roll_rad'])) if valid.any() else None
    rms=lambda x:float(np.sqrt(np.mean(x[valid]**2))) if valid.any() else None
    tail_max=lambda x:float(np.max(np.abs(x[tail]))) if tail.any() else None
    return dict(task_metric_status='observed_failure' if known_failure else 'complete' if complete else 'unknown',
        task_observed_ticks=int(valid.sum()),task_observed_s=float(valid.sum()*dt),complete_task_observation=complete,
        goal_crossed_while_task_valid=goal,final_hold=bool(complete and within),
        task_recovered=recovered,working_failure=working,
        physical_working_failure_observed=bool(np.any(peak[active]>band['working_roll_rad'])) if active.any() else None,
        performance_success=None if recovered is None else bool(recovered and not working),
        tail_observed_ticks=int(tail.sum()),tail_path_max_m=tail_max(ey),tail_heading_error_max_rad=tail_max(eh),
        tail_speed_error_max_m_s=tail_max(speed_error),
        task_speed_rmse_m_s=rms(speed_error),task_path_rmse_m=rms(ey),task_heading_rmse_rad=rms(eh),
        task_speed_abs_error_integral_m=float(np.abs(speed_error[valid]).sum()*dt) if valid.any() else None,
        task_path_abs_error_time_integral_m_s=float(np.abs(ey[valid]).sum()*dt) if valid.any() else None,
        contract='Original configured goal/progress, final-hold bands; working-roll separate; no task data during held tail.')


def zero_relative_cases(rows):
    """Pair each actual force case with its own controller/noise/alpha zero case."""
    key=lambda r:tuple(r[k] for k in ('panel','role','route','speed','alpha','replicate'))
    zeros={};seen=set()
    for r in rows:
        if r['case_id'] in seen:raise ValueError('duplicate actual case identity')
        seen.add(r['case_id'])
        if not r.get('force_N',0.):
            if key(r) in zeros:raise ValueError('duplicate own zero-force baseline')
            zeros[key(r)]=r
    changes=[]
    for r in rows:
        if not r.get('force_N',0.):continue
        if key(r) not in zeros:raise ValueError('missing own zero-force baseline')
        z=zeros[key(r)]
        physical_change=(int(r['outcome']=='survived')-int(z['outcome']=='survived')
            if r['outcome'] in ('contact','survived') and z['outcome'] in ('contact','survived') else None)
        task_change=(int(r['task_recovered'])-int(z['task_recovered'])
            if r['task_recovered'] is not None and z['task_recovered'] is not None else None)
        changes.append(dict(case_id=r['case_id'],zero_case_id=z['case_id'],panel=r['panel'],role=r['role'],
            alpha=r['alpha'],physical_change_vs_zero=physical_change,task_change_vs_zero=task_change))
    return changes
