"""V3 environment-owned raw-command schedules.

Rows are ``[start_time_s, target_speed_m_s, target_steer_rad]``. The fixed
16-row shape uses ``[99, 0, 0]`` for unused rows. The Actor must receive only
the current published command and its measured rate, never these rows.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


ROWS = 16
PAD_TIME = 99.0


def _empty_rows(prepared_speed):
    rows = jnp.zeros((ROWS, 3), dtype=jnp.float32)
    rows = rows.at[:, 0].set(PAD_TIME)
    return rows.at[0].set(jnp.array([0.0, prepared_speed, 0.0]))


def _repeating_rows(key, prepared_speed, spec, family):
    commands = spec["commands"]
    branch = commands["nominal"] if family == 0 else commands["random"]
    first_key, speed_key, steer_key, straight_key, hold_key, tail_key = jax.random.split(key, 6)
    first = jax.random.uniform(first_key, (), minval=branch["first_switch_s"][0],
                               maxval=branch["first_switch_s"][1])
    speed = jax.random.uniform(speed_key, (14,), minval=commands["speed_min_m_s"],
                               maxval=commands["speed_max_m_s"])
    steer = jax.random.uniform(steer_key, (14,), minval=-branch.get("steer_abs_max_rad", commands["steer_abs_max_rad"]),
                               maxval=branch.get("steer_abs_max_rad", commands["steer_abs_max_rad"]))
    if family == 0:
        straight = jax.random.uniform(straight_key, (14,)) < branch["straight_probability"]
        steer = jnp.where(straight, 0.0, steer)
    hold = jax.random.uniform(hold_key, (14,), minval=branch["hold_s"][0],
                              maxval=branch["hold_s"][1])
    tail_speed = (jnp.asarray(branch["tail_speed_m_s"], dtype=jnp.float32)
                  if family == 0 else jax.random.uniform(
                      tail_key, (), minval=commands["speed_min_m_s"],
                      maxval=commands["speed_max_m_s"]))

    def add(i, state):
        rows, time, count = state

        def place(state):
            rows, time, count = state
            rows = rows.at[count].set(jnp.stack((time, speed[i], steer[i])))
            return rows, time + hold[i], count + 1

        return jax.lax.cond(time < branch["last_random_target_s"], place,
                            lambda state: state, state)

    rows, _, count = jax.lax.fori_loop(
        0, 14, add, (_empty_rows(prepared_speed), first, jnp.int32(1)))
    return rows.at[count].set(jnp.stack((jnp.asarray(branch["last_random_target_s"]),
                                         tail_speed, jnp.asarray(branch["tail_steer_rad"]))))


def _conflict_rows(key, prepared_speed, spec, slew=None):
    if spec.get('preference_v5') and spec.get('training_stage')==1:
        c=spec['preference_v5']['training']['stage1'];keys=jax.random.split(key,4)
        draw=lambda k,r:jax.random.uniform(k,(),minval=r[0],maxval=r[1])
        high=draw(keys[0],c['conflict_speed_target_range_m_s'])
        start=draw(keys[1],c['turn_start_range_s'])
        delta=draw(keys[2],c['steer_magnitude_range_rad'])*jnp.where(jax.random.uniform(keys[3],())<c['turn_sign_probability_positive'],1.,-1.)
        return _empty_rows(prepared_speed).at[0].set(jnp.array([0.,high,0.])).at[1].set(jnp.array([start,high,delta]))
    if spec.get("smooth_v4"):
        return _smooth_conflict(key,prepared_speed,spec,slew)
    c = spec["commands"]["conflict"]
    speed_key, start_key, mag_key, sign_key, form_key, single_key, first_key, second_key, tail_key = jax.random.split(key, 9)
    high = jax.random.uniform(speed_key, (), minval=c["speed_range_m_s"][0],
                              maxval=c["speed_range_m_s"][1])
    start = jax.random.uniform(start_key, (), minval=c["turn_start_s"][0],
                               maxval=c["turn_start_s"][1])
    magnitude = jax.random.uniform(mag_key, (), minval=c["steer_magnitude_rad"][0],
                                   maxval=c["steer_magnitude_rad"][1])
    steer = jnp.where(jax.random.bernoulli(sign_key), magnitude, -magnitude)
    single = jax.random.uniform(form_key, ()) < c["single_turn_fraction"]
    single_hold = jax.random.uniform(single_key, (), minval=c["single_hold_s"][0],
                                     maxval=c["single_hold_s"][1])
    first_hold = jax.random.uniform(first_key, (), minval=c["reversal_first_hold_s"][0],
                                    maxval=c["reversal_first_hold_s"][1])
    second_hold = jax.random.uniform(second_key, (), minval=c["reversal_second_hold_s"][0],
                                     maxval=c["reversal_second_hold_s"][1])
    tail_speed = jax.random.uniform(tail_key, (), minval=c["tail_speed_range_m_s"][0],
                                    maxval=c["tail_speed_range_m_s"][1])
    rows = _empty_rows(prepared_speed).at[0].set(jnp.stack((jnp.array(0.0), high, jnp.array(0.0))))
    rows = rows.at[1].set(jnp.stack((start, high, steer)))

    def single_rows(rows):
        return rows.at[2].set(jnp.stack((start + single_hold, tail_speed,
                                         jnp.asarray(c["tail_steer_rad"]))))

    def reversal_rows(rows):
        rows = rows.at[2].set(jnp.stack((start + first_hold, high, -steer)))
        return rows.at[3].set(jnp.stack((start + first_hold + second_hold, tail_speed,
                                         jnp.asarray(c["tail_steer_rad"]))))

    return jax.lax.cond(single, single_rows, reversal_rows, rows)


def schedule(env_id, episode_index, prepared_speed, spec, key=None):
    """Return ``(rows, family, slew)`` for one 16-second episode.

    Families are 0 nominal, 1 conflict, 2 continuous random. Without an
    explicit key, the key is seed 66001 folded first by environment and then
    by episode; thus reset order and other environments cannot shift a case.
    An explicit key is treated as the already selected episode key.
    """
    if key is None:
        key = jax.random.fold_in(jax.random.PRNGKey(spec["commands"]["random_seed"]), env_id)
        key = jax.random.fold_in(key, episode_index)
    family_key, speed_slew_key, steer_slew_key, nominal_key, conflict_key, random_key = jax.random.split(key, 6)
    c = spec["commands"]
    u = jax.random.uniform(family_key, ())
    heading_fraction = c.get("heading_recovery_start_fraction", 0.)
    family = jnp.where(u < c["nominal_fraction"], 0,
                       jnp.where(u < c["nominal_fraction"] + c["conflict_fraction"], 1,
                                 jnp.where(u < c["nominal_fraction"] + c["conflict_fraction"] + c["random_fraction"], 2, 3))).astype(jnp.int32)
    if not spec.get('priority_recovery_v51'):
        family = jnp.minimum(family, 2)
    elif not np.isclose(c["nominal_fraction"] + c["conflict_fraction"] + c["random_fraction"] + heading_fraction, 1.):
        raise ValueError('V5.1 training-family probabilities must sum to one')
    slew = jnp.stack((
        jax.random.uniform(speed_slew_key, (), minval=c["episode_speed_slew_range_m_s2"][0],
                           maxval=c["episode_speed_slew_range_m_s2"][1]),
        jax.random.uniform(steer_slew_key, (), minval=c["episode_steer_slew_range_rad_s"][0],
                           maxval=c["episode_steer_slew_range_rad_s"][1]),
    ))
    rows = jax.lax.switch(family, (
        lambda: _repeating_rows(nominal_key, prepared_speed, spec, 0),
        lambda: _conflict_rows(conflict_key, prepared_speed, spec, slew),
        lambda: _repeating_rows(random_key, prepared_speed, spec, 2),
        lambda: _empty_rows(prepared_speed),
    ))
    return rows, family, slew


def heading_recovery_initial_error(env_id, episode_index, spec, family):
    """Deterministic virtual-reference yaw offset for V5.1 family 3 only."""
    if not spec.get('priority_recovery_v51'):
        return jnp.asarray(0., dtype=jnp.float32)
    cfg = spec['commands']['heading_recovery_start']
    key = jax.random.fold_in(jax.random.PRNGKey(spec['commands']['random_seed']), env_id)
    key = jax.random.fold_in(jax.random.fold_in(key, episode_index), 51001)
    sign_key, band_key, magnitude_key = jax.random.split(key, 3)
    small = jax.random.uniform(band_key, ()) < cfg['small_magnitude_probability']
    lo = jnp.where(small, cfg['small_magnitude_rad'][0], cfg['large_magnitude_rad'][0])
    hi = jnp.where(small, cfg['small_magnitude_rad'][1], cfg['large_magnitude_rad'][1])
    magnitude = jax.random.uniform(magnitude_key, (), minval=lo, maxval=hi)
    sign = jnp.where(jax.random.uniform(sign_key, ()) < cfg['sign_positive_probability'], 1., -1.)
    return jnp.where(jnp.asarray(family) == 3, sign*magnitude, 0.)


def publish_command(raw, rows, tick, slew, dt):
    """Publish the target effective at ``tick*dt`` and slew one control tick.

    ``raw`` is the previously issued command. At a target boundary the new
    target applies to this tick; returned rates are the actual issued change
    divided by dt, including the final partial slew step.
    """
    raw = jnp.asarray(raw)
    rows = jnp.asarray(rows)
    slew = jnp.asarray(slew)
    time = jnp.asarray(tick) * dt
    active = (rows[:, 0] <= time) & (rows[:, 0] < PAD_TIME)
    index = jnp.max(jnp.where(active, jnp.arange(ROWS), 0))
    target = rows[index, 1:]
    change = jnp.clip(target - raw, -slew * dt, slew * dt)
    issued = raw + change
    return issued, change / dt, target


def review_rows(spec):
    """Freeze the two 3-method review schedules, independent of training RNG.

    Returns ``(main_rows, random_rows, fixed_slew)``. The random sequence uses
    NumPy Generator(PCG64(88001)) in documented first-switch, then per-segment
    speed, steer, duration order. Serialize the returned rows before review.
    """
    e = spec["evaluation"]
    c = spec["commands"]
    main = np.zeros((ROWS, 3), dtype=np.float64)
    main[:, 0] = PAD_TIME
    source = np.asarray(e["main_target_rows"], dtype=np.float64)
    main[:len(source)] = source
    random_rows = np.zeros((ROWS, 3), dtype=np.float64)
    random_rows[:, 0] = PAD_TIME
    random_rows[0] = (0.0, e["initial_speed_m_s"], 0.0)
    rng = np.random.Generator(np.random.PCG64(e["post_training_random_seed"]))
    branch = c["random"]
    time = rng.uniform(*branch["first_switch_s"])
    index = 1
    while time < branch["last_random_target_s"]:
        speed = rng.uniform(c["speed_min_m_s"], c["speed_max_m_s"])
        steer = rng.uniform(-c["steer_abs_max_rad"], c["steer_abs_max_rad"])
        duration = rng.uniform(*branch["hold_s"])
        if index >= ROWS - 1:
            raise ValueError("review command schedule exceeds fixed row capacity")
        random_rows[index] = (time, speed, steer)
        index += 1
        time += duration
    random_rows[index] = (branch["last_random_target_s"], e["initial_speed_m_s"],
                          branch["tail_steer_rad"])
    return main, random_rows, np.asarray(
        (e["main_speed_slew_m_s2"], e["main_steer_slew_rad_s"]), dtype=np.float64)


def _smooth_conflict(key, prepared_speed, spec, slew):
    c=spec['smooth_v4']['training_commands']; keys=jax.random.split(key,9)
    draw=lambda k,r:jax.random.uniform(k,(),minval=r[0],maxval=r[1])
    high=draw(keys[0],c['conflict_speed_range_m_s']); start=draw(keys[1],c['conflict_start_s'])
    delta=draw(keys[2],c['conflict_steer_magnitude_rad'])*jnp.where(jax.random.uniform(keys[3],())<c['conflict_sign_probability_positive'],1.,-1.)
    single=jax.random.uniform(keys[4],())<c['conflict_single_fraction']
    hold1=draw(keys[5],c['conflict_hold_after_limited_target_reached_s']); hold2=draw(keys[6],c['conflict_hold_after_limited_target_reached_s'])
    tail=jnp.where(jax.random.uniform(keys[7],())<c['conflict_tail_same_high_speed_probability'],high,draw(keys[8],c['conflict_other_tail_speed_range_m_s']))
    # Retain inherited high-speed target from t=0. Account for its remaining slew at turn onset.
    speed_at_turn=prepared_speed+jnp.clip(high-prepared_speed,-slew[0]*start,slew[0]*start)
    end1=start+jnp.maximum(jnp.abs(delta)/slew[1],jnp.abs(high-speed_at_turn)/slew[0])+hold1
    end2=end1+2*jnp.abs(delta)/slew[1]+hold2
    rows=_empty_rows(prepared_speed).at[0].set(jnp.array([0.,high,0.])).at[1].set(jnp.array([start,high,delta]))
    return jax.lax.cond(single,lambda r:r.at[2].set(jnp.array([end1,tail,0.])),lambda r:r.at[2].set(jnp.array([end1,high,-delta])).at[3].set(jnp.array([end2,tail,0.])),rows)


def publish_issued_reference(raw,rows,tick,slew,dt,prepublished=False):
    """Exact replay of a dataset reference; no further screening or slew."""
    if not prepublished:return publish_command(raw,rows,tick,slew,dt)
    row=rows[jnp.minimum(tick,rows.shape[0]-1)]
    return row[1:3],row[3:5],row[1:3]
