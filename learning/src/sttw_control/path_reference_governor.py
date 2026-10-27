"""Finite-horizon reference-governor comparator for the current path stack.

This is an adaptation of predictive reference governance, not a novel safety
method. Preview requires a complete simulator carry; it is not deployable state
estimation. There is no future-event input. Constant neural proposals are held
inside the unchanged follower/feedback/relay/filter/lower loop during preview.
"""
from dataclasses import dataclass
import math
import numpy as np

FRACTIONS=(0.,.25,.5,.75,1.)


def native_cost_candidates(previous, config):
    """Fixed ordinary grid, with no learned middle-policy dependency.

    Index zero retains the previous target. The following 25 targets span a
    fixed normalized5x5 grid; actual filtering and asymmetric permissions remain
    the native environment's responsibility. A held target is not a held final
    speed/steering reference. This grid is a finite comparator, not optimal MPC.
    """
    previous = np.asarray(previous, dtype=np.float32)
    if previous.shape != (2,) or not np.isfinite(previous).all():
        raise ValueError('finite previous two-channel latent required')
    levels = np.array([-.95, -.475, 0., .475, .95], dtype=np.float32)
    pairs = np.array([(a, b) for a in levels for b in levels], dtype=np.float32)
    latents = np.concatenate([previous[None], np.arctanh(pairs)], axis=0)
    c = config['residual_interface']
    scales = np.array([c['speed_negative_scale_m_s'], c['speed_positive_scale_m_s'], c['steer_scale_rad']])
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('positive finite native reference scales required')
    normalized = np.tanh(latents)
    offsets = normalized * np.column_stack([
        np.where(normalized[:, 0] >= 0, scales[1], scales[0]),
        np.full(len(latents), scales[2])])
    return dict(latents=latents, physical_offsets=offsets, previous_index=0, zero_index=13)


def choose_native_cost(summary):
    """Minimize original cost, retaining native failure penalties and exact ties.

    An unknown forecast is an engineering error. Known failed predictions are
    not called safe: they retain the native failure cost and remain visible.
    First-index ties retain the previous candidate; no safety backup is claimed.
    """
    costs = np.asarray(summary['original_user_cost'], dtype=float)
    finite = np.asarray(summary['finite'], dtype=bool)
    if costs.ndim != 1 or costs.shape != finite.shape or not len(costs):
        raise ValueError('matching nonempty candidate scores required')
    if not finite.all() or not np.isfinite(costs).all():
        raise ValueError('unknown native forecast; no optimistic candidate deletion')
    return int(np.argmin(costs))


def make_native_cost_preview(env, *, upper_steps=50, horizon_s=10., collect_trace=False):
    """Use an original PathCommandEnv without modern feedback or middle NN.

    The complete carry and original user alpha are forwarded unchanged at the
    start. No future force argument is exposed. Original native physical/domain
    termination, full remaining-task failure penalties and four-tick rewards
    are retained. Only the finite forecast tail is truncated, with no invented
    terminal value or equilibrium certificate.
    """
    import jax
    import jax.numpy as j
    if type(upper_steps) is not int or upper_steps <= 0:
        raise ValueError('positive integral prediction steps required')
    if not math.isfinite(horizon_s) or horizon_s <= 0 or abs(horizon_s/.02-round(horizon_s/.02)) > 1e-6:
        raise ValueError('upper-aligned positive native horizon required')
    if env.config['timing']['lower_dt_s'] != .005 or env.config['timing']['upper_dt_s'] != .02:
        raise ValueError('native four-tick timing required')
    limit = round(horizon_s/.005)

    def idle(state):
        log = jax.tree.map(lambda x: j.repeat(x[None], 4, axis=0), env.zero_log)
        log['scored_tick_reward'] = j.zeros(4)
        return state, log

    def preview(origin, latent, path, speed):
        def step(value, _):
            current, ticks, cost, fault = value
            _, _, obs_fault = env.observation(current, path, speed, horizon_s)
            candidate = current.replace(fault=current.fault | obs_fault)
            end, log = jax.lax.cond(current.tick < limit,
                lambda s: env.policy_step(s, latent, path, speed, horizon_s, j.zeros(5)),
                idle, candidate)
            active = log['active_tick'].astype(bool)
            reward = j.where(active, log['scored_tick_reward'], 0.)
            fault = fault | end.fault | j.any(active & (log['policy_fault'] | log['lower_fault'])) | ~j.all(j.isfinite(reward))
            trace = {k: log[k] for k in ('active_tick', 'actual_xy', 'governed', 'nominal',
                     'actual_forward_speed', 'phi', 'peak_roll', 'physical_failure',
                     'domain_exit', 'policy_fault', 'lower_fault', 'scored_tick_reward')} if collect_trace else None
            return (end, ticks+j.sum(active, dtype=j.int32), cost-j.sum(reward), fault), trace
        initial = (origin, j.int32(0), j.float32(0), j.bool_(False))
        (end, ticks, cost, fault), trace = jax.lax.scan(step, initial, None, length=upper_steps)
        finite = ~fault & j.isfinite(cost)
        summary = dict(original_user_cost=cost, finite=finite, observed_lower_ticks=ticks,
            allocated_lower_slots=j.int32(upper_steps*4),
            requested_lower_ticks=j.minimum(upper_steps*4, j.maximum(0, limit-origin.tick)),
            native_physical_failure=end.physical.failed, native_domain_exit=end.domain_exit,
            end_task_tick=end.tick)
        return (end, summary, trace) if collect_trace else (end, summary)
    return preview


@dataclass(frozen=True)
class CandidateSet:
    fractions: np.ndarray
    latents: np.ndarray
    physical_offsets: np.ndarray


def candidates(previous,proposed,config):
    """Interpolate physical offsets, preserving both exact latent endpoints."""
    from .path_command_policy import map_latent
    from .reference_prefix import inverse_offset_target
    import jax.numpy as j
    previous=np.asarray(previous);proposed=np.asarray(proposed)
    if previous.shape!=(2,) or proposed.shape!=(2,) or not np.isfinite([previous,proposed]).all():
        raise ValueError('two finite previous/proposed neural reference latents required')
    scales=config['residual_interface']
    if not all(math.isfinite(scales[k]) and scales[k]>0 for k in
               ['speed_positive_scale_m_s','speed_negative_scale_m_s','steer_scale_rad']):
        raise ValueError('original asymmetric physical scales required')
    old=np.asarray(map_latent(j.asarray(previous),config));new=np.asarray(map_latent(j.asarray(proposed),config))
    offsets=old[None]+np.asarray(FRACTIONS)[:,None]*(new-old)[None]
    latent=np.stack([np.asarray(inverse_offset_target(j.asarray(x),config)[0]) for x in offsets])
    latent[0]=previous;latent[-1]=proposed
    actual=np.stack([np.asarray(map_latent(j.asarray(x),config)) for x in latent])
    return CandidateSet(np.asarray(FRACTIONS),latent,actual)


@dataclass(frozen=True)
class Decision:
    index: int | None
    status: str
    predicted_admissible: bool
    fallback_used: bool


def choose(predictions):
    """Largest tested fraction, without monotonicity or safe-backup assumptions.

    Any unknown forecast is an engineering error, not a negative control label.
    If every known candidate contacts, use the previous proposal explicitly as
    an uncertified operational fallback. Never call this physical impossibility.
    """
    if len(predictions)!=len(FRACTIONS):raise ValueError('all five actual candidates required')
    statuses=[r['status'] for r in predictions]
    if any(s not in ('survived','contact','unknown') for s in statuses):raise ValueError('explicit forecast statuses required')
    if 'unknown' in statuses:return Decision(None,'prediction_error',False,False)
    valid=[i for i,s in enumerate(statuses) if s=='survived']
    return Decision(max(valid),'tested_candidate',True,False) if valid else Decision(0,'no_predicted_admissible_candidate',False,True)


class HeldProposalActor:
    def apply(self,parameters,observation):
        # Parameters are the current candidate, never an event or future actor.
        del observation
        if parameters.shape!=(2,):raise ValueError('two held neural reference latents required')
        return parameters


def make_preview_rollout(advance,*,upper_steps=50,task_lower_limit=2000,collect_trace=False):
    """Roll current complete carry forward without resetting its clock/history.

    `advance` has the existing make_goal_step interface. No pulse can be supplied
    to the returned function: the predictor assumes zero future external force,
    retaining the present physical/ESO/lower/history/filter/gate state. Actual
    execution must use a separate call with its real external event.
    """
    import jax
    import jax.numpy as j
    if type(upper_steps) is not int or upper_steps<=0 or type(task_lower_limit) is not int or task_lower_limit<=0 or task_lower_limit%4:
        raise ValueError('positive upper steps and upper-aligned task horizon required')
    def preview(carry,latent,path,speed):
        parameters=dict(actor=dict(native=latent,keep_channels=j.ones(2,bool)),feedback_gain_rad=j.float32(.1))
        requested=j.minimum(4*upper_steps,j.maximum(0,task_lower_limit-carry.state.tick))
        initial=(carry,j.int32(0),j.bool_(False),j.bool_(False),j.float32(0))
        def step(value,_):
            current,ticks,contact,fault,cost=value
            end,log=advance(current,parameters,path,speed,j.zeros(5),j.int32(1))
            active=log['active_tick']
            contact=contact|j.any(active&log['contact_failure'])
            fault=fault|end.state.fault|j.any(active&(log['nonfinite']|log['lower_fault']|log['policy_fault']))
            reward=j.where(active,log['scored_tick_reward'],0.)
            fault=fault|~j.all(j.isfinite(reward))
            trace=({k:log[k] for k in ['active_tick','actual_xy','actual_forward_speed','wheel_speed_proxy','phi','phi_dot','actual_delta','reference','scored_tick_reward','actor_action_valid']} if collect_trace else None)
            return (end,ticks+j.sum(active,dtype=j.int32),contact,fault,cost-j.sum(reward)),trace
        (end,ticks,contact,fault,cost),trace=jax.lax.scan(step,initial,None,length=upper_steps)
        unexplained_failure=end.state.physical.failed&~contact
        finite=~fault&~unexplained_failure&j.isfinite(cost)
        admissible=finite&~contact&(ticks==requested)&(requested>0)
        summary=dict(observed_lower_ticks=ticks,requested_lower_ticks=requested,allocated_lower_slots=j.int32(4*upper_steps),
            contact=contact,finite=finite,admissible=admissible,original_user_cost=cost)
        return (end,summary,trace) if collect_trace else (end,summary)
    return preview


def make_path_preview(env,*,feedback_spec,upper_steps=50,collect_trace=False):
    """Reuse the real current task kernel, including its existing causal relay."""
    from .goal_preview_baseline import make_goal_step,GoalPreviewEnv
    from .contact_only_environment import ContactOnlyPhysics
    if not isinstance(env,GoalPreviewEnv) or not isinstance(env.physics,ContactOnlyPhysics):
        raise ValueError('current goal-preview/contact-only environment required')
    if env.config['timing']['lower_dt_s']!=.005 or env.config['timing']['upper_dt_s']!=.02:
        raise ValueError('current physical/task timing required')
    advance=make_goal_step(env,HeldProposalActor(),horizon_s=10.,spec=feedback_spec)
    return make_preview_rollout(advance,upper_steps=upper_steps,task_lower_limit=2000,collect_trace=collect_trace)


def forecast_rows(summary):
    """Host classification keeps incomplete/faulted forecasts unknown."""
    n=len(FRACTIONS)
    required=['observed_lower_ticks','requested_lower_ticks','allocated_lower_slots','contact','finite','admissible','original_user_cost']
    arrays={k:np.asarray(summary[k]) for k in required}
    if any(a.shape!=(n,) for a in arrays.values()):raise ValueError('five unpadded forecast lanes required')
    if any(arrays[k].dtype!=np.dtype(bool) for k in ['contact','finite','admissible']):
        raise ValueError('forecast status arrays must be actual booleans')
    rows=[]
    for i in range(n):
        observed,requested,allocated=(float(arrays[k][i]) for k in required[:3])
        count_ok=(all(math.isfinite(v) and v.is_integer() for v in [observed,requested,allocated]) and 0<requested<=allocated and 0<=observed<=requested)
        finite=bool(arrays['finite'][i]) and np.isfinite(arrays['original_user_cost'][i]) and count_ok
        status=('unknown' if not finite else 'contact' if arrays['contact'][i] else
                'survived' if observed==requested else 'unknown')
        if bool(arrays['admissible'][i])!=(status=='survived'):raise ValueError('forecast admissibility/count contract differs')
        rows.append(dict(fraction=FRACTIONS[i],status=status,observed_lower_ticks=observed,requested_lower_ticks=requested,
                         allocated_lower_slots=allocated,original_user_cost=float(arrays['original_user_cost'][i]) if finite else None))
    return rows
