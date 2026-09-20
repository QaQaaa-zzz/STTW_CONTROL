"""Complete engineering traces and explicitly scoped recovery summaries."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jp
from .path import tracking_metrics,eight_trace_features
from .events import profile
from .actuator import residual_target,composition_base
from .energy import mechanical_work


def evaluate(env,path,*,seed=0,policy=None,policy_identity=None,priority_alpha=None):
    if priority_alpha is not None and (env.config.priority is None or not np.isfinite(priority_alpha) or not 0<=priority_alpha<=1):raise ValueError("invalid priority intervention")
    path=Path(path)
    if policy is not None and not policy_identity:
        raise ValueError('residual evaluation requires a frozen policy identity')
    path.mkdir(parents=True,exist_ok=False)
    config=asdict(env.config)
    identity={'config':config,'model':env.bundle.identity,'seed':seed,'backend':env.backend,
              'controller':'baseline' if policy is None else 'residual','policy':policy_identity,
              'reset_semantics':('closed_loop_straight_preparation_preserving_controller_actuator_history_then_task_clock_zero'
                                 if env.config.preparation_seconds else 'synthetic_forward_velocity_model_initial_pose')}
    if priority_alpha is not None:identity["priority_override"]=float(priority_alpha)
    import mujoco
    identity['actuator_diagnostics']={'names':[mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(env.model.nu)],
        'force_limited':np.asarray(env.model.actuator_forcelimited).tolist(),'force_range':np.asarray(env.model.actuator_forcerange).tolist(),
        'sampling':'backend data at captured state; CPU after existing mj_forward; no substep peak coverage',
        'request_alignment':'row i request at time[i-1], final command and response at time[i]; reset row invalid',
        'force_units':'raw actuator scalar force; generalized_actuator_force includes transmission'}
    declaration=json.dumps(identity,sort_keys=True,allow_nan=False)
    (path/'declaration.json').write_text(declaration+'\n')
    (path/'status.json').write_text('{"status":"running"}\n')
    begin=time.monotonic()
    frames=[]
    try:
        reset=jax.jit(env.reset) if env.backend=='mjx' else env.reset
        step=jax.jit(env.step) if env.backend=='mjx' else env.step
        state=reset(jax.random.PRNGKey(seed))
        if priority_alpha is not None:state=env.set_priority(state,priority_alpha)
        event=np.asarray(state.event).tolist()
        (path/'event.json').write_text(json.dumps({'start_seconds':event[0]*env.config.controller.dt,'end_seconds':event[1]*env.config.controller.dt,'steer_rate_peak':event[2],'rear_disturbance_mode':env.config.rear_disturbance_mode,'rear_load_torque_nm':event[5] if env.config.rear_disturbance_mode=='torque' else 0.,'rear_command_bias_rad_s':-event[5] if env.config.rear_disturbance_mode=='command' else 0.,'force_peak':event[3],'waveform':'half_sine' if event[4] else 'constant'},indent=2)+'\n')
        if env.config.motion_commands is not None:
            (path/'commands.json').write_text(json.dumps({'columns':['start_seconds','speed_m_s','yaw_rate_rad_s','scheduled_alpha'],'schedule':np.asarray(state.command_schedule).tolist(),'reward_alignment':'transition i uses raw command and alpha at row i-1; yaw_rate is mean world heading rate over the control interval'},indent=2)+'\n')
        if env.config.timed_reference is not None:
            columns=(['start_seconds','speed_m_s','yaw_rate_rad_s','speed_slew_m_s2','yaw_slew_rad_s2','recovery_entry','scenario']
                     if env.config.timed_reference.training_mix else ['start_seconds','speed_m_s','yaw_rate_rad_s'])
            (path/'commands.json').write_text(json.dumps({'columns':columns,'schedule':np.asarray(state.command_schedule).tolist(),'reference':('fixed reset-integrated geometric curve; projection follows position, speed retains external clock' if env.config.timed_reference.mode=='geometry' else 'causal committed geometric prefix; pre-step command drives independent integration; no future projection or rebasing' if env.config.tracking.geometric else 'independent timed SE2 integration; pre-step command drives transition; no rebasing')},indent=2)+'\n')
        first_position=np.asarray(state.data.qpos[:3]).copy()
        # Fixed world frame anchored at the initial position; orientation and
        # swept-body envelopes are not yet planning-ready space descriptors.
        balance_time=task_time=None
        transitions=0
        total_reward=0.
        action=np.zeros(2)
        effective_action=jp.zeros(2)
        composed_base=jp.zeros(2)
        request=jp.zeros(2)
        request_time=0.
        def prepare(s,a):
            base,mapped=env.prepare_action(s,a)
            return residual_target(base,mapped,env.config.actuator),mapped,composition_base(base,env.config.actuator)
        prepare=jax.jit(prepare)
        recorded_command=jax.jit(env.command)
        def capture(s,a):
            row={'priority_alpha':float(s.priority_alpha),'attitude_risk':float(s.history.frames[-1,-1]) if env.config.priority is not None and env.config.observation.include_attitude_risk else 0.,'attitude_risk_observed':env.config.priority is not None and env.config.observation.include_attitude_risk,'prelimit_command':np.asarray(request).copy(),'request_time':request_time,
                    'actuator_diagnostic_valid':int(s.tick)>0,
                    'actuator_force':np.asarray(s.data.actuator_force).copy(),
                    'generalized_actuator_force':np.asarray(s.data.qfrc_actuator).copy(),
                    'actuator_velocity':np.asarray(s.data.actuator_velocity).copy(),
                    'actuator_ctrl':np.asarray(s.data.ctrl).copy(),
                    'qpos':np.asarray(s.data.qpos).copy(),'qvel':np.asarray(s.data.qvel).copy(),
                    'event':np.asarray(s.event).copy(),'injected_steer_rate':float(s.event[2]*profile(jp.maximum(s.tick-1,0),s.event)) if int(s.tick)>0 else 0.,'injected_rear_rate':float(-s.event[5]*profile(jp.maximum(s.tick-1,0),s.event)) if int(s.tick)>0 and env.config.rear_disturbance_mode=='command' else 0.,'applied_generalized_force':np.asarray(s.data.qfrc_applied).copy(),'applied_wrench':np.asarray(s.data.xfrc_applied[env.bundle.chassis]).copy(),
                    'time':float(s.tick*env.config.controller.dt),'simulation_time':float(s.data.time),'observation':np.asarray(s.obs).copy(),
                    'measurement':np.asarray(s.measurement).copy(),
                    'path_progress':float(s.path_progress),'path_id':int(s.path_id),'pose':np.asarray(s.pose).copy(),'reference_roll':float(s.reference),
                    'motion_command':np.asarray(recorded_command(s.tick,s.pose,s.command_schedule,path_id=s.path_id,path_progress=s.path_progress) if env.config.reference_paths is not None else recorded_command(s.tick,s.pose,s.command_schedule)),
                    'user_command':np.asarray(env.requested(s.tick,s.command_schedule)[:2]) if env.config.motion_commands is not None else np.asarray([0.,0.]),'yaw_rate_world':float(s.yaw_rate),
                    'command':np.asarray(s.actuator.previous).copy(),'base':np.asarray(s.base).copy(),
                    'active_base_output_scale':float(s.active_base_output_scale),
                    'effective_action':np.asarray(effective_action).copy(),'composition_base':np.asarray(composed_base).copy(),
                    'action':np.asarray(a).copy(),'reward':float(s.reward),
                    'terminated':bool(s.terminated),'truncated':bool(s.truncated),'end_code':int(s.end_code)}
            if env.config.tracking is not None:
                from .tracking_reward import return_observation
                row['path_features']=np.asarray(env.path_features(s.pose,s.path_id,s.path_progress))
                row['true_forward_speed']=float(jp.dot(s.data.qvel[:3],jp.asarray(s.data.xmat[env.bundle.chassis]).reshape(3,3)[:,0]))
                row['return_state']=np.asarray(return_observation(s.tracking_state,env.config.tracking))
                row.update({'reward_'+name:float(value) for name,value in s.tracking_components.items()})
                if s.tracking_raw_costs is not None:
                    row.update({'raw_cost_'+name:float(value) for name,value in s.tracking_raw_costs.items()})
            if env.config.timed_reference is not None:
                from .timed_reference import errors
                feature=np.asarray(errors(s.pose,s.reference_pose,s.reference_command))
                previous=frames[-1]['reference_command'] if frames else np.asarray(s.reference_command)
                row.update(reference_pose=np.asarray(s.reference_pose).copy(),reference_command=np.asarray(s.reference_command).copy(),
                    raw_reference_request=np.asarray(s.raw_reference_request).copy(),
                    path_features=np.asarray(s.geometric_features) if env.config.tracking.geometric else feature[:3],longitudinal_error=float(feature[3]),
                    yaw_rate_error=float(s.yaw_rate-previous[1]),user_command=np.asarray(s.reference_command).copy(),
                    motion_command=np.asarray(env.control_reference(s.pose,s.reference_pose,s.reference_command,s.geometric_features)))
                if env.config.tracking.geometric:
                    row.update(path_segment=int(s.path_segment),timed_path_features=feature[:3],geometric_frontier_clamped=bool(float(s.path_progress)>=float(s.geometric_table[int(s.tick),0])-1e-6))
            return row
        frames.append(capture(state,action))
        for _ in range(env.horizon):
            action=np.zeros(2) if policy is None else policy(state.obs)
            request,effective_action,composed_base=prepare(state.replace(data=None),jp.asarray(action))
            request_time=float(state.tick*env.config.controller.dt)
            state=step(state,jp.asarray(action))
            transitions+=1
            frames.append(capture(state,action))
            total_reward+=float(state.reward)
            elapsed=float(state.data.time)-event[0]*env.config.controller.dt
            if bool(state.balance_recovered) and balance_time is None: balance_time=elapsed
            if bool(state.task_recovered) and task_time is None: task_time=elapsed
            if bool(state.done): break
        arrays={key:np.asarray([frame[key] for frame in frames]) for key in frames[0]}
        if env.config.timed_reference is not None:
            arrays['command_schedule']=np.asarray(state.command_schedule)
        if state.reference_geometry is not None:
            arrays['reference_geometry']=np.asarray(state.reference_geometry)
            arrays['geometric_schedule']=np.asarray(state.command_schedule)
        np.savez_compressed(path/'trace.npz',**arrays)
        position=arrays['qpos'][:,:3]-first_position
        summary={'controller':identity['controller'],'backend':env.backend,'seed':seed,
                 'transitions':transitions,'captured_states':len(frames),'episode_return':total_reward,
                 'end_code':int(state.end_code),'physical_failure':bool(state.terminated),
                 'recovery_eligible':(event[2]!=0 or event[3]!=0 or event[5]!=0) and int(state.tick)>=event[1],
                 'balance_recovery_success':balance_time is not None and not bool(state.terminated),
                 'task_recovery_success':task_time is not None and not bool(state.terminated),
                 'first_balance_hold_completion_from_event_seconds':balance_time,
                 'first_task_hold_completion_from_event_seconds':task_time,
                 'roll_abs_max_rad':float(np.abs(arrays['measurement'][:,0]).max()),
                 'whole_episode_root_left_world_y_m':float(max(0.,position[:,1].max())),
                 'whole_episode_root_right_world_y_m':float(max(0.,-position[:,1].min())),
                 'whole_episode_root_forward_world_x_m':float(position[:,0].max()),
                 'wall_seconds':time.monotonic()-begin,'declaration_sha256':hashlib.sha256(declaration.encode()).hexdigest(),
                 'scope':'engineering_baseline_not_recovery_domain_or_swept_body_envelope'}
        summary['preparation']={'seconds':env.config.preparation_seconds,
            'failed':bool(state.preparation_failed),'simulation_time_at_task_end':float(state.data.time)}
        summary['command_headroom']=headroom_metrics(arrays['base'][:-1],arrays['measurement'][:-1,2],env.config.actuator)
        summary['sampled_mechanical_work']=mechanical_work(arrays)
        summary['command_limits']=command_limit_metrics(arrays['command'],env.config.actuator)
        if env.config.bend is not None and env.config.reference_paths is None:
            from .path import bend_trace_features
            features=bend_trace_features(arrays['pose'],env.config.bend)
            summary['path_tracking']={'right_error_rmse_m':float(np.sqrt(np.mean(features[:,0]**2))),'right_error_peak_m':float(np.max(abs(features[:,0]))),'coordinate':'bend_right_normal'}
        if env.config.figure_eight is not None:
            features=eight_trace_features(arrays['pose'],env.config.figure_eight)
            summary['path_tracking']={'right_error_rmse_m':float(np.sqrt(np.mean(features[:,0]**2))),'right_error_peak_m':float(np.max(abs(features[:,0]))),'coordinate':'figure_eight_right_normal'}
        if env.config.circle is not None:
            summary['circle_tracking']=tracking_metrics(arrays['qpos'][:,:2],env.config.circle)
        if env.config.tracking is not None:
            from .tracking_diagnostics import audit_trace, write_diagnostics
            audited=audit_trace(path)
            summary.update(audited['summary'])
            write_diagnostics(path, audited=audited, plots=False)
        (path/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        (path/'status.json').write_text('{"status":"complete"}\n')
        return summary
    except Exception as exc:
        (path/'status.json').write_text(json.dumps({'status':'error','error':str(exc)})+'\n')
        raise


def command_limit_metrics(commands,config):
    """Observed final rate-command limit occupancy; excludes initial reset frame.

    This measures rate limits, not torque saturation, joint stops or clipping of
    pre-limit requests. It must not be reported as total actuator saturation.
    """
    commands=np.asarray(commands)[1:]
    if len(commands)==0:
        return {'samples':0,'steer_rate_limit_fraction':None,'rear_rate_limit_fraction':None,'either_rate_limit_fraction':None}
    limits=np.array([config.steer_rate_limit,config.rear_rate_limit])
    hit=np.abs(commands)>=limits*(1-1e-6)
    return {'samples':len(commands),'steer_rate_limit_fraction':float(hit[:,0].mean()),
            'rear_rate_limit_fraction':float(hit[:,1].mean()),
            'either_rate_limit_fraction':float(hit.any(axis=1).mean()),
            'scope':'final_rate_command_limit_occupancy_not_torque_saturation'}


def headroom_metrics(base,steer,config):
    """Decision-state instantaneous command margins; no delayed/torque prediction.

    Inputs are pre-action baseline commands and measured steering joint angles.
    Restricted means some part of the configured residual box is unavailable,
    not that the actual policy requested that part or suffered a performance loss.
    """
    base=np.asarray(base);steer=np.asarray(steer)
    if base.ndim!=2 or base.shape[1]!=2 or steer.shape!=(len(base),) or not len(base):
        raise ValueError('headroom requires aligned nonempty baseline and steering states')
    if not np.isfinite(base).all() or not np.isfinite(steer).all():
        raise ValueError('nonfinite headroom inputs')
    limits=np.array([config.steer_rate_limit,config.rear_rate_limit])
    lower=np.broadcast_to(-limits,base.shape).copy();upper=np.broadcast_to(limits,base.shape).copy()
    lower[:,0]=np.clip((-config.steer_limit-steer)/config.dt,-limits[0],limits[0])
    upper[:,0]=np.clip((config.steer_limit-steer)/config.dt,-limits[0],limits[0])
    positive=upper-base;negative=base-lower
    scale=config.strength*np.array([config.steer_residual_scale,config.rear_residual_scale])
    restricted=np.any((positive<scale-1e-6)|(negative<scale-1e-6),axis=1)
    return {'samples':len(base),'residual_box_restricted_fraction':float(restricted.mean()),
            'baseline_outside_command_bounds_fraction':float(np.any((positive<0)|(negative<0),axis=1).mean()),
            'minimum_positive_steer_margin_rad_s':float(positive[:,0].min()),
            'minimum_negative_steer_margin_rad_s':float(negative[:,0].min()),
            'minimum_positive_rear_margin_rad_s':float(positive[:,1].min()),
            'minimum_negative_rear_margin_rad_s':float(negative[:,1].min()),
            'scope':'instantaneous_command_headroom_not_torque_or_delayed_authority'}


def candidate_offset(action, offset, step, count):
    """Finite, symmetric bias window; never expands normalized action limits."""
    if count < 1:raise ValueError('candidate window must contain at least one step')
    a=np.asarray(action,float);b=np.asarray(offset,float)
    if a.shape!=(2,) or b.shape!=(2,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('invalid candidate action/offset')
    envelope=np.sin(np.pi*(step+.5)/count)**2 if 0<=step<count else 0.
    raw=a+envelope*b
    return np.clip(raw,-1.,1.),bool(np.any(np.abs(raw)>1.))


def candidate_search_report(rows):
    """Finite-set Pareto and reward ordering, never a global feasibility claim."""
    def front(group):
        return [r['candidate'] for r in group if not any(
            q['speed_rmse']<=r['speed_rmse'] and q['path_rmse']<=r['path_rmse']
            and (q['speed_rmse']<r['speed_rmse'] or q['path_rmse']<r['path_rmse']) for q in group)]
    finite=[r for r in rows if r['finite']]
    task=[r for r in finite if r['task_qualified']]
    work=[r for r in task if r['work_envelope_qualified']]
    selected={str(a):(max(work,key=lambda r:r['scores'][str(a)])['candidate'] if work else None) for a in (0.,.5,1.)}
    return {'scope':'declared local residual-bias candidates, not global reachability or deployable expert proof',
        'count':len(rows),'actual_search_transitions':sum(r['steps'] for r in rows),
        'task_qualified_count':len(task),'work_envelope_qualified_count':len(work),
        'task_pareto_candidates':front(task),'work_pareto_candidates':front(work),
        'work_reward_winners':selected,'rows':rows,
        'decision':('No candidate satisfied the declared work gates; this finite search does not prove global infeasibility.'
                    if not work else 'Inspect non-dominated candidates and raw work metrics; do not force different winners.')}


def search_candidates(env,path,*,seed,policy,policy_identity,alpha,start_seconds=3.,duration=.5,
                      offsets=(-.5,-.25,0.,.25,.5)):
    """Real CPU rollouts from one complete immutable closed-loop prefix state.

    This is offline controllability exploration. Biases are causal timed pulses;
    choosing them after seeing whole rollouts is NOT a deployable policy. All
    candidates, including failures, are retained. No optimizer or model edits.
    """
    import math,itertools
    from .tracking_reward import transition
    from .timed_reference import recovery_entry_at
    c=env.config;dt=c.controller.dt;path=Path(path)
    if env.backend!='cpu' or c.tracking.objective!='soft_budget_v1' or not c.tracking.geometric or c.timed_reference is None:
        raise ValueError('candidate search requires CPU V1 committed geometric task')
    if alpha not in (0.,.5,1.) or not policy_identity:raise ValueError('declare exact alpha and policy identity')
    if (not math.isfinite(start_seconds+duration) or start_seconds<0 or duration<=0
            or start_seconds+duration>c.horizon_seconds):raise ValueError('invalid search interval')
    if any(not math.isclose(x/dt,round(x/dt),abs_tol=1e-7) for x in (start_seconds,duration)):
        raise ValueError('search interval must align to control steps')
    if not offsets or len(set(offsets))!=len(offsets) or any(not math.isfinite(x) or abs(x)>1 for x in offsets):
        raise ValueError('invalid declared normalized offsets')
    path.mkdir(parents=True,exist_ok=False)
    prepared=env.set_priority(env.reset(seed),alpha)
    start_tick=round(start_seconds/dt);window=round(duration/dt)
    for _ in range(start_tick):
        if bool(prepared.done):raise RuntimeError('shared zero-residual prefix failed before search')
        prepared=env.step(prepared,np.zeros(2))
    # Entire EnvState, not only qpos/qvel: includes ESO, actuator, history,
    # committed path, projection progress, task clock and return-debt state.
    original_qpos=np.asarray(prepared.data.qpos).copy()
    metadata={'config':asdict(c),'seed':seed,'alpha':alpha,'policy':policy_identity,
        'start_seconds':start_seconds,'duration_seconds':duration,'offsets':list(offsets),
        'max_search_transitions':len(offsets)**2*(env.horizon-start_tick),
        'prefix_transitions':start_tick,'preparation_transitions':round(c.preparation_seconds/dt),
        'prefix':'shared full-state zero-residual prefix, immutable; physical preparation counted separately',
        'work_gates':'roll <= configured working .3 and overspeed <= configured .05 throughout search, plus final task requirements; diagnostic gates, no changed physical termination'}
    (path/'declaration.json').write_text(json.dumps(metadata,indent=2)+'\n')
    summaries=[]
    for idx,bias in enumerate(itertools.product(offsets,repeat=2)):
        state=prepared;replays=[jax.tree.map(lambda x:np.asarray(x).copy(),prepared.tracking_state) for _ in range(3)]
        scores=np.zeros(3);rows=[];clip_count=0;max_error=0.
        for k in range(env.horizon-start_tick):
            if bool(state.done):break
            action,clipped=candidate_offset(np.asarray(policy(state.obs)),bias,k,window);clip_count+=int(clipped)
            nxt=env.step(state,action)
            speed=float(jp.dot(nxt.data.qvel[:3],nxt.data.xmat[env.bundle.chassis].reshape(3,3)[:,0]))
            errors=np.asarray(nxt.geometric_features);ev=speed-float(state.reference_command[0])
            scenario=int(state.command_schedule[0,6]) if c.timed_reference.training_mix else -1
            trigger=bool(recovery_entry_at(state.tick,state.command_schedule,dt)) if c.timed_reference.training_mix else False
            # The same physical trace may be rescored; it is NOT off-policy PPO data.
            for h,a in enumerate((0.,.5,1.)):
                replays[h],parts=transition(replays[h],roll=float(nxt.measurement[0]),roll_rate=float(nxt.measurement[1]),
                    speed_error=ev,lateral_error=errors[0],heading_error=errors[1],action=action,alpha=a,
                    dt=dt,alive_rate=c.alive_reward_rate,failure_penalty=c.failure_penalty,failed=bool(nxt.terminated),
                    enabled=int(state.tick)*dt>=c.tracking.start_seconds,config=c.tracking,
                    longitudinal_error=0.,yaw_rate_error=0.,recovery_trigger=trigger,clock_from_departure=scenario!=1,xp=np)
                scores[h]+=float(sum(parts.values()))
                if a==alpha:max_error=max(max_error,abs(float(sum(parts.values()))-float(nxt.reward)))
            rows.append({'time':float(nxt.tick)*dt,'speed_error':ev,'speed':speed,'lateral_error':float(errors[0]),
                         'heading_error':float(errors[1]),'roll':float(nxt.measurement[0]),'roll_rate':float(nxt.measurement[1]),
                         'action':action.copy(),'base':np.asarray(state.base),'command':np.asarray(nxt.actuator.previous),
                         'pose':np.asarray(nxt.pose),'reward':float(nxt.reward),'failed':bool(nxt.terminated)})
            state=nxt
        if not rows:raise RuntimeError('empty candidate trajectory')
        if max_error>3e-5:raise RuntimeError('candidate reward reconstruction mismatch')
        trace={key:np.asarray([row[key] for row in rows]) for key in rows[0]}
        name=f'alpha_{alpha:g}_candidate_{idx:02d}'
        np.savez_compressed(path/(name+'.npz'),**trace,offset=np.asarray(bias),scores=scores)
        finite=all(np.isfinite(trace[k]).all() for k in ('speed','lateral_error','roll','heading_error'))
        task=bool(finite and not state.terminated and int(state.tick)>=env.horizon and not state.tracking_state.pending
                  and not state.tracking_state.deadline_missed and float(state.tracking_state.hold)+1e-6>=c.tracking.hold_seconds)
        roll_peak=float(np.max(np.abs(trace['roll']))) if finite else None
        over_peak=float(np.maximum(trace['speed_error'],0).max()) if finite else None
        summary={'candidate':name,'alpha':alpha,'offset':list(bias),'steps':len(rows),'finite':finite,
            'physical_failure':bool(state.terminated),'task_qualified':task,
            'work_envelope_qualified':bool(task and roll_peak<=c.tracking.roll_working_limit and over_peak<=c.tracking.overspeed_band),
            'roll_peak':roll_peak,'overspeed_peak':over_peak,
            'speed_rmse':float(np.sqrt(np.mean(trace['speed_error']**2))) if finite else None,
            'path_rmse':float(np.sqrt(np.mean(trace['lateral_error']**2))) if finite else None,
            'minimum_speed':float(trace['speed'].min()) if finite else None,
            'underspeed_integral':float(np.maximum(-trace['speed_error'],0).sum()*dt) if finite else None,
            'path_absolute_integral':float(np.abs(trace['lateral_error']).sum()*dt) if finite else None,
            'scores':{str(a):float(v) for a,v in zip((0.,.5,1.),scores)},
            'action_clipped_fraction':clip_count/len(rows),'reward_reconstruction_max_abs':max_error}
        summaries.append(summary)
        (path/'results.json').write_text(json.dumps(candidate_search_report(summaries),indent=2,allow_nan=False)+'\n')
        print(json.dumps(summary),flush=True)
        np.testing.assert_array_equal(prepared.data.qpos,original_qpos)
    return summaries
