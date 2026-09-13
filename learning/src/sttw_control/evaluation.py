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
from .actuator import residual_target
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
              'reset_semantics':'synthetic_forward_velocity_model_initial_pose'}
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
        (path/'event.json').write_text(json.dumps({'start_seconds':event[0]*env.config.controller.dt,'end_seconds':event[1]*env.config.controller.dt,'steer_rate_peak':event[2],'rear_load_torque_nm':event[5],'force_peak':event[3],'waveform':'half_sine' if event[4] else 'constant'},indent=2)+'\n')
        first_position=np.asarray(state.data.qpos[:3]).copy()
        # Fixed world frame anchored at the initial position; orientation and
        # swept-body envelopes are not yet planning-ready space descriptors.
        balance_time=task_time=None
        transitions=0
        total_reward=0.
        action=np.zeros(2)
        request=jp.zeros(2)
        request_time=0.
        def prepare(s,a):
            base,mapped=env.prepare_action(s,a)
            return residual_target(base,mapped,env.config.actuator)
        prepare=jax.jit(prepare)
        recorded_command=jax.jit(env.command)
        def capture(s,a):
            return {'priority_alpha':float(s.priority_alpha),'attitude_risk':float(s.history.frames[-1,-1]) if env.config.priority is not None and env.config.observation.include_attitude_risk else 0.,'attitude_risk_observed':env.config.priority is not None and env.config.observation.include_attitude_risk,'prelimit_command':np.asarray(request).copy(),'request_time':request_time,
                    'actuator_diagnostic_valid':int(s.tick)>0,
                    'actuator_force':np.asarray(s.data.actuator_force).copy(),
                    'generalized_actuator_force':np.asarray(s.data.qfrc_actuator).copy(),
                    'actuator_velocity':np.asarray(s.data.actuator_velocity).copy(),
                    'actuator_ctrl':np.asarray(s.data.ctrl).copy(),
                    'qpos':np.asarray(s.data.qpos).copy(),'qvel':np.asarray(s.data.qvel).copy(),
                    'event':np.asarray(s.event).copy(),'injected_steer_rate':float(s.event[2]*profile(jp.maximum(s.tick-1,0),s.event)) if int(s.tick)>0 else 0.,'applied_generalized_force':np.asarray(s.data.qfrc_applied).copy(),'applied_wrench':np.asarray(s.data.xfrc_applied[env.bundle.chassis]).copy(),
                    'time':float(s.data.time),'observation':np.asarray(s.obs).copy(),
                    'measurement':np.asarray(s.measurement).copy(),
                    'pose':np.asarray(s.pose).copy(),'reference_roll':float(s.reference),
                    'motion_command':np.asarray(recorded_command(s.tick,s.pose)),
                    'command':np.asarray(s.actuator.previous).copy(),'base':np.asarray(s.base).copy(),
                    'action':np.asarray(a).copy(),'reward':float(s.reward),
                    'terminated':bool(s.terminated),'truncated':bool(s.truncated),'end_code':int(s.end_code)}
        frames.append(capture(state,action))
        for _ in range(env.horizon):
            action=np.zeros(2) if policy is None else policy(state.obs)
            request=prepare(state.replace(data=None),jp.asarray(action))
            request_time=float(state.data.time)
            state=step(state,jp.asarray(action))
            transitions+=1
            frames.append(capture(state,action))
            total_reward+=float(state.reward)
            elapsed=float(state.data.time)-event[0]*env.config.controller.dt
            if bool(state.balance_recovered) and balance_time is None: balance_time=elapsed
            if bool(state.task_recovered) and task_time is None: task_time=elapsed
            if bool(state.done): break
        arrays={key:np.asarray([frame[key] for frame in frames]) for key in frames[0]}
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
        summary['command_headroom']=headroom_metrics(arrays['base'][:-1],arrays['measurement'][:-1,2],env.config.actuator)
        summary['sampled_mechanical_work']=mechanical_work(arrays)
        summary['command_limits']=command_limit_metrics(arrays['command'],env.config.actuator)
        if env.config.bend is not None:
            from .path import bend_trace_features
            features=bend_trace_features(arrays['pose'],env.config.bend)
            summary['path_tracking']={'right_error_rmse_m':float(np.sqrt(np.mean(features[:,0]**2))),'right_error_peak_m':float(np.max(abs(features[:,0]))),'coordinate':'bend_right_normal'}
        if env.config.figure_eight is not None:
            features=eight_trace_features(arrays['pose'],env.config.figure_eight)
            summary['path_tracking']={'right_error_rmse_m':float(np.sqrt(np.mean(features[:,0]**2))),'right_error_peak_m':float(np.max(abs(features[:,0]))),'coordinate':'figure_eight_right_normal'}
        if env.config.circle is not None:
            summary['circle_tracking']=tracking_metrics(arrays['qpos'][:,:2],env.config.circle)
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
