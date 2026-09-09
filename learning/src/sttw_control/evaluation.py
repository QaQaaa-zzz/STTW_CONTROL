"""Complete engineering traces and explicitly scoped recovery summaries."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jp
from .path import tracking_metrics
from .events import profile


def evaluate(env,path,*,seed=0,policy=None,policy_identity=None):
    path=Path(path)
    if policy is not None and not policy_identity:
        raise ValueError('residual evaluation requires a frozen policy identity')
    path.mkdir(parents=True,exist_ok=False)
    config=asdict(env.config)
    identity={'config':config,'model':env.bundle.identity,'seed':seed,'backend':env.backend,
              'controller':'baseline' if policy is None else 'residual','policy':policy_identity,
              'reset_semantics':'synthetic_forward_velocity_model_initial_pose'}
    declaration=json.dumps(identity,sort_keys=True,allow_nan=False)
    (path/'declaration.json').write_text(declaration+'\n')
    (path/'status.json').write_text('{"status":"running"}\n')
    begin=time.monotonic()
    frames=[]
    try:
        reset=jax.jit(env.reset) if env.backend=='mjx' else env.reset
        step=jax.jit(env.step) if env.backend=='mjx' else env.step
        state=reset(jax.random.PRNGKey(seed))
        event=np.asarray(state.event).tolist()
        (path/'event.json').write_text(json.dumps({'start_seconds':event[0]*env.config.controller.dt,'end_seconds':event[1]*env.config.controller.dt,'steer_rate_peak':event[2],'force_peak':event[3],'waveform':'half_sine' if event[4] else 'constant'},indent=2)+'\n')
        first_position=np.asarray(state.data.qpos[:3]).copy()
        # Fixed world frame anchored at the initial position; orientation and
        # swept-body envelopes are not yet planning-ready space descriptors.
        balance_time=task_time=None
        transitions=0
        total_reward=0.
        action=np.zeros(2)
        def capture(s,a):
            return {'qpos':np.asarray(s.data.qpos).copy(),'qvel':np.asarray(s.data.qvel).copy(),
                    'event':np.asarray(s.event).copy(),'injected_steer_rate':float(s.event[2]*profile(jp.maximum(s.tick-1,0),s.event)) if int(s.tick)>0 else 0.,'applied_wrench':np.asarray(s.data.xfrc_applied[env.bundle.chassis]).copy(),
                    'time':float(s.data.time),'observation':np.asarray(s.obs).copy(),
                    'measurement':np.asarray(s.measurement).copy(),
                    'pose':np.asarray(s.pose).copy(),'reference_roll':float(s.reference),
                    'motion_command':np.asarray(env.command(s.tick,s.pose)),
                    'command':np.asarray(s.actuator.previous).copy(),'base':np.asarray(s.base).copy(),
                    'action':np.asarray(a).copy(),'reward':float(s.reward),
                    'terminated':bool(s.terminated),'truncated':bool(s.truncated),'end_code':int(s.end_code)}
        frames.append(capture(state,action))
        for _ in range(env.horizon):
            action=np.zeros(2) if policy is None else policy(state.obs)
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
                 'recovery_eligible':(event[2]!=0 or event[3]!=0) and int(state.tick)>=event[1],
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
        summary['command_limits']=command_limit_metrics(arrays['command'],env.config.actuator)
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
