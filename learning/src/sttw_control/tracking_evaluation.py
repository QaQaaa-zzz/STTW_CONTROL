"""Paired CPU audit for the alpha-conditioned geometric tracking task.

Writes actual per-step rewards/components, original-path XY, true/estimated
speed and failure endpoints. Never fills an early-failed trace to the horizon.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from .tracking_reward import reward_components, tolerances


def evaluate_tracking(task_path, training_path, output, checkpoint=None, *,
                      seeds=(47011,), alphas=(0.,.25,.5,.75,1.)):
    from .env import RecoveryEnv, load_config
    import jax.numpy as jp
    from .network import make_policy_identity, load_policy
    c=load_config(task_path)
    if c.tracking_reward is None:
        raise ValueError('use this audit only with a tracking_reward geometric task')
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('declare unique evaluation seeds')
    if not alphas or len(set(alphas))!=len(alphas) or any(not math.isfinite(a) or not 0<=a<=1 for a in alphas):
        raise ValueError('declare unique alphas in [0,1]')
    training=json.loads(Path(training_path).read_text())
    cases=training.get('validation_events')
    if not cases:raise ValueError('explicit fixed cases are required')
    dt=c.controller.dt
    for case in cases:
        if case['start']<0 or case['duration']<=0 or case['start']+case['duration']+c.tracking_reward.return_deadline_seconds>c.horizon_seconds:
            raise ValueError('invalid event/return observation window')
        if any(not math.isclose(case[k]/dt,round(case[k]/dt),abs_tol=1e-7) for k in ('start','duration')):
            raise ValueError('events must align to control ticks')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    env=RecoveryEnv(c,backend='cpu')
    identity=make_policy_identity(env.bundle.identity,asdict(c),c.observation.history_steps)
    policy=load_policy(checkpoint,expected=identity) if checkpoint else None
    declaration=dict(task=asdict(c),identity=identity,checkpoint=str(checkpoint) if checkpoint else None,
        cases=cases,seeds=list(seeds),alphas=list(alphas),role='declared paired evaluation; no training',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        termination='real endpoint; no padding',return_deadline='since observed geometric departure, not hidden event end')
    (output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    methods=('baseline','residual') if policy is not None else ('baseline',)
    summaries=[]; links=[]
    for i,case in enumerate(cases):
        event=np.array([round(case['start']/dt),round((case['start']+case['duration'])/dt),
            case.get('steer_rate',0.),case.get('force',0.),float(case.get('waveform','constant')=='half_sine'),case.get('rear_torque',0.)])
        for seed in seeds:
            for alpha in alphas:
                directory=output/f'case_{i:02d}'/f'alpha_{alpha:g}_seed_{seed}'
                directory.mkdir(parents=True)
                traces={}
                for method in methods:
                    state=env.set_priority(env.reset(int(seed)),alpha).replace(event=jp.asarray(event))
                    rows=[];error_max=0.
                    for _ in range(env.horizon):
                        if bool(state.done):break
                        action=np.zeros(2,dtype=np.float32) if method=='baseline' else np.asarray(policy(state.obs))
                        new=env.step(state,action)
                        path=np.asarray(env.path_features(new.pose))
                        speed=float(np.dot(new.data.qvel[:3],new.data.xmat[env.bundle.chassis].reshape(3,3)[:,0]))
                        rebuilt=reward_components(float(new.measurement[0]),float(new.measurement[1]),speed-c.speed_reference,
                            path[0],path[1],action,np.asarray(state.tracking.previous_action),float(state.priority_alpha),new.tracking,
                            c.tracking_reward,dt,c.alive_reward_rate,c.failure_penalty,bool(new.terminated))
                        err=abs(float(sum(rebuilt.values()))-float(new.reward));error_max=max(error_max,err)
                        if not math.isfinite(err) or err>3e-5*(1+abs(float(new.reward))):
                            raise RuntimeError('geometric reward reconstruction mismatch')
                        row=dict(time=float(new.tick)*dt,reward=float(new.reward),speed=speed,
                            speed_estimate=float(new.measurement[5])*.1,target_speed=c.speed_reference,
                            lateral=float(path[0]),heading=float(path[1]),roll=float(new.measurement[0]),
                            roll_rate=float(new.measurement[1]),pose=np.asarray(new.pose),
                            qpos=np.array(new.data.qpos),qvel=np.array(new.data.qvel),action=action,
                            base=np.asarray(state.base),command=np.asarray(new.actuator.previous),
                            failed=bool(new.terminated),end_code=int(new.end_code),
                            return_pending=bool(new.tracking.pending),return_age=float(new.tracking.age_ticks)*dt,
                            deadline_missed=bool(new.tracking.deadline_missed),alpha=float(state.priority_alpha))
                        row.update({f'reward_{k}':float(v) for k,v in new.reward_parts.items()})
                        rows.append(row);state=new
                    if not rows:raise RuntimeError('empty evaluation trace')
                    trace={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
                    np.savez_compressed(directory/f'{method}.npz',**trace,event=event)
                    rc=c.tracking_reward;hold=max(1,int(math.ceil(rc.hold_seconds/dt-1e-9)))
                    final=(len(rows)>=hold and int(state.tick)>=env.horizon and not bool(state.terminated)
                        and np.all(np.abs(trace['lateral'][-hold:])<=rc.return_lateral_tolerance)
                        and np.all(np.abs(trace['heading'][-hold:])<=rc.return_heading_tolerance)
                        and np.all(np.abs(trace['speed'][-hold:]-c.speed_reference)<=rc.return_speed_tolerance)
                        and np.all(np.abs(trace['roll'][-hold:])<=rc.roll_working_limit)
                        and np.all(np.abs(trace['roll_rate'][-hold:])<=rc.return_roll_rate_tolerance))
                    bv,by=tolerances(alpha,rc)
                    summary=dict(case=i,seed=int(seed),alpha=float(alpha),method=method,steps=len(rows),
                        endpoint_seconds=float(trace['time'][-1]),failed=bool(state.terminated),
                        episode_return=float(trace['reward'].sum()),lateral_rmse=float(np.sqrt(np.mean(trace['lateral']**2))),
                        heading_rmse=float(np.sqrt(np.mean(trace['heading']**2))),
                        speed_rmse=float(np.sqrt(np.mean((trace['speed']-c.speed_reference)**2))),
                        final_tracking_hold=bool(final),ever_departed=bool(state.tracking.ever_departed),
                        task_recovered=bool(final and state.tracking.ever_departed and not state.tracking.deadline_missed),
                        deadline_missed=bool(state.tracking.deadline_missed),
                        speed_budget_fraction=float(np.mean(np.abs(trace['speed']-c.speed_reference)>bv)),
                        path_budget_fraction=float(np.mean(np.abs(trace['lateral'])>by)),
                        reward_reconstruction_max_abs=error_max,
                        reward_component_sums={k:float(v.sum()) for k,v in trace.items() if k.startswith('reward_')})
                    summaries.append(summary);traces[method]=trace
                _plots(env,traces,directory,case)
                links.append(str(directory.relative_to(output)))
    # Invalid simulation evidence is retained as null in JSON rather than invented zeros.
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,float) and not math.isfinite(x):return None
        return x
    (output/'summary.json').write_text(json.dumps(clean(summaries),indent=2,allow_nan=False)+'\n')
    (output/'INDEX.md').write_text('# Conditional geometric tracking audit\n\n'
        'Each directory contains actual NPZ traces and PNG/PDF plots. Failed traces stop at their real endpoint. '
        'Survival and final hold do not establish full-task success or generalization.\n\n'
        +'\n'.join(f'- [{p}]({p}/)' for p in links)+'\n')
    return summaries


def _plots(env,traces,directory,case):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('png','pdf'):fig.savefig(directory/f'{name}.{suffix}',dpi=140)
        plt.close(fig)
    fig,ax=plt.subplots()
    if env.config.bend is not None:ref=np.asarray(env.bend_table)[:,1:3]
    elif env.config.circle is not None:
        from .path import circle_reference
        ref=circle_reference(env.config.circle)
    else:
        from .path import eight_reference
        ref=eight_reference(env.config.figure_eight)
    ax.plot(ref[:,0],ref[:,1],linestyle='--',label='original geometric path')
    for label,t in traces.items():ax.plot(t['pose'][:,0],t['pose'][:,1],label=label)
    ax.set(xlabel='x [m]',ylabel='y [m]');ax.axis('equal');ax.legend();save(fig,'trajectory')
    for key,unit in [('reward','reward / step'),('speed','m/s'),('lateral','m'),('heading','rad'),('roll','rad')]:
        fig,ax=plt.subplots()
        for label,t in traces.items():
            ax.plot(t['time'],t[key],label=label)
            if key=='speed':ax.plot(t['time'],t['speed_estimate'],linestyle=':',label=label+' wheel-speed estimate')
        if key=='speed':ax.axhline(env.config.speed_reference,linestyle='--',label='request')
        if key=='roll':
            ax.axhline(env.config.tracking_reward.roll_working_limit,linestyle='--')
            ax.axhline(-env.config.tracking_reward.roll_working_limit,linestyle='--')
        ax.axvline(case['start'],linestyle=':');ax.axvline(case['start']+case['duration'],linestyle=':')
        if key=='reward':ax.set_yscale('symlog',linthresh=.02)
        ax.set(xlabel='time [s]',ylabel=f'{key} [{unit}]');ax.legend();save(fig,key)
    for label,t in traces.items():
        fig,ax=plt.subplots()
        for k in t:
            if k.startswith('reward_'):ax.plot(t['time'],t[k],label=k.removeprefix('reward_'))
        ax.set(xlabel='time [s]',ylabel='signed reward / step');ax.set_yscale('symlog',linthresh=.005)
        ax.legend(fontsize=6,ncol=2);save(fig,label+'_reward_components')
