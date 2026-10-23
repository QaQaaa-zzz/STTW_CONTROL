"""Full-deadline signed force scan of authentic TRAIN prefix states.

The prefix actor and suffix actor may differ: this measures conditional suffix
recoverability, not single-actor reachability. No optimizer or new reset physics.
"""
from pathlib import Path
import json
import numpy as np


def scan_contract(declaration,manifest,ids,forces):
    """Resolve original deadline and state-level roles before any simulation."""
    horizon=float(manifest['config']['training_future_phase_C']['episode_s'])
    if (not np.isfinite(horizon) or horizon<=0 or
        abs(horizon/.02-round(horizon/.02))>1e-6 or
        manifest.get('original_task_horizon_s',horizon)!=horizon):
        raise ValueError('bank/config deadline mismatch')
    if declaration['maximum_lower_ticks']!=len(ids)*len(forces)*round(horizon/.005):
        raise ValueError('inconsistent declared scan budget')
    roles=declaration.get('roles_by_bank_index')
    if roles is None:resolved=['validation' if i%4==3 else 'calibration' for i in range(len(ids))]
    else:
        if set(roles)!={str(int(i)) for i in ids} or any(r not in ('calibration','validation') for r in roles.values()):
            raise ValueError('roles must cover exactly the declared bank states')
        resolved=[roles[str(int(i))] for i in ids]
    return dict(horizon_s=horizon,maximum_policy_intervals=round(horizon/.02),roles=resolved)


def outcome_labels(data,*,speed,goal_s,working_roll,horizon=20.,hold_s=1.):
    t=np.asarray(data['time']);n=len(t)
    if not n:raise ValueError('empty executed trajectory is UNKNOWN')
    physical=bool(np.any(data['physical_failure']))
    domain=bool(np.any(data['domain_exit']))
    complete=bool(t[-1]+.005>=horizon-1e-5)
    tail=t>=horizon-hold_s-1e-5
    hold=bool(complete and tail.sum()==round(hold_s/.005) and
        np.all(np.abs(data['path_cross_track'][tail])<=.1) and
        np.all(np.abs(data['path_heading_error'][tail])<=.05) and
        np.all(np.abs(data['actual_forward_speed'][tail]-speed)<=.1))
    goal=bool(np.any((data['goal_section_signed_distance']>=0)&(data['path_progress']>=goal_s)))
    work=bool(np.any(data['peak_roll']>working_roll))
    recovered=bool(complete and not physical and not domain and hold and goal)
    return dict(physical_failure=physical,domain_failure=domain,
        physical_survived_horizon=False if physical else True if complete else None,
        complete_horizon=complete,final_hold=hold,goal=goal,working_failure=work,
        task_recovered=recovered,performance_success=recovered and not work,
        peak_roll_rad=float(np.max(data['peak_roll'])),
        peak_abs_path_m=float(np.max(np.abs(data['path_cross_track']))),
        tail_max_path_m=float(np.max(np.abs(data['path_cross_track'][tail]))) if tail.any() else None,
        actual_suffix_ticks=n)


def collect(declaration_path,output):
    import os,time,pickle,subprocess
    import jax
    import jax.numpy as j
    from .preference_best import atomic_json,digest
    from .path_command_training import load_environment
    from .path_command_policy import make_path_actor,validate_actor_parameters
    from .path_command_reward import configured_cost_keys
    from .runtime import configure_compilation_cache
    from .tolerance_state_features import snapshot_features
    from .empirical_tolerance import summarize_direction
    declaration=json.loads(Path(declaration_path).read_text());out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('immutable scan attempt exists')
    started=time.monotonic();ticks=0
    state=dict(state='running',phase='running',pid=os.getpid(),training_steps=0)
    def status(**kw):
        state.update(kw);state.update(checked_unix=time.time(),elapsed_s=time.monotonic()-started,active_lower_ticks=ticks)
        atomic_json(out/'status.json',state)
    status(stage='loading',declared_cases=len(declaration['bank_indices'])*len(declaration['forces_N']),maximum_lower_ticks=declaration['maximum_lower_ticks'])
    try:
        bankdir=Path(declaration['bank_dir']);bank_manifest=json.loads((bankdir/'manifest.json').read_text())
        if 'bank_manifest_sha256' in declaration and digest(bankdir/'manifest.json')!=declaration['bank_manifest_sha256']:
            raise ValueError('bank manifest identity changed')
        if bank_manifest['engineering_only'] or bank_manifest['data_role']!='new random TRAIN prefixes; never DEV states':
            raise ValueError('authentic TRAIN bank required')
        if digest(bankdir/'bank.pkl')!=declaration['bank_sha256']:raise ValueError('bank identity changed')
        if digest(declaration['suffix_actor'])!=declaration['suffix_actor_sha256']:raise ValueError('suffix actor changed')
        with (bankdir/'bank.pkl').open('rb') as f:bank=pickle.load(f)
        ids=np.asarray(declaration['bank_indices'],np.int32);forces=np.asarray(declaration['forces_N'],np.float32)
        if len(set(ids.tolist()))!=len(ids) or not np.all(np.asarray(bank['valid'])[ids]):raise ValueError('invalid/duplicate bank states')
        if len(set(forces.tolist()))!=len(forces) or np.sum(forces==0)!=1:raise ValueError('one zero and unique signed forces required')
        indices=np.repeat(ids,len(forces));force=np.tile(forces,len(ids));n=len(indices)
        contract=scan_contract(declaration,bank_manifest,ids,forces)
        horizon=contract['horizon_s'];deadline=round(horizon/.005)
        roles=dict(zip(ids,contract['roles']))
        cfg=bank_manifest['config'];source=bank_manifest['source']
        b=cfg['stage_B_baseline']
        if (b['final_y_error_m'],b['final_path_heading_error_rad'],b['final_speed_error_m_s'])!=(.1,.05,.1):
            raise ValueError('label implementation requires original hold bands')
        cache=configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,_=load_environment(cfg,source,'terminal_contract_v2')
        states=jax.tree.map(lambda x:j.asarray(x)[indices],bank['states'])
        routes=jax.tree.map(lambda x:j.asarray(x)[indices],bank['routes'])
        env.set_log_template(jax.tree.map(lambda x:x[0],states),jax.tree.map(lambda x:x[0],routes.path),routes.speed[0])
        if np.any(np.asarray(states.tick)%4):raise ValueError('control-aligned prefix states required')
        with Path(declaration['suffix_actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
        validate_actor_parameters(params,cfg);actor=make_path_actor(cfg)
        if 'suffix_checkpoint' in declaration and digest(declaration['suffix_checkpoint'])!=declaration['suffix_checkpoint_sha256']:
            raise ValueError('suffix checkpoint changed')
        heading=jax.vmap(lambda s:env.physics.helpers.pose(s.physical.data)[2])(states)
        vectors=j.asarray(force)[:,None]*j.stack([-j.sin(heading),j.cos(heading),j.zeros(n)],axis=1)
        onset=states.physical.physical_tick*.005
        pulse_duration=float(declaration.get('duration_s',.1))
        if not np.isfinite(pulse_duration) or pulse_duration<=0 or np.any(np.asarray(states.tick)*.005+pulse_duration>horizon):
            raise ValueError('pulse extends beyond original deadline')
        pulses=j.concatenate([onset[:,None],(onset+pulse_duration)[:,None],vectors],axis=1)
        keep=['active_tick','time','actual_xy','reference_xy','actual_forward_speed','wheel_speed_proxy','v_user',
            'path_cross_track','path_heading_error','path_progress','goal_section_signed_distance',
            'phi','phi_dot','actual_delta','peak_roll','physical_failure','domain_exit','policy_fault',
            'lower_fault','governed','nominal','applied_force_impulse','scored_tick_reward']
        zero={k:(j.zeros(4) if k=='scored_tick_reward' else env.zero_log[k][None,...].repeat(4,axis=0)) for k in keep}
        cost_keys=configured_cost_keys(cfg);zero['effective_costs']=j.zeros((4,len(cost_keys)))
        def one(s,path,v,pulse):
            def advance(st):
                ob,_,fault=env.observation(st,path,v,horizon)
                end,logs=env.policy_step(st.replace(fault=st.fault|fault),actor.apply(params,ob),path,v,horizon,pulse)
                return end,dict({k:logs[k] for k in keep},effective_costs=j.stack([logs['effective_components'][k] for k in cost_keys],axis=-1))
            stopped=(s.tick>=deadline)|s.physical.failed|s.domain_exit|s.fault
            return jax.lax.cond(stopped,lambda st:(st,zero),advance,s)
        def chunk(st,length):
            return jax.lax.scan(lambda s,_:jax.vmap(one)(s,routes.path,routes.speed,pulses),st,None,length=length)
        functions={}
        def execute(st,length):
            if length not in functions:functions[length]=jax.jit(lambda x:chunk(x,length))
            return functions[length](st)
        features=[]
        for idx in ids:
            s=jax.tree.map(lambda x:x[idx],bank['states']);r=jax.tree.map(lambda x:x[idx],bank['routes'])
            features.append(dict(bank_index=int(idx),features=snapshot_features(s,speed_request=float(r.speed),
                onset_s=float(s.tick)*.005,controller=source['controller'],horizon_s=horizon)))
        manifest=dict(declaration=declaration,prefix_bank=bank_manifest,config=cfg,source=source,
            features=features,batch=n,scan_policy_steps=50,training_steps=0,cache=str(cache),
            scan_contract=contract,reward_components=list(cost_keys),
            code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            semantics='real parent TRAIN prefix + frozen declared suffix; original clocks retained; not single-actor full-episode evidence')
        atomic_json(out/'manifest.json',manifest);saved=[]
        maximum_intervals=contract['maximum_policy_intervals']
        for start in range(0,maximum_intervals,50):
            length=min(50,maximum_intervals-start)
            status(stage='scanning',completed_policy_intervals=start,maximum_policy_intervals=maximum_intervals)
            states,logs=execute(states,length);jax.block_until_ready(states);host=jax.device_get(logs)
            ticks+=int(host['active_tick'].sum());saved.append(host)
            if np.any(host['policy_fault']) or np.any(host['lower_fault']) or np.any(np.asarray(states.fault)):
                raise RuntimeError('engineering fault; remaining cases UNKNOWN')
            if ticks>declaration['maximum_lower_ticks']:raise RuntimeError('scan exceeded declared physics')
            if np.all(np.asarray((states.tick>=deadline)|states.physical.failed|states.domain_exit)):break
        logs={k:np.concatenate([block[k] for block in saved],axis=0) for k in saved[0]};rows=[]
        for lane,(idx,f) in enumerate(zip(indices,force)):
            data={k:x[:,lane].reshape((-1,)+x.shape[3:]) for k,x in logs.items()}
            data={k:x[data['active_tick']] for k,x in data.items()}
            name=f'bank{idx:03d}_f{f:+g}';np.savez_compressed(out/(name+'.npz'),**data)
            labels=outcome_labels(data,speed=float(routes.speed[lane]),goal_s=float(routes.path.goal[lane]),working_roll=b['working_roll_rad'],horizon=horizon,hold_s=b['hold_s'])
            impulse=data['applied_force_impulse'].sum(axis=0);expected=np.asarray(vectors[lane])*pulse_duration
            complete_pulse=bool(data['time'][-1]+.005>=float(bank['states'].tick[idx])*.005+pulse_duration-1e-5)
            rows.append(dict(name=name,bank_index=int(idx),force_N=float(f),speed=float(routes.speed[lane]),
                alpha=float(bank['states'].alpha[idx]),onset_s=float(bank['states'].tick[idx])*.005,
                family=int(routes.family[lane]),data_role=roles[idx],
                **labels,pulse_complete=complete_pulse,impulse_error_Ns=float(np.linalg.norm(impulse-expected)),
                prefix_actor_sha256=bank_manifest['parent']['actor_sha256'],suffix_actor_sha256=declaration['suffix_actor_sha256']))
        boundaries=[]
        for idx in ids:
            for direction in [-1,1]:
                group=[r for r in rows if r['bank_index']==idx and (r['force_N']==0 or np.sign(r['force_N'])==direction)]
                boundaries.append(dict(bank_index=int(idx),direction=direction,
                    task=summarize_direction([(abs(r['force_N']),r['task_recovered']) for r in group]),
                    performance=summarize_direction([(abs(r['force_N']),r['performance_success']) for r in group]),
                    physical=summarize_direction([(abs(r['force_N']),r['physical_survived_horizon']) for r in group])))
        atomic_json(out/'results.json',dict(cases=rows,boundaries=boundaries))
        status(state='completed',phase='completed',stage='dataset_ready',completed_cases=len(rows),maximum_lower_ticks=declaration['maximum_lower_ticks'])
    except BaseException as exc:
        status(state='error',phase='error',error=f'{type(exc).__name__}: {exc}');raise
