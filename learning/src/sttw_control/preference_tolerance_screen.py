"""Declared frozen preference-Actor pulse panel, including logged V3 rewards."""
from pathlib import Path
import json


def evaluate(declaration_path,output):
    import os,time,pickle,subprocess
    from itertools import product
    from dataclasses import asdict
    import numpy as np
    import jax
    import jax.numpy as j
    from .preference_best import atomic_json,digest
    from .path_command_training import load_environment
    from .path_command_policy import make_path_actor,validate_actor_parameters
    from .path_command_reward import configured_cost_keys
    from .geometric_path import build_path
    from .tolerance_comparison import chunk_schedule
    from .tolerance_state_features import snapshot_features
    from .recovery_margin_scan import outcome_labels
    from .runtime import configure_compilation_cache
    decl=json.loads(Path(declaration_path).read_text());out=Path(output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('immutable preference screen exists')
    cfg=decl['config'];source=decl['source'];horizon=cfg['training_future_phase_C']['episode_s']
    rows=[dict(policy=p,route=r['name'],speed=v,alpha=a,force_N=f)
          for p,r,v,a,f in product(decl['policies'],decl['routes'],decl['speeds'],decl['alphas'],decl['forces_N'])]
    count=len(rows)*len(decl['onsets_s']);maximum=count*round(horizon/.005)
    if maximum!=decl['maximum_lower_ticks']:raise ValueError('declared physics mismatch')
    n=len(rows);ticks=0;cases=[];started=time.monotonic()
    status=dict(state='running',phase='running',pid=os.getpid(),training_steps=0,
        declared_cases=count,maximum_lower_ticks=maximum)
    def update(**kw):
        status.update(kw);status.update(checked_unix=time.time(),elapsed_s=time.monotonic()-started,
            active_lower_ticks=ticks,completed_cases=len(cases));atomic_json(out/'status.json',status)
    update(stage='loading');actors={}
    try:
        for name,record in decl['policies'].items():
            if digest(record['actor'])!=record['actor_sha256']:raise ValueError('Actor identity changed')
            if digest(record['checkpoint'])!=record['checkpoint_sha256']:raise ValueError('checkpoint identity changed')
            with Path(record['actor']).open('rb') as f:actors[name]=jax.tree.map(j.asarray,pickle.load(f))
            validate_actor_parameters(actors[name],cfg)
        cache=configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,snapshot=load_environment(cfg,source,'terminal_contract_v2')
        pose=np.asarray(env.physics.helpers.pose(snapshot.data));actor=make_path_actor(cfg)
        pathbank={r['name']:build_path(r['builder'],pose,cfg) for r in decl['routes']}
        for name,path in pathbank.items():np.savez_compressed(out/(name+'_path.npz'),xy=np.asarray(path.xy),s=np.asarray(path.s),goal=np.asarray(path.goal))
        paths=jax.tree.map(lambda *x:j.stack(x),*[pathbank[r['route']] for r in rows])
        params=jax.tree.map(lambda *x:j.stack(x),*[actors[r['policy']] for r in rows])
        speeds=j.asarray([r['speed'] for r in rows]);alphas=j.asarray([r['alpha'] for r in rows])
        initial=jax.jit(jax.vmap(lambda p,v,a:env.reset(snapshot,p,v,a)))(paths,speeds,alphas)
        sample=jax.tree.map(lambda x:x[0],initial)
        env.set_log_template(sample,pathbank[rows[0]['route']],speeds[0])
        cc=asdict(env.cc);keys=configured_cost_keys(cfg)
        manifest=dict(declaration=decl,config=cfg,source=source,lane_layout=rows,
            physical_identity=env.physics.bundle.identity,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            batch=n,cache=str(cache),training_steps=0,semantics='one frozen alpha-conditioned Actor; own zero-force complete prefix branched only within identical policy/alpha/path/speed context',
            labels='task return versus original working-performance; source hold/horizon preserved',reward_components=list(keys))
        atomic_json(out/'manifest.json',manifest)
        context=lambda r:tuple(r[k] for k in ('policy','route','speed','alpha'))
        zeros={context(r):i for i,r in enumerate(rows) if r['force_N']==0}
        branch_index=j.asarray([zeros[context(r)] for r in rows])
        keep=['active_tick','time','actual_xy','reference_xy','actual_forward_speed','wheel_speed_proxy','v_user',
            'path_cross_track','path_heading_error','path_progress','goal_section_signed_distance',
            'phi','phi_dot','actual_delta','peak_roll','physical_failure','domain_exit','policy_fault',
            'lower_fault','governed','nominal','applied_force_impulse','scored_tick_reward']
        def one(st,par,path,v,pulse):
            obs,_,fault=env.observation(st,path,v,horizon)
            end,log=env.policy_step(st.replace(fault=st.fault|fault),actor.apply(par,obs),path,v,horizon,pulse)
            return end,dict({k:log[k] for k in keep},effective_costs=j.stack([log['effective_components'][k] for k in keys],axis=-1))
        functions={}
        def execute(length,states,pulses):
            if length not in functions:
                functions[length]=jax.jit(lambda st,ps:jax.lax.scan(
                    lambda s,_:jax.vmap(one)(s,params,paths,speeds,ps),st,None,length=length))
            return functions[length](states,pulses)
        absolute_start=float(snapshot.physical_tick)*env.cc.dt
        for onset in decl['onsets_s']:
            states=initial;pulses=j.zeros((n,5));saved=[];features={};vectors=None
            for start,length,is_branch in chunk_schedule(onset,duration=horizon):
                if is_branch:
                    states=jax.tree.map(lambda x:x[branch_index],states)
                    saved=[jax.tree.map(lambda x:x[:,np.asarray(branch_index)],block) for block in saved]
                    for idx in zeros.values():
                        row=rows[idx];st=jax.device_get(jax.tree.map(lambda x:x[idx],states))
                        name=f"{row['policy']}_{row['route']}_v{row['speed']:g}_a{row['alpha']:g}_t{onset:g}_snapshot"
                        with (out/(name+'.pkl')).open('wb') as f:pickle.dump(st,f)
                        valid=int(st.tick)==round(onset/.005) and not bool(st.fault|st.domain_exit|st.physical.failed)
                        features[context(row)]=snapshot_features(st,speed_request=row['speed'],onset_s=onset,controller=cc,horizon_s=horizon) if valid else None
                    heading=np.asarray(jax.vmap(lambda s:env.physics.helpers.pose(s.physical.data)[2])(states))
                    direction=np.stack([-np.sin(heading),np.cos(heading),np.zeros(n)],axis=-1)
                    vectors=np.asarray([r['force_N'] for r in rows])[:,None]*direction
                    pulses=j.asarray(np.column_stack([np.full(n,absolute_start+onset),np.full(n,absolute_start+onset+decl['duration_s']),vectors]))
                update(stage='screening',onset_s=onset,task_seconds=start*.02)
                states,logs=execute(length,states,pulses);jax.block_until_ready(states);host=jax.device_get(logs)
                ticks+=int(host['active_tick'].sum());saved.append(host)
                if ticks>maximum:raise RuntimeError('physics accounting exceeded declaration')
                if np.any(host['policy_fault']) or np.any(host['lower_fault']) or np.any(np.asarray(states.fault)):
                    raise RuntimeError('engineering fault; incomplete cases UNKNOWN')
                update(stage='screening',onset_s=onset,task_seconds=(start+length)*.02)
            logs={k:np.concatenate([block[k] for block in saved],axis=0) for k in saved[0]}
            for idx,row in enumerate(rows):
                data={k:x[:,idx].reshape((-1,)+x.shape[3:]) for k,x in logs.items()}
                data={k:x[data['active_tick']] for k,x in data.items()}
                name=f"{row['policy']}_{row['route']}_v{row['speed']:g}_a{row['alpha']:g}_t{onset:g}_f{row['force_N']:+g}"
                np.savez_compressed(out/(name+'.npz'),**data)
                b=cfg['stage_B_baseline'];label=outcome_labels(data,speed=row['speed'],goal_s=float(pathbank[row['route']].goal),
                    working_roll=b['working_roll_rad'],horizon=horizon,hold_s=b['hold_s'])
                before=data['time']<onset-1e-5
                prefix_valid=not bool(np.any((data['physical_failure']|data['domain_exit']|(data['peak_roll']>b['working_roll_rad']))&before)) and len(data['time'])>=round(onset/.005)
                impulse=data['applied_force_impulse'].sum(0);expected=vectors[idx]*decl['duration_s']
                cases.append(dict(**row,name=name,onset_s=onset,**label,prefix_valid=prefix_valid,
                    prefix_features=features[context(row)],pulse_complete=len(data['time'])>=round((onset+decl['duration_s'])/.005),
                    impulse_error_Ns=float(np.linalg.norm(impulse-expected)),logged_return=float(data['scored_tick_reward'].sum())))
            atomic_json(out/'results.json',dict(cases=cases))
        update(state='completed',phase='completed',stage='screen_ready')
    except BaseException as exc:
        update(state='error',phase='error',error=f'{type(exc).__name__}: {exc}');raise
