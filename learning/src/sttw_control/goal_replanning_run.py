"""Sealed, opt-in 20-second original/replanned-goal comparison; no training."""
import copy
import json
import os
import pickle
import time
import traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest


def prepare(source,root):
    source=Path(source).resolve();root=Path(root).resolve()
    p=json.loads((source/'plan.json').read_text())
    if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('changed historical source plan')
    contexts={k:v for k,v in p['contexts'].items() if k.startswith('v1.7_yaw+0__')}
    if len(contexts)!=2:raise ValueError('exact two user-illustrated routes required')
    original=json.loads((source/'prefix_inventory.json').read_text())['cases']
    selected=[c for c in original if c['context'] in contexts and c['force_N'] in (0.,-32.,32.)]
    if len(selected)!=12:raise ValueError('two routes x two original alpha x three events required')
    files=[source/'plan.json',source/'prefix_inventory.json',Path(p['model']['actor'])]
    for c in contexts.values():
        for key in ('initial','path'):
            if digest(c[key])!=c[key+'_sha256']:raise ValueError('historical initial/path changed')
            files.append(Path(c[key]))
    if digest(p['model']['actor'])!=p['model']['actor_sha256']:raise ValueError('middle identity changed')
    root.mkdir(parents=True,exist_ok=False)
    for name in ('common_environment_config.json','source_declaration.json'):
        (root/name).write_bytes((source/name).read_bytes());files.append(root/name)
    cases=[]
    for c in selected:
        for role in ('original_full','replan_full','original_zero','replan_zero'):
            new=copy.deepcopy(c);new.pop('fork_task_tick')
            new.update(role=role,case_id=c['case_id'].rsplit('/',1)[0]+'/'+role,
                       source_keep_channels=[not role.endswith('zero')]*2)
            cases.append(new)
    atomic_json(root/'inventory.json',cases);files.append(root/'inventory.json')
    modules=list(Path(__file__).parent.rglob('*.py'))+[Path('learning/cli/run_goal_replanning.py').resolve()]
    files.extend([Path(__file__).parent/'_vendor/pythonrobotics'/n for n in ('LICENSE','UPSTREAM.json')])
    archive=root/'source';archive.mkdir()
    for i,f in enumerate(modules):(archive/f'{i:03d}_{f.name}').write_bytes(f.read_bytes())
    plan=dict(project='STTW_CONTROL',experiment=root.name,source=str(source),
        contexts=contexts,model=p['model'],feedback_spec=p['feedback_spec'],
        cases=48,horizon_s=20.,lower_dt_s=.005,planning_period_s=1.,radius_m=1.8,
        departure_m=1.,goal_radius_m=1.,max_replans_per_case=1,
        maximum_active_lower_ticks=192000,maximum_allocated_lower_slots=192000,
        optimizer_updates=0,initialization='weights_only_for_inference; complete original task0 carry retained',
        baseline='Original route and replanned route both 20s; original 10s slice retained; full and zero neural corrections',
        data_role='Seen user-illustrated DEV contexts, related common ancestor; not independent TEST',
        trigger='At ordinary 1s boundaries: current follower invalid OR abs(cross_track)>1m, not ever-arrived/contact/fault/replanned. No force/onset/end input.',
        replanning='Only projection/follower and current path change. Original world destination, alpha, task/physical clock, history/ESO/lower/filter/raw reference, relay and cumulative domain-exit remain.',
        path='Pinned MIT PythonRobotics Dubins six families, eight world goal headings plus goal bearing, radius1.8. Continuous position/heading, bounded curvature with segment curvature jumps; no global free-heading shortest/safety certificate.',
        scoring='Original goal1m first entry before contact/fault at 10/20s; physical survival separately. Reference-conditioned reward is diagnostic, not a fair original-path cost after replanning.',
        stop='One fixed48 comparison, original contact/fault/20s; no retries/extra radii/times/gates/training or adoption. Any engineering error preserves partial evidence.',
        input_sha256={str(f.resolve()):digest(f) for f in files},
        source_sha256={str(f.resolve()):digest(f) for f in modules})
    atomic_json(root/'plan.json',plan);atomic_json(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
    return plan


def load(root):
    import jax
    import jax.numpy as j
    from .contact_only_environment import load_environment_contact_only
    from .goal_preview_baseline import GoalPreviewEnv,make_goal_step
    from .finite_value_evaluation import rebind_path_context
    from .finite_value_evaluation_run import authenticate_models
    from .path_command_policy import make_path_actor
    from .geometric_path import Path as GeometricPath
    from .causal_recovery_relay import CausalRelayState,initial_gate
    from .causal_reference_baseline import initial_gate as initial_reference_gate
    from .runtime import configure_compilation_cache
    from .evaluation_clock import physical_pulse
    from .goal_replanning import rebind_geometry
    root=Path(root).resolve();p=json.loads((root/'plan.json').read_text())
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('plan changed')
    for f,sha in {**p['input_sha256'],**p['source_sha256']}.items():
        if digest(f)!=sha:raise ValueError('sealed input/source changed '+f)
    cfg=json.loads((root/'common_environment_config.json').read_text())
    decl=json.loads((root/'source_declaration.json').read_text())
    authenticate_models(cfg,dict(free_GAE=p['model']))
    cache=configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
    env,_=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2');env=GoalPreviewEnv(env)
    cases=json.loads((root/'inventory.json').read_text());states=[];paths=[];goals=[]
    for c in cases:
        context=p['contexts'][c['context']]
        with np.load(context['path']) as z:path=GeometricPath(*(j.asarray(z[k]) for k in ('s','xy','heading','curvature','turn_end','goal')))
        with Path(context['initial']).open('rb') as f:bank=pickle.load(f).state
        state=jax.tree.map(lambda x:j.asarray(x[0]),bank)
        if int(state.tick)!=0 or int(state.physical.physical_tick)!=context['initial_physical_tick']:
            raise ValueError('complete initial clock differs')
        state=rebind_path_context(state.replace(alpha=j.float32(c['alpha'])),path,j.float32(c['requested_speed_m_s']),env)
        states.append(state);paths.append(path)
        goals.append([np.interp(float(path.goal),np.asarray(path.s),np.asarray(path.xy)[:,i]) for i in (0,1)])
    stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
    with Path(p['model']['actor']).open('rb') as f:native=jax.tree.map(j.asarray,pickle.load(f))
    env.set_log_template(states[0],paths[0],j.float32(cases[0]['requested_speed_m_s']))
    one=make_goal_step(env,make_path_actor(cfg),horizon_s=20.,spec=p['feedback_spec'])
    @jax.jit
    def chunk(carry,parameters,paths,speeds,pulses,modes):
        return jax.lax.scan(lambda c,_:jax.vmap(one)(c,parameters,paths,speeds,pulses,modes),carry,None,length=50)
    @jax.jit
    def measured_pose(carry):
        return jax.vmap(lambda d:env.physics.observe(d)[1])(carry.state.physical.data)
    @jax.jit
    def rebind_batch(carry,paths,speeds,mask):
        rebound=jax.vmap(lambda c,p,v:rebind_geometry(c,p,v,env))(carry,paths,speeds)
        return jax.tree.map(lambda a,b:j.where(mask.reshape((48,)+(1,)*(a.ndim-1)),a,b),rebound,carry)
    carry=CausalRelayState(stack(states),stack([initial_reference_gate() for _ in cases]),j.zeros(48,j.int32),stack([initial_gate() for _ in cases]))
    parameters=stack([dict(actor=dict(native=native,keep_channels=j.asarray(c['source_keep_channels'])),feedback_gain_rad=j.float32(.1)) for c in cases])
    speeds=j.asarray([c['requested_speed_m_s'] for c in cases],j.float32)
    pulses=j.asarray([physical_pulse(c,p['contexts'][c['context']]) for c in cases],j.float32)
    modes=j.ones(48,j.int32)
    return dict(root=root,plan=p,config=cfg,env=env,cases=cases,initial=states,
                original_paths=paths,goals=np.asarray(goals),cache=cache,chunk=chunk,
                measured_pose=measured_pose,rebind_batch=rebind_batch,
                args=(carry,parameters,stack(paths),speeds,pulses,modes))


def run(root,*,shape_only=False):
    import jax
    import jax.numpy as j
    from .anticipatory_run import flatten_ticks
    from .goal_replanning import dubins_goal_path,replan_requested
    root=Path(root).resolve();directory=root/('shape' if shape_only else 'physical')
    directory.mkdir(exist_ok=False);started=time.monotonic()
    atomic_json(directory/'status.json',dict(phase='initializing',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0,pid=os.getpid()))
    try:
        runtime=load(root);carry,parameters,batched_paths,speeds,pulses,modes=runtime['args']
        shaped=jax.eval_shape(runtime['chunk'],*runtime['args'])
        atomic_json(directory/'shape_receipt.json',dict(real_lanes=48,padding=0,upper_steps=50,
                    active_log_shape=shaped[1]['active_tick'].shape,physics_steps=0,cache_directory=runtime['cache']))
        if shape_only:
            atomic_json(directory/'status.json',dict(phase='completed',physics_steps=0,optimizer_updates=0));return
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('GPU required')
        with (directory/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        paths=list(runtime['original_paths']);changed=np.zeros(48,bool);arrived=np.zeros(48,bool)
        decisions=[];logs_all=[];actual=0;slots=0
        for block in range(20):
            if block:
                # Only measured current state / geometry enters these decisions.
                poses=jax.device_get(runtime['measured_pose'](carry))
                cross,invalid,failed,fault,ticks,domain,clean=jax.device_get((
                    carry.state.projection.cross_track,carry.state.follower.invalid,
                    carry.state.physical.failed,carry.state.fault,carry.state.tick,
                    carry.state.domain_exit,carry.clean_blocks))
                mask=np.zeros(48,bool)
                for i,c in enumerate(runtime['cases']):
                    pose=poses[i]
                    if c['role'].startswith('replan') and replan_requested(
                        xy=np.asarray(pose)[:2],goal_xy=runtime['goals'][i],
                        cross_track_m=float(cross[i]),
                        follower_invalid=bool(invalid[i]),already_replanned=bool(changed[i]),
                        contact=bool(failed[i]),fault=bool(fault[i]),ever_arrived=bool(arrived[i])):
                        path,meta=dubins_goal_path(pose,runtime['goals'][i])
                        paths[i]=path;changed[i]=True;mask[i]=True
                        np.savez_compressed(directory/f'replanned_path_{i:02d}.npz',**{k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')})
                        decisions.append(dict(case_id=c['case_id'],case_index=i,task_tick=int(ticks[i]),
                            ever_domain_exit=bool(domain[i]),retained_clean_blocks=int(clean[i]),**meta))
                if mask.any():
                    batched_paths=jax.tree.map(lambda *x:j.stack(x),*paths)
                    carry=runtime['rebind_batch'](carry,batched_paths,speeds,j.asarray(mask))
                atomic_json(directory/'replanning_decisions.json',decisions)
            carry,logs=runtime['chunk'](carry,parameters,batched_paths,speeds,pulses,modes);jax.block_until_ready(carry)
            logs=jax.device_get(logs);logs=flatten_ticks(logs);logs_all.append(logs)
            np.savez_compressed(directory/f'chunk_{block:02d}.npz',**logs)
            actual+=int(logs['active_tick'].sum());slots+=200*48
            fatal=logs['contact_failure']|logs['nonfinite']|logs['lower_fault']|logs['policy_fault']
            eligible=logs['active_tick']&(np.cumsum(fatal,axis=0)==0)
            distance=np.linalg.norm(logs['actual_xy']-runtime['goals'][None,:,:],axis=-1)
            arrived|=np.any(eligible&(distance<=1.),axis=0)
            atomic_json(directory/'status.json',dict(phase='running',completed_blocks=block+1,active_lower_ticks=actual,
                allocated_lower_slots=slots,replanned_cases=int(changed.sum()),optimizer_updates=0,elapsed_s=time.monotonic()-started,pid=os.getpid()))
            fault,done=jax.device_get((carry.state.fault,carry.state.physical.failed|(carry.state.tick>=4000)))
            if np.any(fault):raise RuntimeError('engineering fault; original partial chunks retained')
            if np.all(done):break
        host=jax.device_get(carry)
        with (directory/'final_carry.pkl').open('wb') as f:pickle.dump(host,f)
        (directory/'cases').mkdir();data={k:np.concatenate([x[k] for x in logs_all]) for k in logs_all[0]}
        index=[]
        for i,c in enumerate(runtime['cases']):
            file=directory/'cases'/f'case_{i:02d}.npz'
            np.savez_compressed(file,**{k:v[:,i] for k,v in data.items()})
            index.append(dict(c,raw_file=str(file),raw_sha256=digest(file),original_goal_xy=runtime['goals'][i].tolist(),replanned=bool(changed[i])))
        atomic_json(directory/'case_index.json',index)
        atomic_json(directory/'replanning_decisions.json',decisions)
        atomic_json(directory/'status.json',dict(phase='completed',completed_cases=48,active_lower_ticks=actual,
            allocated_lower_slots=slots,replanned_cases=int(changed.sum()),optimizer_updates=0,elapsed_s=time.monotonic()-started))
    except BaseException as exc:
        (directory/'traceback.txt').write_text(traceback.format_exc())
        status=json.loads((directory/'status.json').read_text());atomic_json(directory/'status.json',dict(status,phase='error',error=repr(exc)))
        raise
