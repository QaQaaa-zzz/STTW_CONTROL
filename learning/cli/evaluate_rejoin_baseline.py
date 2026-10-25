"""Prepare, trace, or execute the sealed 36-episode held/live-zero DEV pilot."""
import argparse
import copy
import json
import os
import pickle
from pathlib import Path
import time
import traceback
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def prepare(endpoint,output):
    from sttw_control.rejoin_comparison import prepare_cases
    from sttw_control.survival_endpoint_models import authenticate_endpoint
    endpoint=endpoint.resolve();output=output.resolve()
    source_status=json.loads((endpoint/'status.json').read_text())
    if source_status['phase']!='completed':raise ValueError('completed endpoint required')
    original=json.loads((endpoint/'plan.json').read_text())
    protocol=json.loads((Path(original['protocol'])/'plan.json').read_text())
    authenticated=authenticate_endpoint(protocol['training_run'])
    cases=prepare_cases(json.loads((endpoint/'cases.json').read_text()))
    training=json.loads((Path(protocol['training_run'])/'plan.json').read_text())
    declaration=Path(training['source_declaration'])
    output.mkdir(parents=True,exist_ok=False)
    (output/'common_environment_config.json').write_bytes((endpoint/'common_environment_config.json').read_bytes())
    (output/'source_declaration.json').write_bytes(declaration.read_bytes())
    atomic_json(output/'cases.json',cases);atomic_json(output/'authenticated_models.json',authenticated)
    module=Path(__file__).parents[1]/'src/sttw_control'
    files=list(module.glob('*.py'))+[Path(__file__)]
    archive=output/'source_snapshot';archive.mkdir()
    for i,p in enumerate(sorted(files)):(archive/f'{i:03d}_{p.name}').write_bytes(p.read_bytes())
    plan=dict(project='STTW_CONTROL',purpose='Test whether live original-path following with zero correction after domain exit adds recovery beyond held references.',
        source_endpoint=str(endpoint),source_endpoint_plan_sha256=digest(endpoint/'plan.json'),
        training_run=protocol['training_run'],seed=protocol['seed'],data_role='DEV',
        declared_actual_episodes=36,paired_source_conditions=18,horizon_s=10.,lower_dt_s=.005,
        maximum_active_lower_ticks=72000,maximum_allocated_lower_slots=79200,
        steps_per_chunk=50,max_chunks_per_group=11,optimizer_updates=0,
        config_sha256=digest(output/'common_environment_config.json'),
        declaration_sha256=digest(output/'source_declaration.json'),cases_sha256=digest(output/'cases.json'),
        models_sha256=digest(output/'authenticated_models.json'),
        source_sha256={str(p.resolve()):digest(p) for p in files},
        termination='Real contact, engineering fault or 10s; domain exit latches but both physical tails continue. Actor never queried after exit.',
        comparison='Same full task start/model/user alpha/event for each tail pair; not independent seeds, not identical physical states across different controllers.',
        metrics='Original fixed world path/progress/goal and final 1s .1m/.05rad/.1m_s bands, recomputed offline from exact actual yaw/XY for BOTH tails. Domain crossing is a diagnostic under this new contract, not automatic task failure. Continuous errors, displacement and reference traces retained.',
        legacy='Original 488 cases/labels remain immutable. Original prefix terminal rewards retained as telemetry; do not treat them as new recovery-task returns.',
        next_decision='Inspect paired joint task gains/losses and contacts. A useful simple baseline becomes a common component, not a new learning contribution. No automatic extra training, candidates, restarts or horizon extension.')
    atomic_json(output/'plan.json',plan)
    atomic_json(output/'status.json',dict(phase='prepared',declared_cases=36,completed_cases=0,
        active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0,checked_unix=time.time()))


def setup(root):
    import jax
    import jax.numpy as j
    from sttw_control.contact_only_environment import load_environment_contact_only
    from sttw_control.geometric_path import build_path
    from sttw_control.survival_endpoint_batch import partition_cases
    plan=json.loads((root/'plan.json').read_text())
    for name,key in [('common_environment_config.json','config_sha256'),('source_declaration.json','declaration_sha256'),('cases.json','cases_sha256'),('authenticated_models.json','models_sha256')]:
        if digest(root/name)!=plan[key]:raise ValueError('changed sealed '+name)
    for p,h in plan['source_sha256'].items():
        if digest(p)!=h:raise ValueError('changed sealed source '+p)
    models=json.loads((root/'authenticated_models.json').read_text())['models']
    for model in models.values():
        for kind in ('actor','checkpoint'):
            if model.get(kind) and digest(model[kind])!=model[kind+'_sha256']:raise ValueError('changed frozen '+kind)
    cfg=json.loads((root/'common_environment_config.json').read_text())
    declaration=json.loads((root/'source_declaration.json').read_text())
    env,initial=load_environment_contact_only(cfg,declaration['source'],'terminal_contract_v2')
    pose=np.asarray(env.physics.helpers.pose(initial.data))
    paths={r['name']:build_path(r['builder'],pose,cfg) for r in declaration['routes']}
    cases=json.loads((root/'cases.json').read_text());groups=[]
    # Same compiled function is reused for the two tail rules; tail is a shared
    # scalar within each batch, avoiding evaluation of both physical tail modes.
    for tail in ['held','live_zero']:
        indices=[i for i,c in enumerate(cases) if c['tail_rule']==tail]
        selected=[cases[i] for i in indices]
        pieces=partition_cases(selected,{('physical',k):tuple(v.s.shape) for k,v in paths.items()})
        for piece in pieces:
            groups.append(dict(piece,group_id=f'group_{len(groups):03d}',tail_rule=tail,
                               case_indices=[indices[i] for i in piece['case_indices']]))
    params={}
    for role,model in models.items():
        if model.get('actor'):
            with Path(model['actor']).open('rb') as f:params[role]=jax.tree.map(j.asarray,pickle.load(f))
    return plan,cfg,env,initial,paths,cases,groups,params,pose


def execute(root,*,shape_only=False):
    from prepare_short_recovery_campaign import register
    target=root/('interface_status.json' if shape_only else 'status.json')
    state=dict(phase='running',stage='authenticate_and_load',pid=os.getpid(),
               physics_steps=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0,
               declared_cases=36,completed_cases=0,checked_unix=time.time())
    if not shape_only and json.loads(target.read_text())['phase']!='prepared':raise ValueError('only first prepared execution allowed')
    def update(**kwargs):
        state.update(kwargs,checked_unix=time.time());atomic_json(target,state)
    update()
    register(root/('INTERFACE_ACTIVE_RUN.json' if shape_only else 'ACTIVE_RUN.json'),
        'STTW_CONTROL 实时原路径接回对照'+('零积分接口' if shape_only else '36回合物理筛查'),target)
    try:
        import jax
        import jax.numpy as j
        from sttw_control.rejoin_comparison import make_comparison_step
        from sttw_control.survival_collection import SurvivalCollectionState
        from sttw_control.survival_endpoint_collection import evaluation_config
        from sttw_control.survival_endpoint_batch import simulation_alpha
        from sttw_control.path_command_policy import make_path_actor
        from sttw_control.runtime import configure_compilation_cache
        plan,cfg,env,initial,paths,cases,groups,params,pose=setup(root)
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        atomic_json(root/'groups.json',groups)
        if not shape_only:
            (root/'reference_paths').mkdir();(root/'cases').mkdir()
            for name,path in paths.items():
                np.savez_compressed(root/'reference_paths'/f'{name}.npz',**{k:np.asarray(getattr(path,k)) for k in ['s','xy','heading','curvature','goal','turn_end']})
        stack=lambda values:jax.tree.map(lambda *x:j.stack(x),*values)
        compiled={};index=[];rows=[];shape_records=[];began=time.monotonic()
        for group in groups:
            selected=[cases[i] for i in group['case_indices']]
            key=(group['inference_mode'],group['speed_reference_floor_m_s'],len(selected),tuple(group['path_shape']))
            local=copy.copy(env);local.config=evaluation_config(cfg,speed_floor=group['speed_reference_floor_m_s'])
            states=[local.reset(initial,paths[c['route']],j.float32(c['speed']),j.float32(simulation_alpha(c))) for c in selected]
            local.set_log_template(states[0],paths[selected[0]['route']],j.float32(selected[0]['speed']))
            if any(bool(s.physical.failed|s.fault|s.domain_exit) for s in states):raise ValueError('invalid full task initial state')
            carry=stack([SurvivalCollectionState(s,jax.random.PRNGKey(plan['seed']),j.zeros(2),j.bool_(False)) for s in states])
            zero=group['inference_mode']=='zero'
            parameters=None if zero else stack([params[c['role']] for c in selected])
            batchpath=stack([paths[c['route']] for c in selected]);speed=j.asarray([c['speed'] for c in selected],j.float32)
            start=float(initial.physical_tick)*local.cc.dt
            pulse=j.asarray([[start+c['pulse_onset_task_tick']*local.cc.dt,
                start+c['pulse_onset_task_tick']*local.cc.dt+c['pulse_duration_s'],
                *(c['force_N']*np.asarray(c['world_direction']))] for c in selected],j.float32)
            use_live=j.bool_(group['tail_rule']=='live_zero')
            if key not in compiled:
                one=make_comparison_step(local,None if zero else make_path_actor(cfg),horizon_s=10.,mode=group['inference_mode'])
                def build(one,zero):
                    @jax.jit
                    def chunk(c,p,path,v,force,live):
                        return jax.lax.scan(lambda st,_:jax.vmap(one,in_axes=(0,None if zero else 0,0,0,0,None))(
                            st,p,path,v,force,live),c,None,length=50)
                    return chunk
                compiled[key]=build(one,zero)
            chunk=compiled[key]
            if shape_only:
                shape=jax.eval_shape(chunk,carry,parameters,batchpath,speed,pulse,use_live)
                shape_records.append(dict(group=group['group_id'],batch=len(selected),
                    active_shape=list(shape[1]['active_tick'].shape),actual_yaw_shape=list(shape[1]['actual_yaw'].shape)))
                continue
            directory=root/group['group_id'];directory.mkdir()
            with (directory/'initial.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
            chunks=[]
            for block in range(plan['max_chunks_per_group']):
                update(stage='collecting',current_group=group['group_id'],current_block=block)
                carry,logs=chunk(carry,parameters,batchpath,speed,pulse,use_live);jax.block_until_ready(carry)
                host,data=jax.device_get((carry,logs))
                flat={k:np.swapaxes(v,1,2).reshape((-1,v.shape[1])+v.shape[3:]) for k,v in data.items()}
                np.savez_compressed(directory/f'chunk_{block:02d}.npz',**flat)
                with (directory/'latest_state.pkl').open('wb') as f:pickle.dump(host,f)
                chunks.append(flat)
                update(active_lower_ticks=state['active_lower_ticks']+int(flat['active_tick'].sum()),
                       allocated_lower_slots=state['allocated_lower_slots']+200*len(selected),elapsed_s=time.monotonic()-began)
                if state['active_lower_ticks']>plan['maximum_active_lower_ticks'] or state['allocated_lower_slots']>plan['maximum_allocated_lower_slots']:raise ValueError('declared physical accounting exceeded')
                if np.any(host.state.fault):raise ValueError('engineering fault; raw saved, no restart')
                if np.all(host.state.physical.failed|(host.state.tick>=2000)):break
            else:raise ValueError('missing full horizon after declared chunks')
            for position,(case_id,case) in enumerate(zip(group['case_indices'],selected)):
                data={k:np.concatenate([a[k][:,position] for a in chunks]) for k in chunks[0]}
                file=root/'cases'/f'case_{case_id:03d}.npz';np.savez_compressed(file,**data)
                active=data['active_tick'];contact=bool(np.any(data['contact_failure']&active))
                n=int(active.sum());outcome='contact' if contact else 'survived' if n==2000 else 'unknown'
                rows.append(dict(case,physical_outcome=outcome,physical_ticks=n,
                    prefix_ticks=int(data['prefix_active'].sum()),live_ticks=int(data['live_tracking_active'].sum()),
                    initial_xy=pose[:2].tolist(),initial_yaw=float(pose[2]),
                    actual_impulse_Ns=data['applied_force_impulse'][active].sum(axis=0).tolist()))
                index.append(dict(case_id=case['case_id'],file=str(file.relative_to(root)),sha256=digest(file),
                    group=group['group_id'],reference=f"reference_paths/{case['route']}.npz"))
            atomic_json(root/'results.json',rows);atomic_json(root/'case_index.json',index)
            update(completed_cases=len(rows))
        if shape_only:
            atomic_json(root/'interface_receipt.json',dict(groups=shape_records,physics_steps=0,optimizer_updates=0,scope='Actual batch/kernel tracing only, not GPU rollout.'))
            update(phase='completed',stage='all_six_groups_traced_no_integration')
        else:
            if len(rows)!=36 or any(r['physical_outcome']=='unknown' for r in rows):raise ValueError('incomplete physical matrix')
            update(phase='completed',stage='physical_collection_complete_analysis_pending',physics_steps=state['active_lower_ticks'])
    except BaseException as exc:
        (root/('interface_traceback.txt' if shape_only else 'traceback.txt')).write_text(traceback.format_exc())
        update(phase='error',error=repr(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','shape','execute'],required=True)
    args=parser.parse_args()
    if args.phase=='prepare':prepare(args.endpoint,args.output)
    else:execute(args.output.resolve(),shape_only=args.phase=='shape')
