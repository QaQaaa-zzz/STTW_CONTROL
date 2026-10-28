"""Fixed128 endpoint comparison after ordinary goal training, no retraining."""
import copy
import json
import os
import pickle
import shutil
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest
from .goal_evaluation import evaluation_cases


def prepare(training,root):
    training=Path(training).resolve();root=Path(root).resolve()
    state=json.loads((training/'status.json').read_text())
    if state['phase']!='completed' or state['collections_completed']!=64:raise ValueError('completed fixed64 training required, no best checkpoint selection')
    original=json.loads((training/'plan.json').read_text())
    original_seal=json.loads((training/'seal.json').read_text())
    if digest(training/'plan.json')!=original_seal['plan_sha256']:raise ValueError('original training design changed')
    if digest(training/'effective_config.json')!=original_seal['config_sha256']:raise ValueError('original training environment config changed')
    if original['horizon_s']!=10. or original['radius_m']!=1. or original['goal_distance_scale_m']!=10.:
        raise ValueError('original goal contract differs from implemented endpoint evaluation')
    endpoint=training/'checkpoints/learner_0064.pt'
    latest=json.loads((training/'latest_checkpoint.json').read_text())
    if Path(latest['path']).resolve()!=endpoint or latest['sha256']!=digest(endpoint):raise ValueError('fixed64 endpoint identity mismatch')
    cases=evaluation_cases()
    for case in cases:
        case['world_direction']=next(c['world_direction'] for c in original['reference_cases'] if c['context']==case['context'])
    if original['evaluation']['declared_physical_cases']['total']!=len(cases):raise ValueError('frozen evaluation inventory changed')
    # Conditions are fixed by training plan, not selected using endpoint results.
    design=original['evaluation']; t=design['transfer']; c=design['training_geometry_check']
    if (t!=dict(path_radius_m=1.25,angle_deg=90.,left_speed_m_s=1.7,right_speed_m_s=1.6,onsets_task_ticks=[360,640],forces_N=[0.,-32.,32.],duration_s=.1)
        or c!=dict(radius_m=1.1,left_speed_m_s=1.8,right_speed_m_s=1.5,onset_task_tick=500,forces_N=[0.,-32.,32.])):
        raise ValueError('implementation does not match preregistered DEV design')
    mount=Path('/media/qy/6A7E3DB07E3D75C3');stat=os.statvfs('/home/qy/STTW_CONTROL')
    if not os.path.ismount(mount) or not root.is_relative_to(mount) or stat.f_bavail*stat.f_frsize<50*1024**3:
        raise ValueError('data-disk output and50GiB system space required')
    root.mkdir(parents=True,exist_ok=False)
    inputs=dict(original['inputs']);inputs[str(training/'plan.json')]=digest(training/'plan.json')
    inputs[str(training/'seal.json')]=digest(training/'seal.json')
    inputs[str(training/'effective_config.json')]=digest(training/'effective_config.json');inputs[str(endpoint)]=digest(endpoint)
    sources={}
    for f in list(Path('learning/src/sttw_control').rglob('*.py'))+[Path('learning/cli/evaluate_goal_endpoint.py')]:
        sources[str(f.resolve())]=digest(f)
        target=root/'source_snapshot'/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
    plan=dict(project='STTW_CONTROL',experiment_id='STTW-ORDINARY-GOAL-ENDPOINT-V1',training=str(training),endpoint=str(endpoint),
        training_design=design,cases=cases,unique_physical_cases=128,logical_user_score_views=192,
        training_source_differences={name:dict(training_sha256=sha,current_sha256=digest(name) if Path(name).exists() else None)
            for name,sha in original['sources'].items() if not Path(name).exists() or digest(name)!=sha},
        maximum_lower_slots=256000,horizon_s=10.,radius_m=1.,batch_size=16,inputs=inputs,sources=sources,
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        control='deterministic mean, originalalpha, no optimizer; exact zero residual distinct from existing profile7 speed support',
        initialization='training_geometry: original full upper/lower task0; transfer: same original physical snapshot, env.reset for declared new task',
        data_role='related DEV joint geometry/speed/time transfer, not independent TEST or separate training repeats',
        stop='128 fixed executions only; engineering error preserves partials and stops, no retry/condition extension',
        acceleration=['same-shape16lane batches','scan50 blocks','dynamic parameter/path arguments','persistent cache'])
    atomic_json(root/'plan.json',plan);atomic_json(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'status.json',dict(phase='prepared',completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_steps=0))


def execute(root):
    root=Path(root).resolve();state=json.loads((root/'status.json').read_text())
    if state['phase']!='prepared':raise ValueError('fresh evaluation only; no implicit restart')
    def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(root/'status.json',state)
    update(phase='running',stage='loading',pid=os.getpid())
    try:
        import jax
        import jax.numpy as j
        import torch
        from .runtime import configure_compilation_cache
        from .goal_learner import GoalLearner
        from .goal_networks import GoalNetworkBundle,make_goal_collection_evaluator
        from .goal_evaluation import GoalEvaluationState,make_goal_evaluation_step
        from .path_command_policy import make_torch_path_network
        from .contact_only_environment import load_environment_contact_only
        from .causal_reference_baseline import initial_gate
        from .geometric_path import Path as GeometricPath,build_path
        from .anticipatory_run import flatten_ticks
        p=json.loads((root/'plan.json').read_text())
        if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed plan changed')
        for f,sha in dict(p['inputs'],**p['sources']).items():
            if digest(f)!=sha:raise ValueError('sealed input/source changed: '+f)
        training=Path(p['training']);old=json.loads((training/'plan.json').read_text())
        cfg=json.loads((training/'effective_config.json').read_text())
        declaration=json.loads((Path(old['source'])/'source_declaration.json').read_text())['source']
        torch.set_num_threads(4)
        learned=GoalLearner.restore(torch.load(p['endpoint'],map_location='cpu',weights_only=False))
        if learned.bundle.config!=cfg:raise ValueError('trained network environment config differs from original')
        if learned.collection_index!=64 or learned.optimizer_steps!=[248,256]:raise ValueError('wrong endpoint optimizer/collection boundary')
        parent=make_torch_path_network(351,2,cfg)
        saved=torch.load(old['model']['checkpoint'],map_location='cpu',weights_only=False);parent.load_state_dict(saved['actor'],strict=True)
        parent_bundle=GoalNetworkBundle(parent,cfg,saved['log_std'])
        parameters=[learned.bundle.sampling_parameters(),parent_bundle.sampling_parameters()]
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('GPU evaluation required')
        env,_=load_environment_contact_only(cfg,declaration,'terminal_contract_v2')
        origins={};paths={};bases={};goals={};path_records={}
        for ctx,c in old['contexts'].items():
            with Path(c['initial']).open('rb') as f:bank=pickle.load(f).state
            origin=jax.tree.map(lambda x:j.asarray(x[0]),bank);origins[ctx]=origin
            with np.load(c['path']) as z:paths['training_geometry',ctx]=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
            local=copy.deepcopy(cfg);local['path']['turn_radius_m']=1.25
            _,pose,_=env.physics.observe(origin.physical.data)
            paths['transfer',ctx]=build_path(ctx+'90_R2',np.asarray(pose),local)
            for suite in ('transfer','training_geometry'):
                path=paths[suite,ctx]
                goals[suite,ctx]=np.array([np.interp(float(path.goal),np.asarray(path.s),np.asarray(path.xy)[:,i]) for i in (0,1)])
                base=(env.reset(origin.physical,path,j.float32(1.7 if ctx=='left' else 1.6),j.float32(0.)) if suite=='transfer' else origin)
                if int(base.tick)!=0 or bool(base.fault|base.physical.failed|base.domain_exit):raise ValueError('invalid complete task0')
                bases[suite,ctx]=base
                path_file=root/f'path_{suite}_{ctx}.npz'
                np.savez_compressed(path_file,s=np.asarray(path.s),xy=np.asarray(path.xy),heading=np.asarray(path.heading),curvature=np.asarray(path.curvature),turn_end=np.asarray(path.turn_end),goal=np.asarray(path.goal))
                path_records[suite+'/'+ctx]=dict(file=path_file.name,sha256=digest(path_file),original_goal_xy=goals[suite,ctx].tolist())
        atomic_json(root/'path_manifest.json',path_records)
        cases=p['cases'];stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
        env.set_log_template(bases['transfer','left'],paths['transfer','left'],j.float32(1.7))
        one=make_goal_evaluation_step(env,make_goal_collection_evaluator(cfg))
        @jax.jit
        def chunk(carry,param,path,speed,pulse,role,goal):
            return jax.lax.scan(lambda st,_:jax.vmap(one)(st,param,path,speed,pulse,role,goal),carry,None,length=50)
        index=[];started=time.monotonic()
        for offset in range(0,len(cases),16):
            selected=cases[offset:offset+16]
            if len({np.shape(paths[c['suite'],c['context']].s) for c in selected})!=1:raise ValueError('same-shape path batch required')
            bpath=stack([paths[c['suite'],c['context']] for c in selected])
            initial=stack([GoalEvaluationState(bases[c['suite'],c['context']].replace(alpha=j.float32(c['controller_alpha'])),initial_gate(),j.int32(0),j.bool_(False)) for c in selected])
            par=stack([parameters[0 if c['role']==0 else 1] for c in selected])
            speed=j.asarray([c['speed'] for c in selected],j.float32);role=j.asarray([c['role'] for c in selected],j.int32)
            goal=j.asarray([goals[c['suite'],c['context']] for c in selected],j.float32)
            pulse=[]
            for c in selected:
                direction=c['world_direction']
                tick=int(bases[c['suite'],c['context']].physical.physical_tick)
                pulse.append([(tick+c['pulse_onset_task_tick'])*.005,(tick+c['pulse_onset_task_tick'])*.005+.1,*[c['force_N']*d for d in direction]])
            args=(par,bpath,speed,j.asarray(pulse,j.float32),role,goal)
            directory=root/f'batch_{offset//16:02d}';directory.mkdir()
            with (directory/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(initial),f)
            carry=initial;logs=[]
            for block in range(10):
                update(stage='collecting',case_offset=offset,block=block,inflight_slots=3200,inflight_actual='unknown')
                carry,data=chunk(carry,*args);jax.block_until_ready(carry);host,data=jax.device_get((carry,data));data=flatten_ticks(data)
                update(stage='saving',active_lower_ticks=state['active_lower_ticks']+int(data['active_tick'].sum()),allocated_lower_slots=state['allocated_lower_slots']+int(data['active_tick'].size),inflight_slots=0,inflight_actual='known')
                np.savez_compressed(directory/f'chunk_{block:02d}.npz',**data);logs.append(data)
                if np.any(host.state.fault) or np.any(data['active_tick']&(data['nonfinite']|data['lower_fault']|data['policy_fault'])):raise ValueError('evaluation engineering fault, preserve partials')
            merged={k:np.concatenate([x[k] for x in logs]) for k in logs[0]}
            for i,c in enumerate(selected):
                file=root/f'case_{offset+i:03d}.npz';np.savez_compressed(file,**{k:v[:,i] for k,v in merged.items()})
                index.append(dict(case=c,file=file.name,sha256=digest(file),original_goal_xy=goals[c['suite'],c['context']].tolist()))
            atomic_json(root/'case_index.json',index);update(completed_cases=len(index),elapsed_s=time.monotonic()-started)
        atomic_json(root/'physics_receipt.json',dict(plan_sha256=digest(root/'plan.json'),case_index_sha256=digest(root/'case_index.json'),
            path_manifest_sha256=digest(root/'path_manifest.json'),active_lower_ticks=state['active_lower_ticks'],allocated_lower_slots=state['allocated_lower_slots']))
        update(phase='completed',stage='physics_complete',analysis_status='not_started')
    except BaseException as exc:
        (root/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise
