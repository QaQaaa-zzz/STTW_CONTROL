"""Authenticated fixed final64 six-role evaluation, separate from training."""
import json
import os
import pickle
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np

from .finite_value_evaluation import endpoint_records, evaluation_inventory, make_evaluation_step
from .preference_best import atomic_json, digest


def authenticate_models(cfg,models):
    # Authenticate executable Actor leaves against the exact recorded checkpoint.
    import torch
    import jax
    from .finite_value_online import FiniteValueLearner
    from .path_command_policy import make_torch_path_network, export_torch_actor
    from .preference_best import config_hash
    for role, model in models.items():
        for key in ['actor','checkpoint']:
            if digest(model[key]) != model[key+'_sha256']:raise ValueError('endpoint artifact changed: '+role)
        saved = torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
        if role in ['free_GAE','finite_GAE','free_MC']:
            if (saved['next_collection'] != 64 or saved['optimizer_steps'] != [248,256,256]
                    or saved['value_contract'] != model['value_contract']):
                raise ValueError('fixed endpoint optimizer/estimator identity mismatch')
            learner = FiniteValueLearner.restore(cfg,saved)
            if learner.fingerprint() != model['fingerprint']:raise ValueError('endpoint fingerprint mismatch')
            actor = learner.actor
        else:
            if saved['schema'] != 'sttw_dual_survival_boundary_v1' or saved['config_sha256'] != config_hash(cfg):
                raise ValueError('parent native source/config mismatch')
            actor = make_torch_path_network(351,2,cfg)
            actor.load_state_dict(saved['actor'],strict=True)
        with Path(model['actor']).open('rb') as stream:deployed = pickle.load(stream)
        expected = export_torch_actor(actor,cfg)
        if jax.tree.structure(deployed) != jax.tree.structure(expected):raise ValueError('Actor export structure mismatch')
        for a,b in zip(jax.tree.leaves(deployed),jax.tree.leaves(expected)):
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))


def prepare(root, *, training):
    training = Path(training).resolve(); root = Path(root).resolve()
    status = json.loads((training/'status.json').read_text())
    if status.get('phase') != 'completed' or status.get('completed_updates') != 64:
        raise ValueError('complete final64 training required; no evaluation created')
    p = json.loads((training/'plan.json').read_text())
    sha = digest(training/'plan.json')
    if sha != json.loads((training/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('training declaration changed')
    pair_file = training/'checkpoint_pair_0064.json'
    pair = json.loads(pair_file.read_text())
    models = endpoint_records(p,status,pair,plan_sha256=sha)
    for file, expected in p['files_sha256'].items():
        if digest(training/file) != expected:raise ValueError('sealed training input changed: '+file)
    cfg = json.loads((training/'effective_config.json').read_text())
    cases = evaluation_inventory(json.loads((training/'evaluation_cases.json').read_text()))
    authenticate_models(cfg,models)
    root.mkdir(parents=True,exist_ok=False)
    from prepare_short_recovery_campaign import register
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW_CONTROL 价值三臂固定终点300评价准备',root/'prepare_status.json')
    try:
        for old,new in [('effective_config.json','common_environment_config.json'),('source_declaration.json','source_declaration.json')]:
            shutil.copyfile(training/old,root/new)
        atomic_json(root/'inventory.json',dict(cases=cases))
        atomic_json(root/'authenticated_models.json',models)
        files=list(Path('learning/src/sttw_control').glob('*.py'))+[Path('learning/cli/evaluate_finite_value.py'),
            Path('docs/evidence/tolerance_progress_20261022/finite_value_online/EVALUATION_PROTOCOL.md')]
        sources={}
        for file in files:
            sources[str(file.resolve())]=digest(file)
            dest=root/'source_at_launch'/file;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
        plan=dict(project='STTW_CONTROL',experiment='finite_value_online_v1_fixed64_evaluation',
            training=str(training),training_plan_sha256=sha,endpoint_receipt=str(pair_file),
            endpoint_receipt_sha256=digest(pair_file),models=models,contexts=p['contexts'],
            declared_cases=300,new_cases=300,cached_cases=0,maximum_active_lower_ticks=600000,
            maximum_allocated_lower_slots=600000,optimizer_updates=0,horizon_s=10.,lower_dt_s=.005,
            seed=p['evaluation_seed'],data_role='DEV; two old route contexts; one matched optimization repeat',
            sealed_files={str(root/name):digest(root/name) for name in ['inventory.json','authenticated_models.json','common_environment_config.json','source_declaration.json']},
            source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            information='Actual pulse only enters physics; causal parent observes current measured roll rate before advance. Original alpha retained.',
            stop='300 fixed cases, no best selection, optimizer, threshold changes, automatic retry or budget extension')
        atomic_json(root/'plan.json',plan)
        atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=300,completed_cases=0,
                    active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        register(root/'ACTIVE_RUN.json','STTW_CONTROL 偏好抗扰价值三臂固定终点300评价',root/'status.json')
        register(root/'ACTIVE_SHAPE.json','STTW_CONTROL 固定终点300路零积分shape',root/'shape/status.json')
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),checked_unix=time.time()))
    except BaseException as exc:
        (root/'prepare_traceback.txt').write_text(traceback.format_exc())
        atomic_json(root/'prepare_status.json',dict(phase='error',pid=os.getpid(),error=repr(exc),checked_unix=time.time()))
        raise


def execute(root, *, shape_only=False):
    root=Path(root).resolve();previous=json.loads((root/'status.json').read_text())
    if previous['phase']!='prepared':raise ValueError('fresh prepared evaluation required; no implicit restart')
    out=root/('shape' if shape_only else 'physical');out.mkdir(exist_ok=False)
    state=dict(previous,phase='running',stage='authenticating',pid=os.getpid())
    status_file=out/'status.json' if shape_only else root/'status.json'
    def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(status_file,state)
    update()
    try:
        import jax
        import jax.numpy as j
        from .runtime import configure_compilation_cache
        from .contact_only_environment import load_environment_contact_only
        from .path_command_policy import make_path_actor
        from .geometric_path import Path as GeometricPath
        from .causal_reference_baseline import initial_gate
        from .causal_recovery_relay import CausalRelayState,initial_gate as initial_relay
        from .anticipatory_run import flatten_ticks,physical_outcome
        p=json.loads((root/'plan.json').read_text())
        if digest(root/'plan.json')!=json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('evaluation plan changed')
        for file,sha in {**p['sealed_files'],**p['source_sha256'],p['endpoint_receipt']:p['endpoint_receipt_sha256']}.items():
            if digest(file)!=sha:raise ValueError('sealed evaluation input changed: '+file)
        cases=json.loads((root/'inventory.json').read_text())['cases']
        if len(cases)!=300:raise ValueError('complete300 sealed cases required')
        cfg=json.loads((root/'common_environment_config.json').read_text());decl=json.loads((root/'source_declaration.json').read_text())
        params={}
        for role,model in p['models'].items():
            for key in ['actor','checkpoint']:
                if digest(model[key])!=model[key+'_sha256']:raise ValueError('frozen endpoint changed')
            with Path(model['actor']).open('rb') as stream:params[role]=jax.tree.map(j.asarray,pickle.load(stream))
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env,_=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2')
        origins={};paths={}
        for context,c in p['contexts'].items():
            if digest(c['initial'])!=c['initial_sha256'] or digest(c['path'])!=c['path_sha256']:raise ValueError('complete starting context changed')
            with Path(c['initial']).open('rb') as stream:bank=pickle.load(stream).state
            original=jax.tree.map(lambda x:j.asarray(x[0]),bank)
            if int(original.tick)!=0 or int(original.physical.physical_tick)!=700 or bool(original.physical.failed|original.fault|original.domain_exit):raise ValueError('original complete task0/physical700 required')
            origins[context]=original
            with np.load(c['path'],allow_pickle=False) as z:paths[context]=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
        stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
        states=[origins[c['context']].replace(alpha=j.float32(c['alpha'])) for c in cases]
        carry=CausalRelayState(stack(states),stack([initial_gate() for _ in cases]),j.zeros(300,j.int32),stack([initial_relay() for _ in cases]))
        batch_paths=stack([paths[c['context']] for c in cases]);parameters=stack([params[c['role']] for c in cases])
        speeds=j.asarray([c['requested_speed_m_s'] for c in cases],j.float32)
        modes=j.asarray([c['mode_index'] for c in cases],j.int32)
        pulses=j.asarray([[(700+c['pulse_onset_task_tick'])*.005,(700+c['pulse_onset_task_tick'])*.005+c['pulse_duration_s'],
                           *[c['force_N']*d for d in c['world_direction']]] for c in cases],j.float32)
        env.set_log_template(states[0],paths[cases[0]['context']],speeds[0])
        one=make_evaluation_step(env,make_path_actor(cfg),horizon_s=10.)
        @jax.jit
        def chunk(carry,parameters,batch_paths,speeds,pulses,modes):
            return jax.lax.scan(lambda st,_:jax.vmap(one)(st,parameters,batch_paths,speeds,pulses,modes),carry,None,length=50)
        if shape_only:
            result=jax.eval_shape(chunk,carry,parameters,batch_paths,speeds,pulses,modes)
            if result[1]['active_tick'].shape!=(50,300,4):raise ValueError('complete300 lower layout mismatch')
            atomic_json(out/'receipt.json',dict(lower_shape=[50,300,4],physics_steps=0,optimizer_updates=0))
            update(phase='completed',stage='shape_complete');return
        if json.loads((root/'shape/status.json').read_text())['phase']!='completed':raise ValueError('shape must complete first')
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('actual GPU required')
        with (out/'initial_carry.pkl').open('wb') as stream:pickle.dump(jax.device_get(carry),stream)
        logs=[];start=time.monotonic()
        for block in range(10):
            carry,data=chunk(carry,parameters,batch_paths,speeds,pulses,modes);jax.block_until_ready(carry);host,data=jax.device_get((carry,data));data=flatten_ticks(data);logs.append(data)
            np.savez_compressed(out/f'chunk_{block:02d}.npz',**data)
            with (out/'final_carry.pkl').open('wb') as stream:pickle.dump(host,stream)
            state['active_lower_ticks']+=int(data['active_tick'].sum());state['allocated_lower_slots']+=60000
            if state['active_lower_ticks']>p['maximum_active_lower_ticks'] or state['allocated_lower_slots']>p['maximum_allocated_lower_slots']:raise ValueError('declared evaluation budget exceeded')
            update(stage='collecting',completed_blocks=block+1,elapsed_s=time.monotonic()-start)
            if np.any(host.state.fault):raise RuntimeError('engineering fault; raw retained, no implicit retry')
            if np.all(host.state.physical.failed|(host.state.tick>=2000)):break
        data={k:np.concatenate([x[k] for x in logs]) for k in logs[0]};(out/'cases').mkdir()
        results=[];index=[]
        for lane,c in enumerate(cases):
            raw={k:v[:,lane] for k,v in data.items()};file=out/'cases'/f'case_{lane:03d}.npz';np.savez_compressed(file,**raw)
            outcome=physical_outcome(raw['active_tick'],raw['contact_failure'],raw['nonfinite']|raw['lower_fault']|raw['policy_fault'],2000)
            pose=np.asarray(env.physics.helpers.pose(states[lane].physical.data))
            results.append(dict(c,case_index=lane,physical_outcome=outcome,physical_ticks=int(raw['active_tick'].sum())))
            index.append(dict(case_id=c['case_id'],file=str(file.relative_to(root)),sha256=digest(file),initial_xy=pose[:2].tolist(),initial_yaw=float(pose[2])))
        atomic_json(out/'results.json',results);atomic_json(out/'case_index.json',index)
        if any(r['physical_outcome']=='unknown' for r in results):raise ValueError('unknown endpoints preserved; evaluation incomplete')
        update(phase='completed',stage='physical_complete_analysis_pending',completed_cases=300)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise
