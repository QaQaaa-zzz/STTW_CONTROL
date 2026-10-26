"""Frozen-history weight transfer under one explicitly shared current interface."""
import ast
import copy
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .preference_best import atomic_json, digest

LINEAGE_ROLES = ('lineage_dual', 'lineage_u325', 'lineage_h75')
IDS = ('DUAL_COVERAGE_U200_A0','DUAL_COVERAGE_U200_A1','DISTILLED_V3_U325','FULLTAIL_PPO75_U77')


def authenticate_historical_models(models):
    """Narrow frozen-registry authentication, independent of current64 loader."""
    import jax
    import torch
    from .lineage_inference import embed_legacy_actor
    from .path_command_policy import make_torch_path_network, export_torch_actor
    registry=json.loads(Path('learning/configs/frozen_middle_registry.json').read_text())
    root=Path(registry['root'])
    if digest(root/'manifest.json')!=registry['manifest_sha256']:
        raise ValueError('frozen lineage manifest changed')
    manifest=json.loads((root/'manifest.json').read_text())
    if set(models)!=set(IDS):raise ValueError('exact declared four frozen model IDs required')
    result={}
    for key,model in models.items():
        mid='STTW_MIDDLE_'+key; record=registry['records'][mid]
        for field in ['actor','checkpoint']:
            if model[field]!=record[field] or model[field+'_sha256']!=record[field+'_sha256'] or digest(model[field])!=record[field+'_sha256']:
                raise ValueError('historical model differs from frozen registry')
        node=next(n for n in manifest['nodes'] if n['id']==mid)
        config_file=root/node['resolved_config']['path']
        if digest(config_file)!=node['resolved_config']['sha256']:raise ValueError('native config changed')
        cfg=json.loads(config_file.read_text()); saved=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
        if saved['schema']!='sttw_geometric_path_actor351_v1' or saved['update']!=node['update'] or saved['identity']['lower_actor_sha256']!=manifest['common_frozen_lower']['actor']['sha256']:
            raise ValueError('historical checkpoint schema/update/lower mismatch')
        actor=make_torch_path_network(351,2,cfg)
        actor.load_state_dict({k[6:]:v for k,v in saved['policy'].items() if k.startswith('actor.')},strict=True)
        with Path(model['actor']).open('rb') as f:params=pickle.load(f)
        expected=export_torch_actor(actor,cfg)
        if jax.tree.structure(params)!=jax.tree.structure(expected):raise ValueError('historical Actor structure mismatch')
        for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(expected)):
            np.testing.assert_array_equal(a,b)
        result[key]=embed_legacy_actor(params) if cfg['network'].get('architecture','legacy')=='legacy' else params
    return result


def inventory(original):
    templates=[c for c in original if c['role']=='goal_full_feedback' and c['force_N']==0 and c['alpha'] in [0.,1.]]
    if len(templates)!=24:raise ValueError('exact24 matched nominal endpoint templates required')
    cases=[]
    for c in templates:
        for role in LINEAGE_ROLES:
            key=(IDS[0] if c['alpha']==0 else IDS[1]) if role=='lineage_dual' else IDS[2] if role=='lineage_u325' else IDS[3]
            new=copy.deepcopy(c); new.update(role=role,actor_role=role,model_key=key,lineage_family=role,
                case_id=c['case_id'].rsplit('/',1)[0]+'/'+role)
            cases.append(new)
    if len({c['case_id'] for c in cases})!=72:raise ValueError('unique72 historical cases required')
    return cases


def prepare(root,source):
    from .finite_value_evaluation_run import authenticate_models
    from prepare_short_recovery_campaign import register
    root=Path(root).resolve();source=Path(source).resolve()
    old=json.loads((source/'plan.json').read_text())
    declaration=source/'analysis/lineage_interface_audit_v2/next_comparison_declaration.json'
    contract=json.loads(declaration.read_text())
    if old['experiment']!='entry_history_confirmation_v1' or contract['planned_new_episodes']!=72:
        raise ValueError('original entry-history and sealed planned72 required')
    for name in ['status.json','analysis/status.json']:
        if json.loads((source/name).read_text())['phase']!='completed':raise ValueError('completed source required')
    if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:
        raise ValueError('source plan changed')
    for file,sha in old['sealed_files'].items():
        if digest(file)!=sha:raise ValueError('source input changed')
    module=Path(__file__).parent
    numeric=['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','path_reference.py',
        'path_command_reward.py','closed_loop_kernel.py','teleop_env.py','actuator.py','controller.py',
        'direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py','reference_task_feedback.py',
        'reference_reentry.py','geometric_path.py','middle_channel_ablation.py','causal_reference_baseline.py']
    for name in numeric:
        if digest(module/name)!=old['source_sha256'][str((module/name).resolve())]:raise ValueError('numeric runtime changed: '+name)
    for filename,names in [('finite_value_evaluation.py',['make_evaluation_step','rebind_path_context']),
                           ('goal_preview_baseline.py',['goal_follower','GoalPreviewEnv','make_goal_step'])]:
        def tree(path,name):
            return next(ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
        for name in names:
            if tree(module/filename,name)!=tree(source/'source_at_launch/learning/src/sttw_control'/filename,name):raise ValueError('numeric kernel changed')
    original=json.loads((source/'inventory.json').read_text())['cases']; cases=inventory(original)
    models={k:dict(contract['historical_models'][k],feedback_gain_rad=.1,source_keep_channels=[True,True]) for k in IDS}
    authenticate_historical_models(models)
    cfg=json.loads((source/'common_environment_config.json').read_text())
    native=copy.deepcopy(old['models']['goal_full_feedback']);native.pop('source_keep_channels')
    authenticate_models(cfg,dict(free_GAE=native))
    cached=[c for c in original if c['force_N']==0 and c['alpha'] in [0.,1.] and c['role'] in ['goal_full_feedback','goal_zero_feedback']]
    ids={c['case_id'] for c in cached}
    records=[r for r in json.loads((source/'physical/results.json').read_text()) if r['case_id'] in ids]
    indices=[r for r in json.loads((source/'physical/case_index.json').read_text()) if r['case_id'] in ids]
    if len(cached)!=48 or len(ids)!=48 or len(records)!=48 or len(indices)!=48:raise ValueError('exact48 cache required')
    byid={r['case_id']:r for r in records}
    for c in cached:
        if any(byid[c['case_id']][k]!=v for k,v in c.items()):raise ValueError('cached execution metadata differs')
    for item in indices:
        file=(source/item['file']).resolve()
        if digest(file)!=item['sha256']:raise ValueError('cached raw changed')
        item['file']=str(file)
    root.mkdir(exist_ok=False);atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW冻结中层迭代链72准备',root/'prepare_status.json')
    for name in ['common_environment_config.json','source_declaration.json']:shutil.copyfile(source/name,root/name)
    shutil.copyfile(declaration,root/'protocol.json')
    cache=dict(declared=cached,records=records,indices=indices,active_lower_ticks=sum(r['physical_ticks'] for r in records),allocated_lower_slots=96000,source_run=str(source),source_plan_sha256=digest(source/'plan.json'))
    atomic_json(root/'cache_receipt.json',cache);atomic_json(root/'inventory.json',dict(cases=cases));atomic_json(root/'authenticated_models.json',models)
    files=list(module.glob('*.py'))+[Path('learning/cli/evaluate_middle_lineage.py')]
    sources={str(f.resolve()):digest(f) for f in files}
    for file in files:
        dest=root/'source_at_launch'/file.resolve().relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
    sealed={str(root/name):digest(root/name) for name in ['common_environment_config.json','source_declaration.json','protocol.json','inventory.json','cache_receipt.json','authenticated_models.json']}
    sealed.update({c[k]:c[k+'_sha256'] for c in old['contexts'].values() for k in ['initial','path']})
    plan=dict(copy.deepcopy(old),experiment='lineage_nominal_common_interface_v1',lineage_comparison=True,
        models=models,evaluation_roles=['goal_zero_feedback','goal_full_feedback',*LINEAGE_ROLES],
        declared_cases=72,new_cases=72,cached_cases=48,analysis_cases=120,
        maximum_active_lower_ticks=144000,maximum_allocated_lower_slots=144000,
        cached_active_lower_ticks=cache['active_lower_ticks'],cached_allocated_lower_slots=96000,
        endpoint_receipt=str(root/'authenticated_models.json'),endpoint_receipt_sha256=digest(root/'authenticated_models.json'),
        source_reference_run=str(source),source_evaluation_plan_sha256=digest(source/'plan.json'),
        sealed_files=sealed,source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        data_role=contract['comparison_kind']+'; related DEV/shared ancestor, no independent repeat or TEST.',
        stop=contract['stop_rule'])
    atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=72,completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
    for name,title,path in [('ACTIVE_RUN.json','STTW冻结迭代链72物理',root/'status.json'),('ACTIVE_SHAPE.json','STTW冻结迭代链shape',root/'shape/status.json')]:register(root/name,title,path)
    atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,cached_trajectories_authenticated=48))
