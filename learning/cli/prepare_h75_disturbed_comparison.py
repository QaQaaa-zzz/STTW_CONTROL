"""Fill only the missing H75 endpoint disturbances; reuse all matched controls."""
import argparse
import ast
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

from sttw_control.preference_best import atomic_json, digest


def prepare(root, source, nominal):
    from sttw_control.middle_lineage_comparison import authenticate_historical_models
    from prepare_short_recovery_campaign import register
    root = Path(root).resolve()
    source = Path(source).resolve()
    nominal = Path(nominal).resolve()
    old = json.loads((source/'plan.json').read_text())
    hist = json.loads((nominal/'plan.json').read_text())
    if old['experiment'] != 'entry_history_confirmation_v1':
        raise ValueError('original entry-history confirmation required')
    for directory, plan in [(source, old), (nominal, hist)]:
        if json.loads((directory/'status.json').read_text())['phase'] != 'completed':
            raise ValueError('complete source physics required')
        if digest(directory/'plan.json') != json.loads((directory/'prelaunch_receipt.json').read_text())['plan_sha256']:
            raise ValueError('source plan changed')
        for file, sha in plan['sealed_files'].items():
            if digest(file) != sha:
                raise ValueError('source input changed: '+file)
    if old['contexts'] != hist['contexts'] or digest(source/'common_environment_config.json') != digest(nominal/'common_environment_config.json'):
        raise ValueError('source contexts or shared control configuration differ')
    module = Path('learning/src/sttw_control').resolve()
    for filename in ['anticipatory_reference.py', 'live_path_continuation.py', 'path_command_env.py',
                     'path_reference.py', 'path_command_reward.py', 'closed_loop_kernel.py', 'teleop_env.py',
                     'actuator.py', 'controller.py', 'direct_command_policy.py', 'path_command_policy.py',
                     'reference_reserve_gate.py', 'reference_task_feedback.py', 'reference_reentry.py',
                     'geometric_path.py', 'middle_channel_ablation.py', 'causal_reference_baseline.py']:
        if digest(module/filename) != old['source_sha256'][str(module/filename)]:
            raise ValueError('source numeric runtime changed: '+filename)
    for filename, names in [('finite_value_evaluation.py', ['make_evaluation_step', 'rebind_path_context']),
                            ('goal_preview_baseline.py', ['goal_follower', 'GoalPreviewEnv', 'make_goal_step'])]:
        def tree(file, name):
            return next(ast.dump(n, include_attributes=False) for n in ast.parse(file.read_text()).body
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
        for name in names:
            if tree(module/filename, name) != tree(source/'source_at_launch/learning/src/sttw_control'/filename, name):
                raise ValueError('changed control kernel: '+name)
    models = hist['models']
    authenticate_historical_models(models)
    original = json.loads((source/'inventory.json').read_text())['cases']
    template = [c for c in original if c['role'] == 'goal_full_feedback' and c['alpha'] in [0., 1.] and c['force_N']]
    if len(template) != 96 or len(old['contexts']) != 12:
        raise ValueError('exact twelve-context 96 disturbed endpoint events required')
    cases = []
    for c in template:
        n = copy.deepcopy(c)
        n.update(role='lineage_h75', actor_role='lineage_h75', model_key='FULLTAIL_PPO75_U77',
                 lineage_family='lineage_h75', case_id=c['case_id'].rsplit('/', 1)[0]+'/lineage_h75')
        assert {k:v for k,v in n.items() if k not in ['role','actor_role','model_key','lineage_family','case_id']} == {k:v for k,v in c.items() if k not in ['role','actor_role','model_key','lineage_family','case_id']}
        cases.append(n)
    cache = dict(declared=[], records=[], indices=[], source_runs=[])
    for directory, allowed in [(source, {'goal_full_feedback', 'goal_zero_feedback'}), (nominal, {'lineage_h75'})]:
        inv = json.loads((directory/'inventory.json').read_text())['cases']
        selected = [c for c in inv if c['role'] in allowed and c['alpha'] in [0., 1.]]
        ids = {c['case_id'] for c in selected}
        records = [r for r in json.loads((directory/'physical/results.json').read_text()) if r['case_id'] in ids]
        indices = [r for r in json.loads((directory/'physical/case_index.json').read_text()) if r['case_id'] in ids]
        if len(selected) != len(records) or len(ids) != len(indices):
            raise ValueError('incomplete exact cache')
        byid = {r['case_id']:r for r in records}
        for c in selected:
            if any(byid[c['case_id']][k] != v for k,v in c.items()):
                raise ValueError('cached execution metadata differs')
        for item in indices:
            file = (directory/item['file']).resolve()
            if digest(file) != item['sha256']:
                raise ValueError('cached raw changed')
            item['file'] = str(file)
        for key, values in [('declared',selected),('records',records),('indices',indices)]:
            cache[key] += values
        cache['source_runs'].append(dict(run=str(directory),plan_sha256=digest(directory/'plan.json'),cases=len(selected)))
    if len(cache['declared']) != 264 or len({c['case_id'] for c in cases+cache['declared']}) != 360:
        raise ValueError('exact96 new +264 unique cache required')
    cache.update(active_lower_ticks=sum(r['physical_ticks'] for r in cache['records']), allocated_lower_slots=264*2000)
    root.mkdir(exist_ok=False)
    atomic_json(root/'prepare_status.json', dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW H75受扰强对照准备',root/'prepare_status.json')
    for name in ['common_environment_config.json','source_declaration.json']:
        shutil.copyfile(source/name, root/name)
    protocol = dict(question='Does frozen H75 with the SAME control stack cover the current low-speed recovery gain?',
                    primary_change='Only frozen middle weights; H75 uses existing full-feedback/reference permissions.',
                    new_cases=96,cached_cases=264,analysis_cases=360,alpha=[0,1],forces_N=[-32,-16,16,32],
                    duration_s=.1,horizon_s=10,goal_radius_m=.5,additional_radius_sensitivity_m=[.75,1.],
                    outcomes=['original-world target arrival before contact/fault','full-window physical survival','original-alpha task/reference cost','paired gain/loss','minimum target distance'],
                    source_role='goal_full_feedback', no_controller_force_preview=True,
                    adoption='No automatic model replacement; failures allowed. Determine complementarity or old-baseline coverage.',
                    stop='Exactly96 missing events; error preserves raw and stops. No training, threshold tuning, extra alpha, expanded forces or retries.')
    for name, value in [('protocol.json',protocol),('inventory.json',dict(cases=cases)),('cache_receipt.json',cache),('authenticated_models.json',models)]:
        atomic_json(root/name,value)
    files = list(module.glob('*.py'))+[Path(__file__).resolve(),Path('learning/cli/evaluate_middle_lineage.py').resolve()]
    sources = {str(f):digest(f) for f in files}
    for file in files:
        dest=root/'source_at_launch'/file.relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dest)
    sealed = {str(root/name):digest(root/name) for name in ['common_environment_config.json','source_declaration.json','protocol.json','inventory.json','cache_receipt.json','authenticated_models.json']}
    sealed.update({c[k]:c[k+'_sha256'] for c in old['contexts'].values() for k in ['initial','path']})
    plan = dict(copy.deepcopy(old), experiment='h75_disturbed_common_interface_v1',lineage_comparison=True,
                models=models,evaluation_roles=['goal_zero_feedback','goal_full_feedback','lineage_h75'],
                declared_cases=96,new_cases=96,cached_cases=264,analysis_cases=360,
                maximum_active_lower_ticks=192000,maximum_allocated_lower_slots=192000,
                cached_active_lower_ticks=cache['active_lower_ticks'],cached_allocated_lower_slots=528000,
                endpoint_receipt=str(root/'authenticated_models.json'),endpoint_receipt_sha256=digest(root/'authenticated_models.json'),
                source_reference_run=str(source),source_evaluation_plan_sha256=digest(source/'plan.json'),
                sealed_files=sealed,source_sha256=sources,code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                stop=protocol['stop'],data_role='Frozen H75 vs current/zero under shared expanded interface and existing feedback; related DEV, not native reproduction/independent repeat/TEST. No new mechanism.')
    atomic_json(root/'plan.json',plan)
    atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=96,completed_cases=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
    for name,title,path in [('ACTIVE_RUN.json','STTW H75受扰96对照',root/'status.json'),('ACTIVE_SHAPE.json','STTW H75受扰96 shape',root/'shape/status.json')]:
        register(root/name,title,path)
    atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,authenticated_cached_cases=264))
    print(json.dumps(dict(prepared=str(root),new=96,cache=264,physics_steps=0)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=Path('runs/entry_history_confirmation_v1'))
    parser.add_argument('--nominal',type=Path,default=Path('runs/lineage_nominal_common_interface_v1'))
    args=parser.parse_args();prepare(args.output,args.source,args.nominal)
