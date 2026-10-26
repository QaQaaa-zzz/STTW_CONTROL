"""Seal a nominal-only reference optimization opportunity pilot; no implicit run."""
import ast
import copy
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np

from .preference_best import atomic_json, digest

PREFIXES = ('v1.7_yaw-12', 'v2.1_yaw+12')
GEOMETRIES = ('left55_R1p8_v1p1', 'right120_R1p6_v1p4')


def prepare(root, source):
    from .finite_value_evaluation_run import authenticate_models
    root, source = Path(root).resolve(), Path(source).resolve()
    old = json.loads((source/'plan.json').read_text())
    if old['experiment'] != 'entry_history_confirmation_v1':raise ValueError('original complete entry-history source required')
    for name in ['status.json','analysis/status.json']:
        if json.loads((source/name).read_text())['phase'] != 'completed':raise ValueError('completed source required')
    if digest(source/'plan.json') != json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('source plan changed')
    for file, sha in old['sealed_files'].items():
        if digest(file) != sha:raise ValueError('sealed source input changed: '+file)
    module = Path(__file__).parent
    numeric = ['anticipatory_reference.py','live_path_continuation.py','path_command_env.py','path_reference.py',
               'path_command_reward.py','closed_loop_kernel.py','teleop_env.py','actuator.py','controller.py',
               'direct_command_policy.py','path_command_policy.py','reference_reserve_gate.py','reference_task_feedback.py',
               'reference_reentry.py','geometric_path.py','middle_channel_ablation.py']
    for name in numeric:
        if digest(module/name) != old['source_sha256'][str((module/name).resolve())]:raise ValueError('frozen numerical runtime changed: '+name)
    def tree(file, name):
        return next(ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    for file, names in [('finite_value_evaluation.py',['make_evaluation_step','rebind_path_context']),
                        ('goal_preview_baseline.py',['goal_follower','GoalPreviewEnv','make_goal_step'])]:
        for name in names:
            if tree(module/file,name)!=tree(source/'source_at_launch/learning/src/sttw_control'/file,name):raise ValueError('base control AST changed')
    contexts = {prefix+'__'+geometry:copy.deepcopy(old['contexts'][prefix+'__'+geometry]) for prefix in PREFIXES for geometry in GEOMETRIES}
    for c in contexts.values():
        for field in ['initial','path']:
            if digest(c[field]) != c[field+'_sha256']:raise ValueError('complete initial state/path changed')
        prefix=c['geometry_config']['straight_prefix_m']
        with np.load(c['path']) as path:turn_end=float(path['turn_end'])
        c['bias_positions_m']=[prefix-1.5,prefix,(prefix+turn_end)/2,turn_end,turn_end+1.5]
    cfg=json.loads((source/'common_environment_config.json').read_text())
    model=copy.deepcopy(old['models']['goal_full_feedback']);authenticate_models(cfg,{'free_GAE':model})
    searches=[dict(context=key,alpha=a,method=method,seed=826100+i*10+ai,
                   force_N=0.,search_id=f'{key}/a{a:g}/{method}')
              for i,key in enumerate(contexts) for ai,a in enumerate([0.,.5,1.]) for method in ['cem','random']]
    original=json.loads((source/'inventory.json').read_text())['cases']
    events=[c for c in original if c['context'] in contexts and c['role']=='goal_full_feedback']
    if len(searches)!=24 or len(events)!=60:raise ValueError('four contexts/threealpha/two searches/all60 events required')
    root.mkdir(exist_ok=False)
    from prepare_short_recovery_campaign import register
    atomic_json(root/'prepare_status.json',dict(phase='running',pid=os.getpid()))
    register(root/'ACTIVE_PREPARE.json','STTW定向参考优化合同准备',root/'prepare_status.json')
    try:
        for name in ['common_environment_config.json','source_declaration.json']:
            shutil.copyfile(source/name,root/name)
        atomic_json(root/'search_inventory.json',searches)
        atomic_json(root/'evaluation_events.json',events)
        files=list(module.glob('*.py'))+[Path('learning/cli/prepare_reference_bias.py').resolve()]
        sources={str(f.resolve()):digest(f) for f in files}
        for f in files:
            dest=root/'source_at_prepare'/f.resolve().relative_to(Path.cwd());dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest)
        sealed={str(root/name):digest(root/name) for name in ['common_environment_config.json','source_declaration.json','search_inventory.json','evaluation_events.json']}
        sealed.update({model[k]:model[k+'_sha256'] for k in ['actor','checkpoint']})
        sealed.update({c[k]:c[k+'_sha256'] for c in contexts.values() for k in ['initial','path']})
        plan=dict(project='STTW_CONTROL',experiment='reference_bias_opportunity_v1',source=str(source),
            code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),code_identity='Uncommitted new implementation captured by source SHA; HEAD alone does not include these changes.',
            contexts=contexts,model=model,feedback_spec=old['feedback_spec'],horizon_s=10.,lower_dt_s=.005,
            search_kind='Nominal-only, per original task/context/alpha; fixed selected reference process before any event evaluation.',
            bias=dict(physical_scales=[.25,.05],units=['m/s','rad'],internal_knots=3,normalized_dimensions=6,
                      positions='prefix-1.5,prefix,(prefix+turn_end)/2,turn_end,turn_end+1.5',outer_targets=[0.,0.],
                      placement='Add to full frozen native physical target before original fixed feedback/filter/slew/antiwindup; same original total target/reference limits.'),
            search=dict(methods=['cem','random'],population=16,generations=4,elite=4,initial_std=.5,smoothing=.6,std_floor=.1,
                        same_seed_within_task=True,all_candidates_including_incumbents_charged=True,
                        objective='Original user-alpha whole10s scored cost, provided actual no-contact and goal entry<=1m. Infeasible known trajectories +inf; unknown is error, not failed control. Zero incumbent must be genuinely feasible.',
                        initialization='Zero-bias full original middle; no learned optimizer/critic/Actor updates.'),
            searches=24,search_episodes=1536,evaluation_episodes=120,maximum_new_actual_lower_ticks=3312000,
            maximum_allocated_lower_slots=3312000,optimizer_updates=0,
            primary_evaluation='Original goal1m before first contact/fault in10s; separately0.5/0.75m, full-window contact, paired gains/losses, original-alpha cost, path/speed/XY and total search/integration/wall costs.',
            information='Controller/schedule uses current progress and preselected targets only. Search sees only zero-force nominal rollouts. Hidden actual force/onset/direction never conditions selection; physical pulse only enters simulator.',
            baselines='Same-authority matched64 fixed-distribution random optimization, zero-bias original full middle; cached zero/steer/stage endpoints only with exact identities. Stage after-only remains privileged mechanism comparator.',
            baseline_costs='Every new nominal search transition and final event evaluation counted. Source Actor training and cached raw evidence inherited, not free lifetime cost.',
            nominal_retention='Zero-bias incumbent included every population; return globally best feasible nominal cost. Do not imply generalization, high-speed H75 retention or preference monotonicity from this DEV fit.',
            preference_audit='Cross-score each selected three-alpha nominal trajectory under every original user alpha; report original path/speed errors and all event outcomes. Lower fitted own-alpha cost alone is not preference preservation.',
            data_role='Seen shared bank3 ancestor DEV, four context geometries/entry histories; not independent training repeats or final TEST. Per-task offline search is not real-time deployment.',
            question='Can goal-directed bounded additive reference optimization improve the existing frozen full-middle nominal objective, and do fixed nominal-chosen processes retain arrival/noncontact under hidden old events?',
            stop='One fixed representation/two matched search arms/four generations per task; no knot/gain/sensor/alpha/goal/time/force resweep, no event-wise oracle selection, no automatic retry or extension. No useful increment -> close only this fixed representation/budget, not all model/MPC methods.',
            implementation_status='Core adapter/search implemented and CPU checked; full physics objective/production driver/analyzer still NOT IMPLEMENTED. This prepare does not authorize executing a placeholder or generic runner.',
            sealed_files=sealed,source_sha256=sources)
        atomic_json(root/'plan.json',plan);atomic_json(root/'prelaunch_receipt.json',dict(plan_sha256=digest(root/'plan.json')))
        atomic_json(root/'status.json',dict(phase='prepared',production_ready=False,active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        atomic_json(root/'prepare_status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0,checked_unix=time.time()))
    except BaseException as exc:
        import traceback
        (root/'prepare_traceback.txt').write_text(traceback.format_exc());atomic_json(root/'prepare_status.json',dict(phase='error',error=repr(exc),pid=os.getpid()));raise
