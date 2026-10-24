"""Frozen DEV baseline, with the detector running from complete task origin."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import ast

from sttw_control.preference_best import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numerical_branches(source):
    tree=ast.parse(source)
    definitions={n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                 if isinstance(n,ast.FunctionDef) and n.name in ('normal','fallback','one')}
    definitions['scan']=[ast.dump(n,include_attributes=False) for n in ast.walk(tree)
                         if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='execute' for t in n.targets)]
    return hashlib.sha256(json.dumps(definitions,sort_keys=True).encode()).hexdigest()


def prepare_curve(root, reference):
    import numpy as np
    from prepare_short_recovery_campaign import register
    from sttw_control.reference_confirmation import supplemental_force_cases
    from sttw_control.balance_probe import classify_contact_balance
    from sttw_control.recovery_margin_scan import outcome_labels
    worktree=Path(__file__).resolve().parents[2]
    prior=json.loads((reference/'design.json').read_text())
    status=json.loads((reference/'status.json').read_text())
    analysis=json.loads((reference/'analysis/analysis.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=24:
        raise ValueError('requires the complete frozen 24-case causal panel')
    if sum(not p['baseline_safe'] and p['causal_safe'] for p in analysis['pairs']) <= sum(p['baseline_safe'] and not p['causal_safe'] for p in analysis['pairs']):
        raise ValueError('no net physical gain; do not start the conditional supplement')
    metadata_paths=[reference/'design.json',reference/'status.json',reference/'analysis/analysis.json']
    raw_checks=0
    for plan_path,sha in prior['plan_sha256'].items():
        if digest(plan_path)!=sha:raise ValueError('reference plan changed')
        directory=Path(plan_path).parent
        old_plan=json.loads(Path(plan_path).read_text())
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest['plan']!=old_plan:raise ValueError('reference manifest and pinned plan differ')
        metadata_paths += [Path(plan_path),directory/'manifest.json',directory/'results.json']
        for row in json.loads((directory/'results.json').read_text())['cases']:
            raw=directory/(row['name']+'.npz')
            if digest(raw)!=analysis['raw_sha256'][str(raw)]:raise ValueError('reference raw changed')
            with np.load(raw) as archive:data={k:archive[k] for k in archive.files}
            physical=classify_contact_balance(data,horizon_s=10.)
            for key in ('balance_label','active_ticks','event_contact','full_observation'):
                if physical[key]!=row[key]:raise ValueError('reference physical label/count differs from raw: '+key)
            np.testing.assert_allclose(data['applied_force_impulse'].sum(axis=0,dtype=np.float64),row['impulse_Ns'],atol=1e-8)
            path=directory/(row['route']+'_path.npz')
            metadata_paths.append(path)
            with np.load(path) as path_data:goal=float(path_data['goal'])
            labels=outcome_labels(data,speed=row['speed'],goal_s=goal,working_roll=manifest['config']['stage_B_baseline']['working_roll_rad'],horizon=10.,hold_s=manifest['config']['stage_B_baseline']['hold_s'])
            if labels['task_recovered']!=row['task_recovered'] or row['joint_success']!=bool(labels['task_recovered'] and physical['balance_label']=='NO_FLOOR_CONTACT'):
                raise ValueError('reference task label differs from raw')
            raw_checks+=1
    reference_metadata_sha256={str(p):digest(p) for p in metadata_paths}
    for path,sha in analysis['raw_sha256'].items():
        if digest(path)!=sha:raise ValueError('reference raw identity changed')
    allowed={'learning/cli/evaluate_causal_reference_full_task.py','learning/src/sttw_control/reference_confirmation.py'}
    for path,sha in prior['source_sha256'].items():
        if digest(path)!=sha and str(Path(path).relative_to(worktree)) not in allowed:
            raise ValueError('numerical source changed since reference: '+path)
    archived=reference/'source_at_launch/learning/cli/evaluate_causal_reference_full_task.py'
    core=numerical_branches(Path(__file__).read_text())
    if numerical_branches(archived.read_text())!=core:raise ValueError('numerical scan/control branches changed')
    old_helper=ast.parse((reference/'source_at_launch/learning/src/sttw_control/reference_confirmation.py').read_text())
    new_helper=ast.parse((worktree/'learning/src/sttw_control/reference_confirmation.py').read_text())
    extract=lambda tree:ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_path_evidence'),include_attributes=False)
    if extract(old_helper)!=extract(new_helper):raise ValueError('path check helper changed')
    root.mkdir(exist_ok=False)
    sources=list((worktree/'learning/src/sttw_control').rglob('*.py'))+[Path(__file__).resolve()]
    hashes={str(p):digest(p) for p in sources}
    for path in sources:
        dest=root/'source_at_launch'/path.relative_to(worktree);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
        assert digest(dest)==hashes[str(path)]
    plans=[]
    for old_plan in prior['plans']:
        old=json.loads(Path(old_plan).read_text());cases=supplemental_force_cases(old['cases'])
        assert len(cases)==16
        directory=root/f"bank{old['bank_index']}";directory.mkdir()
        plan={**old,'cases':cases,'declared_cases':16,'maximum_new_lower_ticks':32000,'source_sha256':hashes,
              'scope':'fixed ±16/±48 supplement; reference 24 cases authenticated and reused, not rerun'}
        atomic_json(directory/'plan.json',plan);plans.append(str(directory/'plan.json'))
        register(directory/'ACTIVE_RUN.json',f"STTW_CONTROL 因果容扰曲线补测 bank{old['bank_index']}",directory/'status.json')
    design=dict(project='STTW_CONTROL',purpose='Measure tested no-contact sets using fixed 16/32/48N magnitudes; no continuous safety claim.',
                plans=plans,plan_sha256={p:digest(p) for p in plans},source_sha256=hashes,
                declared_cases=32,maximum_new_lower_ticks=64000,training_steps=0,model_fits=0,
                reference_run=str(reference),reference_raw_sha256=analysis['raw_sha256'],reference_cases_reused=24,
                reference_metadata_sha256=reference_metadata_sha256,reference_raw_label_checks=raw_checks,
                numeric_branch_sha256=core,reuse_boundary='Models/configs/physics sources/gate and numerical scan branches unchanged; only preparation/cardinality/report code generalized.',
                stop='Only fixed ±16/±48 supplement. No automatic 64N, threshold/delay/action/bank changes or new fits. Ties at added forces are not failures; report complete set gains/losses.')
    atomic_json(root/'design.json',design)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=32,completed_cases=0,active_lower_ticks=0,training_steps=0,model_fits=0))
    register(root/'ACTIVE_RUN.json','STTW_CONTROL 因果防摔容扰集合：固定力档补测与合并分析',root/'status.json')
    print(root/'design.json')


def prepare(root):
    import numpy as np
    from prepare_short_recovery_campaign import register
    worktree = Path(__file__).resolve().parents[2]
    root.mkdir(exist_ok=False)
    sources = list((worktree / 'learning/src/sttw_control').rglob('*.py')) + [Path(__file__).resolve()]
    hashes = {str(p): digest(p) for p in sources}
    for source in sources:
        destination = root / 'source_at_launch' / source.relative_to(worktree)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        assert digest(destination) == hashes[str(source)]
    plans = []
    old_dirs = {3: worktree / 'runs/reference_prefix_return_feasibility_v1/evaluation',
                0: worktree / 'runs/reference_prefix_return_confirmation_v1_attempt2/bank0/evaluation'}
    for bank, old in old_dirs.items():
        old_plan = json.loads((old / 'plan.json').read_text())
        declaration = Path(old_plan['source_declaration'])
        source_declaration = json.loads(declaration.read_text())
        assert source_declaration['source']['bank_index'] == bank
        old_rows = json.loads((old / 'results.json').read_text())['cases']
        cases = []
        for route, speed in (('left90_R1p1', 1.8), ('right90_R1p1', 1.5)):
            matches = [r for r in old_rows if r['candidate'] == 'continue_original'
                       and r['route'] == route and r['speed'] == speed and r['alpha'] == 0 and r['force_N'] == 32]
            assert len(matches) == 1
            old_row = matches[0]
            raw = old / (old_row['name'] + '.npz')
            with np.load(raw) as data:
                onset_tick = round(float(data['time'][0]) / .005)
                assert abs(onset_tick * .005 - float(data['time'][0])) < 2e-6
            direction = np.asarray(old_row['expected_impulse_Ns']) / (32. * .1)
            np.testing.assert_allclose(np.linalg.norm(direction), 1., atol=1e-6)
            for force in (0., -32., 32.):
                for arm in ('original', 'causal'):
                    cases.append(dict(route=route, speed=speed, alpha=0., force_N=force, arm=arm,
                                      pulse_onset_task_tick=onset_tick, pulse_duration_s=.1,
                                      world_direction=direction.tolist(), capture_snapshot=old_row['snapshot'],
                                      capture_snapshot_sha256=old_row['snapshot_sha256'],
                                      schedule_source_raw=str(raw), schedule_source_raw_sha256=digest(raw)))
        directory = root / f'bank{bank}'
        directory.mkdir()
        plan = dict(project='STTW_CONTROL', cases=cases, declared_cases=12, bank_index=bank,
                    source_declaration=str(declaration), source_declaration_sha256=digest(declaration),
                    path_evidence_directory=old_plan['path_evidence_directory'], path_evidence_atol_m=1e-4,
                    source_sha256=hashes, original_task_deadline_s=10., maximum_new_lower_ticks=24000,
                    scope='DEV full-task causal baseline; no independent test or tolerance estimator')
        atomic_json(directory / 'plan.json', plan)
        plans.append(str(directory / 'plan.json'))
        register(directory / 'ACTIVE_RUN.json', f'STTW_CONTROL 完整任务因果ESO基线 bank{bank}', directory / 'status.json')
    design = dict(project='STTW_CONTROL', purpose='Test net benefit of a strictly causal simple baseline, including zero and opposite directions.',
                  plans=plans, plan_sha256={p: digest(p) for p in plans}, source_sha256=hashes,
                  declared_cases=24, maximum_new_lower_ticks=48000, training_steps=0, model_fits=0,
                  detector_origin='once at complete task origin; never reset at capture/pulse',
                  target_anchor='current offset at detection; differs from known-onset diagnostic',
                  information='Only environment receives fixed pulse schedules/world forces; gate reads existing actor input and current carry.',
                  stop='No parameter/delay/force/bank sweep. If no net causal benefit, stop this gate proposal. '
                       'Positive results establish only a simple DEV baseline, not a RA-L method.')
    atomic_json(root / 'design.json', design)
    atomic_json(root / 'status.json', dict(phase='prepared', declared_cases=24, completed_cases=0,
                                         active_lower_ticks=0, training_steps=0, model_fits=0))
    register(root / 'ACTIVE_RUN.json', 'STTW_CONTROL 完整任务因果ESO基线与全部对照分析', root / 'status.json')
    print(root / 'design.json')


def prepare_transfer(root, reference):
    """Two new turn/speed contexts, original initial bank, no mirrored state."""
    import numpy as np
    from prepare_short_recovery_campaign import register
    worktree = Path(__file__).resolve().parents[2]
    old = json.loads((reference/'bank3/plan.json').read_text())
    if json.loads((reference/'status.json').read_text())['phase'] != 'completed':
        raise ValueError('parent full task evaluation is incomplete')
    prefix_root = worktree/'runs/tight_midturn_capability_v1'
    scan = prefix_root/'scan'
    scan_manifest = json.loads((scan/'manifest.json').read_text())
    scan_rows = json.loads((scan/'results.json').read_text())['cases']
    cases = []
    context_receipts = []
    for side, speed in (('left',1.5),('right',1.8)):
        prefix_file = prefix_root/side/'prefix_results.json'
        prefix_data = json.loads(prefix_file.read_text())
        row = next(x for x in prefix_data['cases'] if x['speed']==speed and x['alpha']==0.)
        if digest(row['snapshot']) != row['snapshot_sha256']:
            raise ValueError('authentic prefix changed')
        if prefix_data['physical_identity'] != scan_manifest['physical_identity']:
            raise ValueError('prefix and force-direction source physical identities differ')
        matches = [(i,x) for i,x in enumerate(scan_rows)
                   if scan_manifest['case_snapshot_paths'][i]==row['snapshot'] and x['force_N']==32.]
        if len(matches)!=1: raise ValueError('ambiguous force direction source')
        i, raw_row = matches[0]
        raw = scan/(raw_row['name']+'.npz')
        direction = np.asarray(raw_row['expected_impulse_Ns'])/(32.*.1)
        np.testing.assert_allclose(np.linalg.norm(direction),1.,atol=1e-6)
        with Path(row['snapshot']).open('rb') as stream: capture=pickle.load(stream)
        onset = round(row['task_time_s']/.005)
        if int(capture.tick)!=onset or int(capture.physical.physical_tick)!=scan_manifest['case_physical_ticks'][i]:
            raise ValueError('prefix task/physical clocks disagree')
        context_receipts.append(dict(prefix=str(prefix_file),sha256=digest(prefix_file),
                                     scan_manifest_sha256=digest(scan/'manifest.json'),
                                     scan_results_sha256=digest(scan/'results.json'),
                                     source_role='environment pulse timing/direction only; old lower-only labels not reused'))
        for force in (0.,-32.,32.):
            for arm in ('original','causal'):
                cases.append(dict(route=row['route'],speed=speed,alpha=0.,force_N=force,arm=arm,
                                  pulse_onset_task_tick=onset,pulse_duration_s=.1,world_direction=direction.tolist(),
                                  capture_snapshot=row['snapshot'],capture_snapshot_sha256=row['snapshot_sha256'],
                                  schedule_source_raw=str(raw),schedule_source_raw_sha256=digest(raw)))
    assert len(cases)==12 and len({(r['route'],r['speed'],r['force_N'],r['arm']) for r in cases})==12
    root.mkdir(exist_ok=False)
    sources=list((worktree/'learning/src/sttw_control').rglob('*.py'))+[Path(__file__).resolve()]
    hashes={str(p):digest(p) for p in sources}
    for source in sources:
        dest=root/'source_at_launch'/source.relative_to(worktree);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        assert digest(dest)==hashes[str(source)]
    directory=root/'bank3';directory.mkdir()
    plan={**old,'cases':cases,'declared_cases':12,'maximum_new_lower_ticks':24000,
          'controller_variant':'reflected_v1','source_sha256':hashes,
          'context_receipts':context_receipts,'scope':'DEV turn-side transfer of fixed two-speed alpha0 rule; no physical reflection or inherited outcome'}
    atomic_json(directory/'plan.json',plan)
    path=str(directory/'plan.json')
    design=dict(project='STTW_CONTROL',purpose='Test fixed sign-structured reference rule on left1.5/right1.8 from genuine complete task origin.',
                plans=[path],plan_sha256={path:digest(path)},source_sha256=hashes,
                declared_cases=12,maximum_new_lower_ticks=24000,training_steps=0,model_fits=0,
                parent_run=str(reference),controller_variant='reflected_v1',
                stop='Only two new contexts × zero/±32 × original/candidate. Keep all failures and zero/opposite losses. No retuning action/channel/threshold/delay or force sweep.')
    atomic_json(root/'design.json',design)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=12,completed_cases=0,active_lower_ticks=0,training_steps=0,model_fits=0))
    register(directory/'ACTIVE_RUN.json','STTW_CONTROL 左右结构迁移12条 bank3',directory/'status.json')
    register(root/'ACTIVE_RUN.json','STTW_CONTROL 固定参考规则左右迁移与全路径分析',root/'status.json')
    print(root/'design.json')


def prepare_following(root, reference):
    """Fresh matched locked/following pair; inherited runs are context only."""
    from prepare_short_recovery_campaign import register
    worktree=Path(__file__).resolve().parents[2]
    status=json.loads((reference/'status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=24:
        raise ValueError('requires the completed original causal full-task panel')
    old=json.loads((reference/'bank3/plan.json').read_text())
    cases=[{**row,'arm':{'original':'locked','causal':'following'}[row['arm']]} for row in old['cases']]
    assert len(cases)==12
    for row in cases:
        for field in ('capture_snapshot','schedule_source_raw'):
            if digest(row[field])!=row[field+'_sha256']:raise ValueError('pulse context source changed')
    root.mkdir(exist_ok=False)
    sources=list((worktree/'learning/src/sttw_control').rglob('*.py'))+[Path(__file__).resolve()]
    hashes={str(p):digest(p) for p in sources}
    for source in sources:
        dest=root/'source_at_launch'/source.relative_to(worktree);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        assert digest(dest)==hashes[str(source)]
    directory=root/'bank3';directory.mkdir()
    plan={**old,'cases':cases,'declared_cases':12,'maximum_new_lower_ticks':24000,
          'controller_variant':'parent_following_v1','comparison_mode':'locked_following',
          'source_sha256':hashes,'scope':'Fresh matched target-composition comparison, alpha0 only; old outcomes not inherited.'}
    atomic_json(directory/'plan.json',plan);path=str(directory/'plan.json')
    design=dict(project='STTW_CONTROL',purpose='Test whether parent-target increments improve on the strong locked-target intervention under the same detector and permissions.',
                plans=[path],plan_sha256={path:digest(path)},source_sha256=hashes,
                declared_cases=12,maximum_new_lower_ticks=24000,training_steps=0,model_fits=0,
                parent_context_run=str(reference),comparison_mode='locked_following',baseline_arm='locked',candidate_arm='following',
                stop='Only bank3 left1.8/right1.5 × zero/±32 × locked/following. Report all gains/losses, nominal/opposite effects, task cost and reference effort. No gain, duration, threshold, alpha, force or context retuning. Judge net no-contact and meaningful task/effort effects, not perfection. If no usable advantage or worse no-contact balance, stop this composition claim.',
                reuse='Old 68 trajectories retained as development evidence; these 12 are all fresh, including 6 matched strong-baseline lanes because the new carry/composition path must be compared fairly.')
    atomic_json(root/'design.json',design)
    atomic_json(root/'status.json',dict(phase='prepared',declared_cases=12,completed_cases=0,active_lower_ticks=0,training_steps=0,model_fits=0))
    register(directory/'ACTIVE_RUN.json','STTW_CONTROL 父目标跟随与锁定强基线12条 bank3',directory/'status.json')
    register(root/'ACTIVE_RUN.json','STTW_CONTROL 同权限目标合成对照与完整分析',root/'status.json')
    print(root/'design.json')


def execute_bank(root):
    import jax
    import jax.numpy as j
    import numpy as np
    from sttw_control.contact_only_environment import load_environment_contact_only
    from sttw_control.path_command_policy import make_path_actor, validate_actor_parameters
    from sttw_control.path_command_reward import configured_cost_keys
    from sttw_control.reference_confirmation import validate_path_evidence
    from sttw_control.geometric_path import build_path
    from sttw_control.balance_probe import make_lower_tick, classify_contact_balance
    from sttw_control.recovery_margin_scan import outcome_labels
    from sttw_control.runtime import configure_compilation_cache
    from sttw_control.causal_reference_baseline import (initial_gate, causal_reference_action,
                                                       validate_causal_contract, validate_full_task_origin,
                                                       case_has_engineering_error)
    plan = json.loads((root / 'plan.json').read_text())
    from sttw_control.reflected_causal_reference import select_reference_rule, initial_reference_state
    variant=plan.get('controller_variant','baseline')
    causal_reference_action = select_reference_rule(variant)
    locked_comparison=plan.get('comparison_mode')=='locked_following'
    if locked_comparison and (variant!='parent_following_v1' or {r['arm'] for r in plan['cases']}!={'locked','following'}):
        raise ValueError('locked/following comparison requires its explicit controller and arm identities')
    state = dict(phase='running', pid=os.getpid(), stage='loading', declared_cases=len(plan['cases']),
                 completed_cases=0, active_lower_ticks=0, training_steps=0, model_fits=0)
    def update(**changes):
        state.update(changes, checked_unix=time.time()); atomic_json(root / 'status.json', state)
    update()
    try:
        for path, sha in plan['source_sha256'].items():
            if digest(path) != sha: raise ValueError('source changed: ' + path)
        if digest(plan['source_declaration']) != plan['source_declaration_sha256']:
            raise ValueError('source declaration changed')
        if (root / 'manifest.json').exists(): raise FileExistsError('execution already started')
        declaration = json.loads(Path(plan['source_declaration']).read_text())
        cfg = declaration['config']
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        env, initial = load_environment_contact_only(cfg, declaration['source'], 'terminal_contract_v2')
        validate_causal_contract(cfg, env.plant_spec['network']['frame_fields'])
        original_actor = declaration['policies']['H75']
        for key in ('actor', 'checkpoint'):
            if digest(original_actor[key]) != original_actor[key + '_sha256']: raise ValueError('H75 identity mismatch')
        with Path(original_actor['actor']).open('rb') as stream:
            parameters = jax.tree.map(j.asarray, pickle.load(stream))
        validate_actor_parameters(parameters, cfg)
        actor = make_path_actor(cfg)
        pose = np.asarray(env.physics.helpers.pose(initial.data))
        paths = {r['name']: build_path(r['builder'], pose, cfg) for r in declaration['routes']}
        path_checks = []
        for name, path in paths.items():
            with np.load(Path(plan['path_evidence_directory']) / (name + '_path.npz')) as data:
                diff = validate_path_evidence(path.xy, data['xy'], atol_m=plan['path_evidence_atol_m'])
            path_checks.append(dict(route=name, max_abs_difference_m=diff))
            np.savez_compressed(root / (name + '_path.npz'), xy=np.asarray(path.xy), s=np.asarray(path.s), goal=np.asarray(path.goal))
        cases = plan['cases']
        stack = lambda values: jax.tree.map(lambda *xs: j.stack(xs), *values)
        # Complete task reset exactly once; no mid-task capture is used as state.
        path_states = [env.reset(initial, paths[r['route']], r['speed'], r['alpha']) for r in cases]
        gate_states = [initial_reference_state(variant) for _ in cases]
        for s, gate in zip(path_states, gate_states):
            validate_full_task_origin(s.tick, s.physical.physical_tick, initial.physical_tick, gate)
            if variant=='parent_following_v1':
                from sttw_control.following_causal_reference import validate_following_origin
                validate_following_origin(gate)
        captures = {}
        for row in cases:
            if digest(row['schedule_source_raw']) != row['schedule_source_raw_sha256'] or digest(row['capture_snapshot']) != row['capture_snapshot_sha256']:
                raise ValueError('environment-only pulse schedule source changed')
            if row['capture_snapshot'] not in captures:
                with Path(row['capture_snapshot']).open('rb') as stream: captures[row['capture_snapshot']] = pickle.load(stream)
            capture = captures[row['capture_snapshot']]
            if int(capture.tick) != row['pulse_onset_task_tick'] or int(capture.physical.physical_tick) != int(initial.physical_tick) + int(capture.tick):
                raise ValueError('pulse clock differs from authentic capture')
            heading = float(env.physics.helpers.pose(capture.physical.data)[2])
            np.testing.assert_allclose(row['world_direction'], [-np.sin(heading), np.cos(heading), 0.], atol=1e-6)
        carry = (stack(path_states), stack(gate_states))
        path_batch = stack([paths[r['route']] for r in cases])
        speeds = j.asarray([r['speed'] for r in cases])
        enabled = j.asarray([r['arm'] == ('following' if locked_comparison else 'causal') for r in cases])
        absolute_start = int(initial.physical_tick) * env.cc.dt
        pulses = j.asarray([[absolute_start + r['pulse_onset_task_tick'] * env.cc.dt,
                             absolute_start + r['pulse_onset_task_tick'] * env.cc.dt + .1,
                             *[r['force_N'] * v for v in r['world_direction']]] for r in cases])
        deadline = round(plan['original_task_deadline_s'] / env.cc.dt)
        if deadline != 2000 or len(cases)*deadline != plan['maximum_new_lower_ticks']:
            raise ValueError('full task timing or budget differs')
        env.set_log_template(path_states[0], paths[cases[0]['route']], speeds[0])
        lower_tick = make_lower_tick(env, initial.governor.current_reference, contact_only=True)
        cost_keys = configured_cost_keys(cfg)
        keep = ['active_tick', 'actual_xy', 'reference_xy', 'path_cross_track', 'path_heading_error',
                'path_progress', 'goal_section_signed_distance', 'actual_forward_speed', 'wheel_speed_proxy',
                'phi', 'phi_dot', 'actual_delta', 'peak_roll', 'physical_failure', 'nonfinite',
                'contact_failure', 'lower_fault', 'policy_fault', 'domain_exit', 'governed', 'nominal',
                'latent_z', 'offsets', 'filtered_offset', 'target_offset', 'applied_force_impulse',
                'scored_tick_reward', 'chi', 'reference_clip_channels', 'correction_rate_clipped']

        def normal(pair, path, speed, pulse, use):
            s, gate = pair
            obs, _, fault = env.observation(s, path, speed, 10.)
            original = actor.apply(parameters, obs)
            # No pulse, onset, route ID or true force is passed to this function.
            chosen, gate_after, receipt = causal_reference_action(original, obs, s.offsets, s.tick, gate, cfg)
            action = j.where(use, chosen, receipt['locked_action']) if locked_comparison else j.where(use, chosen, original)
            end, log = env.policy_step(s.replace(fault=s.fault | fault | receipt['engineering_fault']), action, path, speed, 10., pulse)
            data = {k: log[k] for k in keep}
            data.update(time=log['time'], relative_time=log['time'], upper_active=log['active_tick'],
                        fallback_active=j.zeros(4, dtype=j.bool_), upper_observation=j.broadcast_to(obs, (4, 351)),
                        original_policy_latent=j.broadcast_to(original, (4, 2)), executed_policy_latent=j.broadcast_to(action, (4, 2)),
                        locked_policy_latent=j.broadcast_to(receipt.get('locked_action',chosen),(4,2)),
                        override_active=j.full(4, (use | locked_comparison) & receipt['override_active']),
                        following_active=j.full(4, use & (variant=='parent_following_v1') & receipt['override_active']),
                        gate_signal=j.full(4, receipt['signal']), gate_valid=j.full(4, receipt['valid']),
                        gate_detected=j.full(4, receipt['detected']), gate_fired=j.full(4, gate_after.fired),
                        gate_start_tick=j.full(4, gate_after.start_tick), gate_mask=j.broadcast_to(gate_after.mask, (4, 2)),
                        gate_target_offset=j.broadcast_to(gate_after.target_offset, (4, 2)),
                        gate_target_clipped=j.broadcast_to(receipt['target_clipped_channels'], (4, 2)),
                        gate_engineering_fault=j.full(4, receipt['engineering_fault']),
                        effective_costs=j.stack([log['effective_components'][k] for k in cost_keys], axis=-1))
            for key in ('parent_target_anchor','parent_target_current','following_requested_target','following_realized_target'):
                data[key]=j.broadcast_to(receipt.get(key,j.zeros(2)),(4,2))
            data['following_target_clipped']=j.broadcast_to(receipt.get('following_target_clipped',j.zeros(2,dtype=j.bool_)) & use,(4,2))
            data['nonfinite'] |= log['policy_fault']
            return (end, gate_after), data
        template = jax.eval_shape(normal, (path_states[0], gate_states[0]), paths[cases[0]['route']], speeds[0], pulses[0], enabled[0])[1]
        zero_block = jax.tree.map(lambda x: j.zeros(x.shape, x.dtype), template)
        zero_tick = jax.tree.map(lambda x: j.zeros(x.shape[1:], x.dtype), template)

        def fallback(pair, pulse):
            s, gate = pair
            def step(st, _):
                def active(st):
                    ref = st.physical.governor.current_reference
                    physical, lower, log = lower_tick(st.physical, st.lower, pulse, st.tick * env.cc.dt, ref)
                    end = st.replace(physical=physical, lower=lower, tick=st.tick+1,
                                     fault=st.fault | log['lower_fault'] | log['nonfinite'],
                                     yaw_rate=(physical.yaw_unwrapped-st.physical.yaw_unwrapped)/env.cc.dt)
                    data = dict(zero_tick)
                    for key in ('active_tick', 'actual_xy', 'actual_forward_speed', 'wheel_speed_proxy', 'phi', 'phi_dot',
                                'actual_delta', 'peak_roll', 'physical_failure', 'nonfinite', 'contact_failure', 'lower_fault', 'applied_force_impulse'):
                        data[key] = log[key]
                    for key in ('reference_xy', 'path_cross_track', 'path_heading_error', 'path_progress', 'goal_section_signed_distance',
                                'latent_z', 'target_offset', 'chi', 'scored_tick_reward', 'effective_costs', 'upper_observation',
                                'original_policy_latent', 'executed_policy_latent', 'locked_policy_latent', 'gate_signal',
                                'parent_target_current','following_requested_target','following_realized_target'):
                        data[key] = j.full_like(zero_tick[key], j.nan)
                    data.update(time=st.tick*env.cc.dt, relative_time=st.tick*env.cc.dt, domain_exit=j.bool_(True),
                                fallback_active=j.bool_(True), governed=ref, nominal=st.physical.raw,
                                offsets=st.offsets, filtered_offset=st.filtered, gate_fired=gate.fired,
                                gate_start_tick=gate.start_tick, gate_mask=gate.mask, gate_target_offset=gate.target_offset,
                                parent_target_anchor=getattr(gate,'parent_anchor',j.zeros(2)))
                    return end, data
                stop = st.physical.failed | st.fault | (st.tick >= deadline)
                return jax.lax.cond(stop, lambda st: (st, zero_tick), active, st)
            end, data = jax.lax.scan(step, s, None, length=4)
            return (end, gate), data
        def one(pair, path, speed, pulse, use):
            s, _ = pair
            def active(pair):
                return jax.lax.cond(pair[0].domain_exit, lambda p: fallback(p, pulse),
                                    lambda p: normal(p, path, speed, pulse, use), pair)
            stop = s.physical.failed | s.fault | (s.tick >= deadline)
            return jax.lax.cond(stop, lambda p: (p, zero_block), active, pair)
        # Paths/pulses are arguments, avoiding baking bank-specific coordinates into compilation.
        execute = jax.jit(lambda carry, paths, speeds, pulses, enabled: jax.lax.scan(
            lambda pair, _: jax.vmap(one)(pair, paths, speeds, pulses, enabled), carry, None, length=25))
        atomic_json(root / 'manifest.json', dict(plan=plan, config=cfg, original_actor=original_actor,
                    physical_identity=env.physics.bundle.identity, reward_components=list(cost_keys), path_checks=path_checks,
                    detector_initialization='once at task tick zero; carry never reset at pulse or capture',
                    initial_physical_tick=int(initial.physical_tick), detector_information='existing obs, current offset, original action, task tick only',
                    controller_variant=plan.get('controller_variant','baseline'),
                    environment_only_pulses=np.asarray(pulses).tolist(), fallback='hold last reference; task ineligible and reward NaN'))
        saved = []
        for block in range(20):
            carry, logs = execute(carry, path_batch, speeds, pulses, enabled)
            jax.block_until_ready(carry)
            host = jax.device_get(logs)
            flat = {k: np.swapaxes(v, 1, 2).reshape((100, len(cases)) + v.shape[3:]) for k,v in host.items()}
            saved.append(flat)
            state['active_lower_ticks'] += int(flat['active_tick'].sum())
            if state['active_lower_ticks'] > plan['maximum_new_lower_ticks']: raise RuntimeError('physics budget exceeded')
            update(stage='full_task_scanning', blocks_completed=block+1, total_blocks=20)
        all_data = {k: np.concatenate([b[k] for b in saved], axis=0) for k in saved[0]}
        states, gates = jax.device_get(carry)
        results = []
        for i, row in enumerate(cases):
            full = {k: v[:, i] for k,v in all_data.items()}
            trace = {k:v[full['active_tick']] for k,v in full.items()}
            name = f'case_{i:03d}'
            np.savez_compressed(root / (name+'.npz'), **trace)
            physical = classify_contact_balance(full, horizon_s=10.)
            labels = outcome_labels(trace, speed=row['speed'], goal_s=float(paths[row['route']].goal),
                                    working_roll=cfg['stage_B_baseline']['working_roll_rad'], horizon=10., hold_s=cfg['stage_B_baseline']['hold_s'])
            for key in ('peak_abs_path_m', 'tail_max_path_m'):
                if labels[key] is not None and not np.isfinite(labels[key]): labels[key]=None
            impulse = trace['applied_force_impulse'].sum(axis=0, dtype=np.float64)
            expected = row['force_N'] * .1 * np.asarray(row['world_direction'])
            amount, intended = float(np.linalg.norm(impulse)), float(np.linalg.norm(expected))
            exposure = 'zero' if row['force_N']==0 else 'none' if amount==0 else 'full' if abs(amount-intended)<=.01*intended else 'partial'
            ix = np.arange(0, len(trace['time']), 4)
            hits = ix[trace['gate_detected'][ix] & trace['upper_active'][ix]]
            if len(hits)>1: raise ValueError('single-shot gate fired more than once')
            fault = bool(np.asarray(states.fault)[i])
            if fault: physical['balance_label']='ENGINEERING_ERROR'
            results.append({**row, **labels, **physical, 'name':name, 'final_fault':fault,
                            'joint_success':bool(labels['task_recovered'] and physical['balance_label']=='NO_FLOOR_CONTACT'),
                            'logged_suffix_return_while_upper_active':float(np.nansum(trace['scored_tick_reward'], dtype=np.float64)),
                            'impulse_Ns':impulse.tolist(), 'expected_impulse_Ns':expected.tolist(), 'force_exposure':exposure,
                            'detection_time_s':None if not len(hits) else float(trace['time'][hits[0]]),
                            'detection_signal':None if not len(hits) else float(trace['gate_signal'][hits[0]]),
                            'override_lower_ticks':int(trace['override_active'].sum()),
                            'following_target_clip_lower_ticks':int(np.any(trace['following_target_clipped'],axis=1).sum()),
                            'upper_active_seconds':float(trace['upper_active'].sum()*env.cc.dt),
                            'override_seconds':float(trace['override_active'].sum()*env.cc.dt),
                            'observed_physical_seconds':float(len(trace['time'])*env.cc.dt),
                            'reference_offset_abs_integral_while_override':(env.cc.dt*np.sum(np.abs(trace['governed']-trace['nominal'])*trace['override_active'][:,None],axis=0,dtype=np.float64)).tolist(),
                            'max_reference_rate_while_override':np.max(np.abs(np.diff(trace['governed'],axis=0,prepend=trace['governed'][:1]))/env.cc.dt*trace['override_active'][:,None],axis=0,initial=0).tolist(),
                            'baseline_gate_role':'both arms execute the same gate; selected target locked or parent-relative' if locked_comparison else 'shadow telemetry; only causal arm executes override'})
        atomic_json(root / 'results.json', dict(cases=results))
        errors = sum(case_has_engineering_error(r['final_fault'], r['balance_label']) for r in results)
        update(phase='error' if errors else 'completed', stage='labels_ready', completed_cases=len(results), engineering_error_cases=errors)
        if errors: raise RuntimeError('engineering faults retained separately from physical outcomes')
    except BaseException as exc:
        update(phase='error', error_type=type(exc).__name__, error=str(exc)); raise


def report(root, design):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'analysis';out.mkdir(exist_ok=False)
    pairs, hashes, ticks = [], {}, 0
    baseline_arm=design.get('baseline_arm','original');candidate_arm=design.get('candidate_arm','causal')
    for plan_path in design['plans']:
        directory=Path(plan_path).parent
        rows=json.loads((directory/'results.json').read_text())['cases']
        manifest=json.loads((directory/'manifest.json').read_text())
        columns=len(rows)//4
        assert len(rows)%4==0
        fig, axes=plt.subplots(2,columns,figsize=(6*columns,11),layout='constrained',squeeze=False)
        for p in range(len(rows)//2):
            baseline, causal=rows[2*p:2*p+2]
            assert baseline['arm']==baseline_arm and causal['arm']==candidate_arm
            assert (baseline['route'],baseline['speed'],baseline['force_N'])==(causal['route'],causal['speed'],causal['force_N'])
            pair=dict(bank=directory.name,route=baseline['route'],speed=baseline['speed'],alpha=0.,force_N=baseline['force_N'],
                      baseline_safe=baseline['balance_label']=='NO_FLOOR_CONTACT',causal_safe=causal['balance_label']=='NO_FLOOR_CONTACT',
                      baseline_joint=baseline['joint_success'],causal_joint=causal['joint_success'],
                      detection_time_s=causal['detection_time_s'],override_lower_ticks=causal['override_lower_ticks'],
                      baseline_exposure=baseline['force_exposure'],causal_exposure=causal['force_exposure'])
            ax=axes.flat[p];progress=[]
            detail, aa=plt.subplots(8,1,figsize=(11,17),layout='constrained')
            for row,color in ((baseline,'black'),(causal,'tab:orange')):
                path=directory/(row['name']+'.npz');hashes[str(path)]=digest(path)
                with np.load(path) as z:data={k:z[k] for k in z.files}
                ticks+=len(data['time']);assert len(data['time'])==row['active_ticks']
                assert abs(float(data['time'][0]))<1e-6
                active=data['upper_active'].astype(bool);override=data['override_active'].astype(bool)
                log_ticks=np.rint(data['time']/.005).astype(np.int64)
                expected_override=(active & (row['arm'] in ('causal','locked','following')) & data['gate_fired'].astype(bool)
                                   & data['gate_mask'].astype(bool).any(axis=1)
                                   & (log_ticks>=data['gate_start_tick']) & (log_ticks<data['gate_start_tick']+80))
                np.testing.assert_array_equal(override,expected_override)
                assert not override[~active].any()
                if row['arm']=='original':assert not override.any()
                untouched=active[:,None] & ~(override[:,None] & data['gate_mask'].astype(bool))
                np.testing.assert_array_equal(data['executed_policy_latent'][untouched],data['original_policy_latent'][untouched])
                if design.get('comparison_mode')=='locked_following':
                    np.testing.assert_array_equal(data['following_active'],override & (row['arm']=='following'))
                    c=manifest['config']['residual_interface']
                    def targets(z):
                        a=np.tanh(z)
                        return np.stack([a[:,0]*np.where(a[:,0]>=0,c['speed_positive_scale_m_s'],c['speed_negative_scale_m_s']),a[:,1]*c['steer_scale_rad']],axis=-1)
                    hits=np.flatnonzero(data['gate_detected'] & active)
                    if len(hits):
                        first=int(hits[0]);anchor=targets(data['original_policy_latent'][first:first+1])[0]
                        np.testing.assert_allclose(data['parent_target_anchor'][first:],np.broadcast_to(anchor,data['parent_target_anchor'][first:].shape),atol=1e-6)
                        np.testing.assert_array_equal(data['executed_policy_latent'][first],data['locked_policy_latent'][first])
                    if row['arm']=='locked':
                        np.testing.assert_array_equal(data['executed_policy_latent'][active],data['locked_policy_latent'][active])
                    elif override.any():
                        selected=override[:,None] & data['gate_mask'].astype(bool)
                        parent=targets(data['original_policy_latent'])
                        requested=targets(data['locked_policy_latent'])+(parent-data['parent_target_anchor'])*data['gate_mask']
                        lo=np.array([-c['speed_negative_scale_m_s'],-c['steer_scale_rad']])*(1-1e-6)
                        hi=np.array([c['speed_positive_scale_m_s'],c['steer_scale_rad']])*(1-1e-6)
                        np.testing.assert_allclose(targets(data['executed_policy_latent'])[selected],np.clip(requested,lo,hi)[selected],atol=1e-6)
                assert bool(data['contact_failure'].any())==row['event_contact']
                if override.any():
                    t0=int(data['gate_start_tick'][override][0]);t=data['time'][override]
                    assert t.min()>=t0*.005-2e-6 and t.max()<(t0+80)*.005+2e-6
                label=f"{row['arm']}; {row['balance_label']}; task={row['task_recovered']}"
                domain=np.flatnonzero(data['domain_exit'].astype(bool));cut=int(domain[0]) if len(domain) else len(data['time'])-1
                ax.plot(*data['actual_xy'][:cut+1].T,color=color,label=label)
                if len(domain):
                    ax.plot(*data['actual_xy'][cut:].T,':',color=color);ax.scatter(*data['actual_xy'][cut],color=color,marker='s')
                ax.scatter(*data['actual_xy'][-1],color=color,marker='x' if row['event_contact'] else 'o')
                prog=data['path_progress'][np.isfinite(data['path_progress'])]
                if len(prog):progress.append(float(prog.max()))
                t=data['time']
                for a,key in zip(aa[:5],('actual_forward_speed','path_cross_track','path_heading_error','phi','gate_signal')):a.plot(t,data[key],color=color,label=label)
                aa[5].plot(t,data['governed'][:,0],color=color);aa[5].plot(t,data['governed'][:,1],'--',color=color)
                ix=np.arange(0,len(t),4)
                reward=np.add.reduceat(np.nan_to_num(data['scored_tick_reward']).astype(np.float64),ix)
                valid=np.logical_or.reduceat(np.isfinite(data['scored_tick_reward']),ix)
                costs=-.005*manifest['config']['path_reward']['scale']*np.add.reduceat(np.nan_to_num(data['effective_costs']).astype(np.float64),ix,axis=0)
                reward[~valid]=np.nan;costs[~valid]=np.nan
                np.testing.assert_allclose(np.nansum(reward),row['logged_suffix_return_while_upper_active'],atol=1e-8)
                remainder=reward-costs.sum(axis=1)
                aa[6].plot(t[ix],reward,color=color);aa[7].plot(t[ix],np.where(valid,np.nancumsum(reward),np.nan),color=color)
                np.savez_compressed(out/f'{directory.name}_{row["name"]}_rewards.npz',task_time=t[ix],step_reward=reward,signed_components=costs,failure_remainder=remainder,component_names=manifest['reward_components'])
                cf, ca=plt.subplots(figsize=(12,5),layout='constrained')
                for k,name in enumerate(manifest['reward_components']):ca.plot(t[ix],costs[:,k],label=name)
                ca.plot(t[ix],remainder,'k--',label='failure remainder');ca.set_yscale('symlog',linthresh=.02)
                ca.set(title=label,xlabel='task seconds',ylabel='signed reward /20ms');ca.legend(fontsize=7,ncol=3)
                cf.savefig(out/f'{directory.name}_{row["name"]}_components.png');plt.close(cf)
            with np.load(directory/(baseline['route']+'_path.npz')) as path:
                show=path['s']<=max(progress)+1;ax.plot(*path['xy'][show].T,'--',color='gray',label='original path')
            title=f"{directory.name} {baseline['route']} v{baseline['speed']} F={baseline['force_N']:+g}N"
            ax.set(title=title,xlabel='x /m',ylabel='y /m');ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=6)
            onset=baseline['pulse_onset_task_tick']*.005
            for a,label in zip(aa,('speed /m/s','cross-track /m','heading /rad','roll /rad','normalized ESO innovation','reference speed/steer','reward /20ms','cumulative reward')):
                a.set_ylabel(label);a.axvspan(onset,onset+.1,color='gray',alpha=.1);a.grid(alpha=.2)
                if causal['detection_time_s'] is not None:a.axvline(causal['detection_time_s'],ls=':',color='tab:orange')
            aa[4].axhline(.015,ls='--',color='gray');aa[4].axhline(-.015,ls='--',color='gray')
            aa[0].legend(fontsize=7);aa[-1].set_xlabel('task seconds')
            for a in aa[6:]:a.set_yscale('symlog',linthresh=.1)
            detail.suptitle(title);detail.savefig(out/f'{directory.name}_pair{p}_timeseries.png');plt.close(detail)
            pairs.append(pair)
        fig.savefig(out/f'{directory.name}_all_xy.png');plt.close(fig)
    assert len(pairs)==design['declared_cases']//2 and ticks<=design['maximum_new_lower_ticks']
    atomic_json(out/'analysis.json',dict(pairs=pairs,raw_sha256=hashes,active_lower_ticks=ticks,training_steps=0,model_fits=0))
    xy_links=' · '.join(f'[{Path(p).parent.name}全部XY]({Path(p).parent.name}_all_xy.png)' for p in design['plans'])
    lines=['# 完整任务因果ESO参考规则', '', xy_links, '',
           f"目的：{design['purpose']}。本轮{design['declared_cases']}条真实全任务，声明的全部力档/反向/提前终止保留；原始参考固定。x=触地，方块=首次域退出，圆点=物理终点，点线=held-reference续行。此为固定DEV简单基线，非安全认证。", '',
           f'|bank|路径/速度|力N|{baseline_arm}不触地|{candidate_arm}不触地|{baseline_arm}联合|{candidate_arm}联合|检测秒|候选干预tick|两臂暴露|', '|---|---|---:|---|---|---|---|---|---:|---|']
    for p in pairs:lines.append(f"|{p['bank']}|{p['route']}/{p['speed']}|{p['force_N']}|{p['baseline_safe']}|{p['causal_safe']}|{p['baseline_joint']}|{p['causal_joint']}|{p['detection_time_s']}|{p['override_lower_ticks']}|{p['baseline_exposure']}/{p['causal_exposure']}|")
    if design.get('comparison_mode')=='locked_following':
        lines+=['', '两臂均实际执行因果gate；locked不是原Actor无干预。各臂父输出来自自己的当前状态。参考偏移积分相对各自当前nominal，仅在实际override期间累计，不称实际执行器能量或纯干预因果效应。积分较小可能来自更早退出/触地/较短窗口，必须结合下列时长和物理/任务结局，不单凭积分判优。', '',
                '|bank/路径/力N|臂|物理观察/上层有效/干预秒|跟随目标新增clip ticks|参考偏移绝对积分 [m, rad·s]|窗口内最大参考变化率 [m/s², rad/s]|', '|---|---|---|---:|---|---|']
        for plan_path in design['plans']:
            directory=Path(plan_path).parent
            for row in json.loads((directory/'results.json').read_text())['cases']:
                lines.append(f"|{directory.name}/{row['route']}/{row['force_N']:g}|{row['arm']}|{row['observed_physical_seconds']:.3f}/{row['upper_active_seconds']:.3f}/{row['override_seconds']:.3f}|{row['following_target_clip_lower_ticks']}|{row['reference_offset_abs_integral_while_override']}|{row['max_reference_rate_while_override']}|")
    stop_scope='若本比较无可用优势，停止本次目标合成方式，保留已验证的locked基线。' if design.get('comparison_mode')=='locked_following' else '若无净收益停止本门控，不扫阈值/延迟。'
    lines+=['',f'新增{ticks} lower ticks，0训练/拟合。逐20ms奖励、有符号分项、失败余项、所有XY/检测/参考曲线与原始SHA已保存。',
            f"{len(design['plans'])}个初态bank，共同继承来源而非独立训练重复；阈值/规则在已看DEV上开发，没有独立泛化、连续alpha、噪声鲁棒或安全认证结论。{stop_scope}"]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return ticks


def report_curve(root, design):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'tolerance_curve';out.mkdir(exist_ok=False)
    reference=Path(design['reference_run'])
    for path,sha in design['reference_metadata_sha256'].items():
        if digest(path)!=sha:raise ValueError('reference metadata changed during supplement')
    samples={};hashes={};totals={}
    for campaign in (reference,root):
        campaign_design=json.loads((campaign/'design.json').read_text())
        ticks=0
        for plan in campaign_design['plans']:
            directory=Path(plan).parent
            manifest=json.loads((directory/'manifest.json').read_text())
            bank=directory.name
            original_plan=json.loads((reference/bank/'plan.json').read_text())
            old_manifest=json.loads((reference/bank/'manifest.json').read_text())
            assert manifest['config']==old_manifest['config'] and manifest['original_actor']==old_manifest['original_actor']
            assert manifest['plan']['source_declaration']==original_plan['source_declaration']
            for row in json.loads((directory/'results.json').read_text())['cases']:
                template=next(c for c in original_plan['cases'] if (c['route'],c['speed'],c['alpha'],c['arm'],c['force_N'])==(row['route'],row['speed'],row['alpha'],row['arm'],32.))
                for field in ('pulse_onset_task_tick','pulse_duration_s','world_direction','capture_snapshot','capture_snapshot_sha256','schedule_source_raw','schedule_source_raw_sha256'):
                    if row[field]!=template[field]:raise ValueError('merged pulse context changed: '+field)
                key=(bank,row['route'],row['speed'],row['alpha'],row['force_N'],row['arm'])
                if key in samples:raise ValueError('force case repeated instead of exact reuse')
                raw=directory/(row['name']+'.npz');hashes[str(raw)]=digest(raw)
                if campaign==reference and hashes[str(raw)]!=design['reference_raw_sha256'][str(raw)]:
                    raise ValueError('reference raw changed during supplement')
                samples[key]=row;ticks+=row['active_ticks']
        totals[campaign.name]=ticks
    assert len(samples)==56
    contexts=sorted({key[:4] for key in samples})
    assert len(contexts)==4
    fig,axes=plt.subplots(4,2,figsize=(13,15),layout='constrained')
    curves=[]
    for i,context in enumerate(contexts):
        bank,route,speed,alpha=context
        for column,sign in enumerate((-1,1)):
            ax=axes[i,column]
            for arm,color,marker in [('original','black','x'),('causal','tab:orange','o')]:
                levels=[];labels=[]
                for magnitude in (0.,16.,32.,48.):
                    row=samples[(*context,sign*magnitude,arm)]
                    label=row['balance_label']
                    if label=='NO_FLOOR_CONTACT' and magnitude and row['force_exposure']!='full':
                        label='UNCONFIRMED_FORCE_EXPOSURE'
                    levels.append(magnitude);labels.append(label)
                value=np.array([1. if x=='NO_FLOOR_CONTACT' else 0. if x=='FLOOR_CONTACT' else np.nan for x in labels])
                ax.plot(levels,value,':',color=color,marker=marker,label=arm+' (tested points only)')
                unknown=np.flatnonzero(np.isnan(value))
                if len(unknown):ax.scatter(np.asarray(levels)[unknown],np.full(len(unknown),.5),color=color,marker='D',label='UNKNOWN')
                safe=[f for f,l in zip(levels,labels) if l=='NO_FLOOR_CONTACT']
                contacts=[f for f,l in zip(levels,labels) if l=='FLOOR_CONTACT']
                nonmonotone=any(a<b for a in contacts for b in safe)
                prefix=[]
                for f,l in zip(levels,labels):
                    if l!='NO_FLOOR_CONTACT':break
                    prefix.append(f)
                next_level=None if len(prefix)==len(levels) else levels[len(prefix)]
                curves.append(dict(bank=bank,route=route,speed=speed,alpha=alpha,direction_sign=sign,arm=arm,
                                   tested_magnitudes_N=levels,labels=labels,safe_tested_N=safe,contact_tested_N=contacts,
                                   nonmonotone_on_tested_grid=nonmonotone,prefix_grid_safe_max_N=prefix[-1] if prefix else None,
                                   next_tested_nonpass_N=next_level,upper_side_unidentified=len(prefix)==len(levels),
                                   interpretation='Grid-only prefix, not a continuous safe radius or certified lower bound.'))
            ax.set(title=f'{bank} {route} v{speed} alpha{alpha:g} direction {sign:+d}',xlabel='force magnitude /N',ylabel='observed no-contact')
            ax.set_xticks([0,16,32,48]);ax.set_yticks([0,.5,1],['contact','UNKNOWN','no contact']);ax.set_ylim(-.1,1.1);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Fixed closed-loop tested force sets; .1s pulses, task0–10s. Dotted guides do not certify intermediate forces.')
    fig.savefig(out/'tested_force_sets.png');plt.close(fig)
    paired=[]
    for context in contexts:
        for force in (-48.,-32.,-16.,0.,16.,32.,48.):
            b=samples[(*context,force,'original')];c=samples[(*context,force,'causal')]
            bs=b['balance_label']=='NO_FLOOR_CONTACT' and (force==0 or b['force_exposure']=='full')
            cs=c['balance_label']=='NO_FLOOR_CONTACT' and (force==0 or c['force_exposure']=='full')
            paired.append(dict(bank=context[0],route=context[1],speed=context[2],alpha=context[3],force_N=force,
                               physical_gain=b['balance_label']=='FLOOR_CONTACT' and cs,
                               physical_loss=bs and c['balance_label']=='FLOOR_CONTACT',
                               baseline_label=b['balance_label'],causal_label=c['balance_label'],
                               baseline_exposure=b['force_exposure'],causal_exposure=c['force_exposure'],
                               baseline_joint=b['joint_success'],causal_joint=c['joint_success']))
    atomic_json(out/'analysis.json',dict(curves=curves,paired=paired,raw_sha256=hashes,unique_cases=56,new_cases=32,reused_cases=24,
                                        acquisition_lower_ticks=totals,new_lower_ticks=totals[root.name],training_steps=0,model_fits=0))
    lines=['# 固定闭环的方向性已测抗扰集合', '', '[本轮全部XY](../analysis/REPORT.md) · [继承24条XY](%s) · [力档集合图](tested_force_sets.png)'%(reference/'analysis/REPORT.md'), '',
           '32条新完整任务＋24条认证复用，共56条唯一轨迹。零力点为同一轨迹，在正/负方向图中各显示一次，汇总不重复计数。原Actor/冻结因果规则，固定完整初态、推力时钟/世界方向、.1s脉冲和原10s观察。', '',
           '|bank/状态|方向|方法|已测无触地N|已测触地N|网格非单调|', '|---|---:|---|---|---|---|']
    for c in curves:lines.append(f"|{c['bank']}/{c['route']}/v{c['speed']}|{c['direction_sign']}|{c['arm']}|{c['safe_tested_N']}|{c['contact_tested_N']}|{c['nonmonotone_on_tested_grid']}|")
    lines+=['',f"非零配对防摔gain/loss：{sum(p['physical_gain'] for p in paired if p['force_N']!=0)}/{sum(p['physical_loss'] for p in paired if p['force_N']!=0)}。逐例任务代价、暴露和原始标签在analysis.json。", '',
            f"本轮新增{totals[root.name]} lower ticks；继承24条实际成本{totals[reference.name]}，两批共{sum(totals.values())}。旧模型训练、初始库及早期扫描为继承成本，未声称完整生命周期成本。0训练/拟合。", '',
            '这些是有限力档的结果，不证明档位之间连续安全或更大力必失败。最高48N仍通过时上侧未辨识，不自动补64N。不按曲线调整gate/动作/阈值/时机。alpha0两情境、相关bank均为DEV，未证明连续alpha协调、独立泛化或实机结果。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (root/'analysis/REPORT.md').open('a') as f:f.write('\n[合并旧24条的方向性已测抗扰集合](../tolerance_curve/REPORT.md)\n')


def execute(root):
    design=json.loads((root/'design.json').read_text())
    state=dict(phase='running',pid=os.getpid(),declared_cases=design['declared_cases'],completed_cases=0,active_lower_ticks=0,training_steps=0,model_fits=0)
    def update(**changes):
        state.update(changes,checked_unix=time.time());atomic_json(root/'status.json',state)
    update(stage='preflight')
    try:
        for path,sha in {**design['source_sha256'],**design['plan_sha256']}.items():
            if digest(path)!=sha:raise ValueError('source or plan changed: '+path)
        for path,sha in design.get('reference_raw_sha256',{}).items():
            if digest(path)!=sha:raise ValueError('reference raw changed: '+path)
        for path,sha in design.get('reference_metadata_sha256',{}).items():
            if digest(path)!=sha:raise ValueError('reference metadata changed: '+path)
        for plan in design['plans']:
            directory=Path(plan).parent
            with (directory/'execution.log').open('w') as log:
                child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--mode','bank','--output',str(directory)],stdout=log,stderr=subprocess.STDOUT)
                update(stage=directory.name,child_pid=child.pid)
                code=child.wait()
            child_state=json.loads((directory/'status.json').read_text())
            state['active_lower_ticks']+=child_state['active_lower_ticks']
            if code or child_state['phase']!='completed':raise RuntimeError(f'{directory.name} exit {code}')
            update(completed_cases=state['completed_cases']+child_state['completed_cases'])
        update(stage='analysis')
        ticks=report(root,design);assert ticks==state['active_lower_ticks']
        if 'reference_run' in design:
            report_curve(root,design)
        update(phase='completed',stage='reported',report=str(root/'analysis/REPORT.md'))
    except BaseException as exc:
        update(phase='error',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','prepare-curve','prepare-transfer','prepare-following','run','bank'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-run',type=Path)
    args=parser.parse_args()
    if args.mode in ('prepare-curve','prepare-transfer','prepare-following'):
        if args.reference_run is None:parser.error('--reference-run required for preparation')
        {'prepare-curve':prepare_curve,'prepare-transfer':prepare_transfer,'prepare-following':prepare_following}[args.mode](args.output.resolve(),args.reference_run.resolve())
    else:
        {'prepare':prepare,'run':execute,'bank':execute_bank}[args.mode](args.output.resolve())
