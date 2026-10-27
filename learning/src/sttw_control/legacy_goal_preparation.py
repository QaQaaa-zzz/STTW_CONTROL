"""Prepare a fixed legacy-controller goal-continuation comparison, without physics.

Copies byte-authenticated archived Python modules into an isolated import tree.
Does not import the archived environment, deserialize states, or start evaluation.
"""
import copy
import hashlib
import json
import os
from pathlib import Path


ROLES = ('parent', 'task_only', 'physical_credit', 'floor', 'zero_middle')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def archived_sources(root):
    """Select by exact original basename AND hash; suffix matches are ambiguous."""
    plan=json.loads((root/'plan.json').read_text())
    selected={}
    for original,expected in plan['source_sha256'].items():
        matches=[p for p in (root/'source_snapshot').iterdir()
                 if p.name.split('_',1)[-1]==Path(original).name and digest(p)==expected]
        if len(matches)!=1:
            raise ValueError('one archived source matching name and launch hash required: '+original)
        selected[original]=matches[0]
    return plan,selected


def selected_cases(route, inventory):
    rows=[]
    for x in inventory:
        if x['role'] not in ROLES:
            continue
        if x['profile']!=0:
            raise ValueError('no unrequested support/ESO/expert profiles')
        floor=1.5 if x['role']=='floor' else 1.
        if x['reference_speed_floor_m_s']!=floor or x['zero_mode']!=(x['role']=='zero_middle'):
            raise ValueError('preserve original floor and zero definitions')
        if x['role']=='zero_middle':
            if x['alpha'] is not None or x['state_alpha']!=0.:
                raise ValueError('original zero has no user-alpha label and state alpha0')
        elif x['alpha']!=x['state_alpha']:
            raise ValueError('original user alpha and prepared-state alpha must agree')
        r=copy.deepcopy(x)
        r.update(source_case_id=x['case_id'],case_id=route+'/'+x['case_id'],
                 route_group=route,horizon_s=20.,stage='goal_continuation',
                 planning='common_once_per_episode',reentry=True)
        rows.append(r)
    expected={(role,alpha,event) for role in ROLES for alpha in
              ([None] if role=='zero_middle' else [.5,.75]) for event in
              ('zero','t380_F-32','t380_F+32','t580_F-32','t580_F+32')}
    actual={(r['role'],r['alpha'],r['event_id']) for r in rows}
    if len(rows)!=45 or actual!=expected or len({r['case_id'] for r in rows})!=45:
        raise ValueError('exact original45 unique role/alpha/event conditions required')
    return rows


def prepare(worktree, output):
    worktree=Path(worktree).resolve();output=Path(output).resolve()
    mount=Path('/media/qy/6A7E3DB07E3D75C3')
    if not os.path.ismount(mount) or not output.is_relative_to(mount):
        raise ValueError('mounted HDD required for runtime and future evaluation outputs')
    stat=os.statvfs(worktree)
    if stat.f_bavail*stat.f_frsize<50*1024**3:
        raise ValueError('system volume must retain50GiB')
    if output.exists():
        raise FileExistsError('never overwrite an existing preparation')
    original=worktree/'runs/hidden_time_reference_confirmation_v1'
    shared=worktree/'runs/reentry_shared_permission_v1'
    oldmodels=json.loads((original/'left/authenticated_models.json').read_text())
    sources={};routes={};all_cases=[];runtime_map={}
    for route in ('left','right'):
        root=original/route
        p=json.loads((root/'plan.json').read_text())
        status=json.loads((root/'timing/status.json').read_text())
        if status['phase']!='completed' or status['completed_cases']!=55:
            raise ValueError('completed historical55 required, never restart it')
        for f,sha in p['sealed_files'].items():
            if digest(f)!=sha:raise ValueError('historical input drift: '+f)
            sources[f]=sha
        for f,sha in [(p['source_initial_state'],p['source_initial_state_sha256']),
                      (p['path_file'],p['path_sha256'])]:
            if digest(f)!=sha:raise ValueError('initial/path drift: '+f)
            sources[f]=sha
        models=json.loads((root/'authenticated_models.json').read_text())
        for role in ('parent','task_only','physical_credit'):
            if models['models'][role]['actor_sha256']!=oldmodels['models'][role]['actor_sha256']:
                raise ValueError('same fixed model required across routes')
            for field in ('actor','checkpoint'):
                if field in models['models'][role]:
                    f=models['models'][role][field];sha=models['models'][role][field+'_sha256']
                    if digest(f)!=sha:raise ValueError('frozen model drift')
                    sources[f]=sha
        config=json.loads((root/'common_environment_config.json').read_text())
        if config['residual_interface']['speed_reference_bounds_m_s']!=[1.,3.5]:
            raise ValueError('legacy permissions required')
        rows=selected_cases(route,json.loads((root/'inventory.json').read_text())['cases'])
        # Use the already-tested original16-block reentry kernel, not modern
        # feedback/goal-preview/short-support kernels or a fresh reconstruction.
        archived_plan,archived=archived_sources(shared/route)
        this_map={Path(f).name:(f,archive,digest(archive)) for f,archive in archived.items()
                  if Path(f).parent.name=='sttw_control'}
        text=this_map['anticipatory_reference.py'][1].read_text()
        if 'allow_reentry=False' not in text or 'independent_support' in text:
            raise ValueError('expected frozen simple reentry runtime, no later permission patch')
        if route=='left':runtime_map=this_map
        elif {k:v[2] for k,v in this_map.items()}!={k:v[2] for k,v in runtime_map.items()}:
            raise ValueError('same archived control code required on both routes')
        sources[str(root/'plan.json')]=digest(root/'plan.json')
        sources[str(shared/route/'plan.json')]=digest(shared/route/'plan.json')
        routes[route]=dict(initial=p['source_initial_state'],initial_sha256=p['source_initial_state_sha256'],
            path=p['path_file'],path_sha256=p['path_sha256'],config=str(root/'common_environment_config.json'),
            declaration=str(root/'source_declaration.json'),models=str(root/'authenticated_models.json'),
            runtime_source=str(shared/route),case_ids=[r['case_id'] for r in rows])
        all_cases.extend(rows)
    output.mkdir(parents=True)
    destination=output/'runtime/learning/src/sttw_control';destination.mkdir(parents=True)
    expected={}
    for name,(original_path,archived,sha) in runtime_map.items():
        dest=destination/name;dest.write_bytes(archived.read_bytes())
        if digest(dest)!=sha:raise ValueError('runtime copy mismatch')
        expected[str(dest)]=sha
    write(output/'inventory.json',all_cases)
    write(output/'runtime_manifest.json',dict(copied_modules=len(expected),files=expected,
        original_to_archive={f:str(p) for f,p,_ in runtime_map.values()},physics_steps=0,
        scope='Only archived sttw_control modules copied; fresh subprocess must use this import root and audit actual imports.'))
    plan=dict(project='STTW_CONTROL',experiment_id='STTW-LEGACY-CREDIT-GOAL-CONTINUATION-V1',owner='/root',
        question='Does old physical-credit64 retain joint goal/survival uses beyond matched ordinary64 and simple controls with common20s causal replanning?',
        hypothesis='Some old task gains may survive reasonable task continuation; alternatives are deadline artifacts or coverage by common geometry/zero/floor.',
        routes=routes,roles=list(ROLES),unique_cases=90,logical_cases_with_zero_reuse=100,
        initialization='Exact original complete task0 bank slots, physical700; no reset/reconstructed physics or alpha override',
        runtime_root=str(destination.parent),runtime_expected_sources=expected,
        model_role='Frozen parent/task_only64/physical_credit64 from original online_preference_survival_v1_attempt3; weights only inference, no optimizer',
        horizon_s=20.,lower_dt_s=.005,upper_dt_s=.02,steps_per_chunk=50,chunks=20,
        maximum_active_lower_ticks=360000,maximum_allocated_lower_slots=360000,optimizer_updates=0,
        original_permission='Speed correction[-1,+.25]m/s, steer±.2rad; original filters/slew; speed[1,3.5], floor role alone lower bound1.5; native ECBC/ESO/local300 unchanged',
        common_changes=['20s contact/fault window','existing16-complete-valid-block Actor handback','one causal Dubins replan at ordinary1s boundaries'],
        excluded_changes=['modern .1rad geometric feedback','short speed support','goal-preview clamp','new physical-credit training','new force/timing/geometry sweep'],
        planning=dict(period_s=1.,radius_m=1.8,departure_m=1.,maximum_per_case=1,information='Own current geometry only, no event schedule',
            geometry_binding='Only projection/follower; keep all histories/ESO/filters/offsets/clocks/domain telemetry and clean counts. Path support must match original array count.'),
        scoring=dict(primary='Original worldgoal1m first entry before contact/fault; whole20s survival separately',
            sensitivity_radii_m=[.5,.75],secondary='10s prefix, first-arrival time, minimum distance, original/executed geometry, original speed and reference changes',
            canonical_Jalpha='NOT_ADOPTED; pending user definition, no scalar optimizer or preference superiority from these binary endpoints'),
        data_role='Already-seen legacy DEV route/initial/event families, one training pair; not independent TEST, training repeats or causal isolated effect of replanning',
        stop='Exactly90 unique trajectories once. Contact/fault/20s; UNKNOWN saves raw/cost and stops. All-pass or simple-control coverage closes this comparison; no force/time/path/threshold/training expansion.',
        outcome_decisions=dict(covered='Retain assets, close credit necessity for this contract',
            goal_only_gain='Retain task-continuation use, no anti-fall claim',
            survival_goal_tradeoff='Report all gains/losses and distances, no automatic default switch',
            engineering_unknown='Preserve partial evidence, no negative capability label'),
        preparation_status='protocol_and_runtime_prepared',execution_ready=False,
        unimplemented=['legacy goal runner/20s host replan orchestration','actual carry/pulse/path-shape certification','real GPU zero-integration shape','final prelaunch source seal'],
        input_sha256=sources,preparation_script_sha256=digest(__file__))
    write(output/'plan.json',plan)
    write(output/'seal.json',dict(plan_sha256=digest(output/'plan.json'),inventory_sha256=digest(output/'inventory.json'),runtime_manifest_sha256=digest(output/'runtime_manifest.json')))
    write(output/'status.json',dict(phase='prepared_protocol_only',execution_ready=False,physics_steps=0,optimizer_updates=0,unique_cases=90))
    return plan
