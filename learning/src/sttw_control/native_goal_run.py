"""Fixed new-context goal/physical-response evaluation on sealed native code.

Load this orchestration by file in a fresh process. It never trains, silently
restarts, or changes old runs. Physics is available only through execute.
"""
import copy
from functools import partial
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)


def file_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


SELECTION_MODE='middle_selection'
SELECTION_MODELS={'H75':'STTW_MIDDLE_FULLTAIL_PPO75_U77',
                  'W400':'STTW_MIDDLE_WITNESS_S918501_SUP400',
                  'u325':'STTW_MIDDLE_DISTILLED_V3_U325'}


def planning_chunk_steps(frequency_hz,horizon_s):
    """Host geometry updates on exact 50Hz upper boundaries, unchanged physical dt."""
    if not math.isfinite(frequency_hz) or frequency_hz<=0 or frequency_hz>50 or 50%frequency_hz:
        raise ValueError('planning frequency must divide 50Hz')
    blocks=frequency_hz*horizon_s
    if not math.isfinite(blocks) or blocks!=int(blocks):raise ValueError('whole planning intervals required')
    return [int(50/frequency_hz)]*int(blocks)


def middle_selection_inventory(p):
    """Five preferences per neural policy; one physical zero trace per condition."""
    if p['mode']!=SELECTION_MODE or p['models']!=SELECTION_MODELS or p['alphas']!=[0.,.25,.5,.75,1.]:
        raise ValueError('declared three single-network/five-alpha selection required')
    if p['horizon_s']!=10. or p['optimizer_updates']!=0 or p['force_frame']!='worldY':
        raise ValueError('fixed H10/no-training/worldY contract required')
    if len(set(p['bank_indices']))!=len(p['bank_indices']) or not p['bank_indices']:
        raise ValueError('unique nonempty bank indices required')
    if set(p['bank_state_fingerprints'])!={str(b) for b in p['bank_indices']}:raise ValueError('bank identities missing')
    if len({c['route'] for c in p['route_contexts']})!=len(p['route_contexts']):raise ValueError('duplicate routes')
    if {c['family'] for c in p['route_contexts']}!={0,1,2}:raise ValueError('straight/single/S coverage required')
    for c in p['route_contexts']:
        if c['direction'] not in (-1.,1.) or c['radius_m']<=0 or c['speed']<=0 or c['angle_deg']<0:raise ValueError('invalid geometry')
        if (c['family']==0)!=(c['angle_deg']==0) or (c['family']==0 and c.get('goal_distance_m',0)<=0):raise ValueError('explicit straight endpoint required')
    events=[dict(event='zero',force_N=0.,onset_s=p['onset_s'])]+[dict(event=f't{p["onset_s"]:g}_F{f:+g}',force_N=f,onset_s=p['onset_s']) for f in p['forces_N']]
    if len(set(p['forces_N']))!=len(p['forces_N']) or 0 in p['forces_N'] or not 0<=p['onset_s']<p['onset_s']+p['duration_s']<10:raise ValueError('invalid events')
    roles=[(m,a) for m in p['models'] for a in p['alphas']]+[('zero',0.)]
    rows=[]
    for b,c,e,(m,a) in itertools.product(p['bank_indices'],p['route_contexts'],events,roles):
        rows.append(dict(route=c['route'],speed=c['speed'],**e,bank_index=b,initial_fingerprint=p['bank_state_fingerprints'][str(b)],
            path_key=f'bank{b}_{c["route"]}',case_id=f'bank{b}/{c["route"]}/{m}/a{a:g}/{e["event"]}',
            mode=m,mode_index=3 if m=='zero' else list(p['models']).index(m),model_id=None if m=='zero' else p['models'][m],
            alpha=a,query_alpha=a,user_alpha=a,effective_user_alpha=a,duration_s=p['duration_s']))
    if len(rows)!=p['expected_cases'] or p['maximum_lower_ticks']!=2000*len(rows):raise ValueError('selection budget/count mismatch')
    planning_chunk_steps(p['planning_frequency_hz'],p['horizon_s'])
    if p['replan_cadence_s']!=1/p['planning_frequency_hz'] or p['maximum_replans']!=int(p['planning_frequency_hz']*p['horizon_s']):raise ValueError('planning budget mismatch')
    for k,v in dict(planner_points=5001,planner_spacing_m=.02,planner_support_m=100.,replanning_radius_m=1.8,departure_m=1.,goal_radius_m=1.).items():
        if p[k]!=v:raise ValueError('shared planner contract differs '+k)
    return rows


def prepare_middle_selection(source,root,protocol):
    """Reuse the native evaluator, authenticate actual named Actors without PPO assumptions."""
    source=Path(source).resolve();root=Path(root).resolve();protocol=Path(protocol).resolve();storage_guard(root)
    old=json.loads((source/'plan.json').read_text());proposal=json.loads(protocol.read_text());rows=middle_selection_inventory(proposal)
    if digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('source seal drift')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status['completed_cases']!=old['cases']:raise ValueError('complete source required')
    if old.get('mode')!='frozen_transfer_evaluation':raise ValueError('native transfer source required')
    for f,sha in old['native_expected_sources'].items():
        if digest(f)!=sha:raise ValueError('native code changed '+f)
    registry_file=Path('learning/configs/frozen_middle_registry.json').resolve();registry=json.loads(registry_file.read_text());registry_root=Path(registry['root'])
    if digest(registry_root/'manifest.json')!=registry['manifest_sha256']:raise ValueError('registry manifest drift')
    manifest=json.loads((registry_root/'manifest.json').read_text());models={}
    files=[protocol,registry_file,registry_root/'manifest.json',source/'plan.json',source/'seal.json',source/'evaluation/status.json',Path(old['initial_source']['prepared_bank'])]
    for role,mid in proposal['models'].items():
        record=registry['records'][mid];models[role]={k:record[k] for k in ('actor','actor_sha256','checkpoint','checkpoint_sha256','checkpoint_kind')}
        models[role]['registry_id']=mid
        for field in ('actor','checkpoint'):
            f=Path(record[field])
            if digest(f)!=record[field+'_sha256']:raise ValueError('named model drift '+mid)
            files.append(f)
        node=next(n for n in manifest['nodes'] if n['id']==mid);f=registry_root/node['resolved_config']['path']
        if digest(f)!=node['resolved_config']['sha256']:raise ValueError('model config drift')
        files.append(f)
        cfg=json.loads(f.read_text())
        # Training settings can differ; deploying a different physical/reference interface cannot be silent.
        if cfg['residual_interface']!=old['config']['residual_interface'] or cfg['network']!=old['config']['network']:
            raise ValueError('named model native network/reference interface differs '+mid)
    if digest(old['initial_source']['prepared_bank'])!=old['initial_source']['prepared_bank_sha256']:raise ValueError('bank drift')
    folder=Path(__file__).resolve().parent
    modules=[folder/n for n in ('native_goal_run.py','native_goal_continuation.py','native_equilibrium_rule.py','contact_only_environment.py','goal_replanning.py','_vendor/pythonrobotics/dubins.py','_vendor/pythonrobotics/angle.py')]+[Path('learning/cli/run_native_goal_confirmation.py').resolve()]
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    for f in modules:(root/'source'/f.name).write_bytes(f.read_bytes())
    save(root/'inventory.json',rows);save(root/'effective_config.json',old['config']);files += [root/'inventory.json',root/'effective_config.json']
    p=dict(project='STTW_CONTROL',experiment_id=proposal['experiment_id'],mode=SELECTION_MODE,source=str(source),
        native_runtime_root=old['native_runtime_root'],native_expected_sources=old['native_expected_sources'],initial_source=old['initial_source'],
        models=models,config=old['config'],contexts=proposal['route_contexts'],cases=len(rows),horizon_s=10.,goal_radius_m=1.,
        chunks=len(planning_chunk_steps(proposal['planning_frequency_hz'],10.)),chunk_upper_steps=planning_chunk_steps(proposal['planning_frequency_hz'],10.),steps_per_chunk=int(50/proposal['planning_frequency_hz']),maximum_active_lower_ticks=proposal['maximum_lower_ticks'],maximum_allocated_lower_slots=proposal['maximum_lower_ticks'],
        optimizer_updates=0,protocol=proposal,data_role=proposal['data_role'],question=proposal['question'],
        initialization='Same native reset from authenticated prepared bank; no learned coordinator, no parameter training',
        replanning=dict(period_s=proposal['replan_cadence_s'],radius_m=1.8,departure_m=1.,maximum_per_case=proposal['maximum_replans'],continuous_replanning=True,information='Original reference until departure, then repeated causal reconnection; no event information'),
        scoring=proposal['scoring'],stop=proposal['stop_rule'],code_revision=os.popen('git rev-parse HEAD').read().strip(),
        input_sha256={str(f):digest(f) for f in files},source_sha256={str(f):digest(f) for f in modules})
    save(root/'plan.json',p);save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')));save(root/'status.json',dict(phase='prepared',optimizer_updates=0,active_lower_ticks=0,allocated_lower_slots=0))


CAPABILITY_MODE='fixed_h75_capability'


def low_speed_lateral_load_inventory(p):
    """Complete paired world-Y equivalent lateral-load screen; never a wind speed."""
    required=dict(mode=CAPABILITY_MODE,model_id=SELECTION_MODELS['H75'],bank_indices=[0],
        horizon_s=10.,
        optimizer_updates=0,planning_frequency_hz=10.,zero_onset_s=0.,
        planner_points=5001,planner_spacing_m=.02,planner_support_m=100.)
    if any(p.get(k)!=value for k,value in required.items()):
        raise ValueError('paired low-speed lateral-load contract differs')
    if (not p['alphas'] or any(not math.isfinite(a) or not 0<=a<=1 for a in p['alphas'])
        or any(b<=a for a,b in zip(p['alphas'],p['alphas'][1:]))):raise ValueError('finite strictly increasing candidate alpha grid required')
    direction=p.get('force_direction_world',[0.,1.,0.])
    if (len(direction)!=3 or any(not math.isfinite(x) for x in direction)
        or direction[2]!=0. or abs(sum(x*x for x in direction)-1.)>1e-6
        or p['force_frame']!=('world_fixed_vector' if 'force_direction_world' in p else 'worldY')):
        raise ValueError('explicit fixed unit horizontal world force direction required')
    reconnection=p.get('reconnection')
    if reconnection is not None and (set(reconnection)!={'goal_heading','maximum_added_length_m'}
        or reconnection['goal_heading']!='original' or not math.isfinite(reconnection['maximum_added_length_m'])
        or reconnection['maximum_added_length_m']<0):
        raise ValueError('explicit original-tangent reconnection length contract required')
    forces=p['forces_N'];onsets=p['onsets_s'];duration=p['duration_s']
    contexts=p['route_contexts']
    boundary=p.get('range_planner_boundary')
    context_grids=[c.get('boundary_expected_alphas',p['alphas']) if boundary else p['alphas'] for c in contexts]
    if any(not g or any(a not in p['alphas'] for a in g) or len(set(g))!=len(g) for g in context_grids):raise ValueError('boundary execution candidates must be declared grid points')
    excluded=p.get('excluded_completed_cases',[])
    if len({r['case_id'] for r in excluded})!=len(excluded):raise ValueError('unique reused complete cases required')
    count=sum(map(len,context_grids))*(int(p.get('include_nominal',True))+len(forces)*len(onsets))-len(excluded)
    if (bool(forces)!=bool(onsets) or (not forces and not p.get('include_nominal',True)) or len(set(forces))!=len(forces) or len(set(onsets))!=len(onsets)
        or any(not math.isfinite(f) or f==0 for f in forces)
        or not math.isfinite(duration) or duration<=0
        or any(not math.isfinite(t) or t<=0 or t+duration>10 for t in onsets)
        or p['expected_cases']!=count or p['maximum_lower_ticks']!=2000*count):
        raise ValueError('unique finite paired load grid and exact budget required')
    candidate=p.get('candidate_model')
    if candidate is not None and (not candidate.get('registry_id') or candidate['registry_id']==SELECTION_MODELS['H75']):
        raise ValueError('explicit distinct candidate identity required')
    if candidate is not None and set(candidate)!={'registry_id','actor','actor_sha256','checkpoint','checkpoint_sha256','training_manifest','training_manifest_sha256','source_update'}:
        raise ValueError('complete candidate artifact identity required')
    scope=p.get('geometry_scope','single_turn')
    if scope not in ('single_turn','low_speed_track_segments','mixed_speed_track_segments'):
        raise ValueError('unknown low-speed geometry scope')
    speed_max=3.5 if scope=='mixed_speed_track_segments' else 2.2
    angle_max=135. if scope=='mixed_speed_track_segments' else 90.
    if (not contexts or len({c['route'] for c in contexts})!=len(contexts)
        or any((c['family']!=1 or c['angle_deg']!=90.) if scope=='single_turn'
               else (c['family'] not in (1,2) or not 30.<=c['angle_deg']<=angle_max) for c in contexts)
        or any(not 1.5<=c['speed']<=speed_max or c['direction'] not in (-1.,1.)
               or not math.isfinite(c['radius_m']) or c['radius_m']<=0 or c.get('goal_offset_after_turn_m')!=4. for c in contexts)):
        raise ValueError('unique declared geometries within speed/angle scope required')
    events=([dict(event='zero',force_N=0.,onset_s=0.)] if p.get('include_nominal',True) else [])+[
        dict(event=f't{onset:g}_F{force:+g}',force_N=force,onset_s=onset)
        for onset,force in itertools.product(p['onsets_s'],p['forces_N'])]
    identity=candidate['registry_id'] if candidate else p['model_id']
    rows=[]
    combinations=((context,event,alpha) for context,grid in zip(contexts,context_grids) for event,alpha in itertools.product(events,grid))
    for context,event,alpha in combinations:
        route=context['route'];source=context.get('state_source');start=source['source_task_tick']*.005 if source else 0.
        if start+p['duration_s']+max(onsets,default=0.)>10:raise ValueError('conditional force extends beyond task horizon')
        onset=event['onset_s']+start if event['force_N'] else 0.
        event_name=f"prep{event['onset_s']:g}_F{event['force_N']:+g}" if source and event['force_N'] else event['event']
        rows.append(dict(route=route,speed=context['speed'],**event,duration_s=p['duration_s'],force_frame=p['force_frame'],
        **({'force_direction_world':direction} if 'force_direction_world' in p else {}),
        **({'source_task_start_s':start,'preparation_s':event['onset_s'] if event['force_N'] else 0.} if source else {}),
        bank_index=0,path_key=f'bank0_{route}',case_id=f"bank0/{route}/{identity}/a{alpha:g}/{event_name}",
        mode='H75',mode_index=0,model_id=identity,alpha=alpha,query_alpha=alpha,user_alpha=alpha,effective_user_alpha=alpha))
        rows[-1]['onset_s']=onset
        rows[-1]['event']=event_name
        if boundary:rows[-1]['original_user_alpha']=context['state_source']['source_alpha']
        if p.get('causal_preference_return'):
            user=p['causal_preference_return']['user_alpha']
            rows[-1].update(original_user_alpha=user,raw_user_alpha=user,user_alpha=user,effective_user_alpha=user,
                case_id=rows[-1]['case_id']+f'/return_user{user:g}')
    if any('state_source' in c for c in contexts) and not all('state_source' in c for c in contexts):raise ValueError('all conditional contexts require explicit sources')
    excluded_ids={r['case_id'] for r in excluded}
    if not excluded_ids.issubset({r['case_id'] for r in rows}):raise ValueError('reused case outside declared complete query matrix')
    return [r for r in rows if r['case_id'] not in excluded_ids]


def restore_conditional_capability_states(plan,cases,paths):
    """Copy complete declared native carries; change candidate alpha only."""
    import pickle
    import jax
    import jax.numpy as j
    import numpy as np
    contexts={c['route']:c for c in plan['contexts']};cache={};copies=[];sources=[]
    for i,case in enumerate(cases):
        context=contexts[case['route']];source=context['state_source'];root=Path(source['run'])
        if str(root) not in cache:
            parent=json.loads((root/'plan.json').read_text())
            if digest(root/'plan.json')!=source['plan_sha256'] or json.loads((root/'seal.json').read_text())['plan_sha256']!=source['plan_sha256']:raise ValueError('conditional source plan changed')
            if parent['config']!=plan['config'] or parent['initial_source']!=plan['initial_source'] or parent['models']['H75']['actor_sha256']!=plan['models']['H75']['actor_sha256']:raise ValueError('conditional source model/physics/config differs')
            cache[str(root)]=(parent,{})
        parent,loaded=cache[str(root)]
        original=next(c for c in parent['contexts'] if c['route']==source['parent_route'])
        if {k:v for k,v in context.items() if k not in ('route','state_source','boundary_expected_alphas')}!={k:v for k,v in original.items() if k!='route'}:raise ValueError('conditional geometry changed')
        file=Path(source['carry_file'])
        if file not in loaded:
            if digest(file)!=source['carry_sha256'] or file.parent!=root/'evaluation':raise ValueError('conditional complete carry identity differs')
            with file.open('rb') as f:loaded[file]=pickle.load(f)
        state_key=('state',file,source['source_lane'])
        if state_key not in loaded:
            selected=jax.tree.map(lambda x:j.asarray(np.asarray(x)[source['source_lane']]),loaded[file])
            loaded[state_key]=(selected,snapshot_fingerprint(selected))
        carry,fingerprint=loaded[state_key];state=carry.state
        if int(state.tick)!=source['source_task_tick'] or int(state.physical.physical_tick)!=source['source_physical_tick'] or float(state.alpha)!=source['source_alpha'] or bool(state.physical.failed)|bool(state.fault):raise ValueError('conditional clock/alpha/failure differs')
        pathfile=Path(source['path_file'])
        if ('path',pathfile) not in loaded:
            if digest(pathfile)!=source['path_sha256'] or pathfile.parent!=root/'evaluation':raise ValueError('conditional original path identity differs')
            with np.load(pathfile) as z:loaded['path',pathfile]=dict(z)
        for key,value in loaded['path',pathfile].items():np.testing.assert_array_equal(np.asarray(getattr(paths,key))[i],value)
        copied=carry.replace(state=state.replace(alpha=j.asarray(case['alpha'],state.alpha.dtype)))
        aa=jax.tree.leaves(carry.replace(state=state.replace(alpha=j.zeros_like(state.alpha))))
        bb=jax.tree.leaves(copied.replace(state=copied.state.replace(alpha=j.zeros_like(state.alpha))))
        if any(not np.array_equal(a,b,equal_nan=True) for a,b in zip(aa,bb)):raise ValueError('conditional copy modified a non-alpha leaf')
        copies.append(copied);sources.append(dict(route=case['route'],source_lane=source['source_lane'],task_tick=int(state.tick),physical_tick=int(state.physical.physical_tick),source_alpha=float(state.alpha),fingerprint=fingerprint))
    return jax.tree.map(lambda *xs:j.stack(xs),*copies),dict(conditional_complete_state_copy=True,all_nonalpha_source_leaves_exact=True,sources=sources)


def query_boundary_alpha(folder,configuration,observation,latents,yaw,task_time_s,interface,user_alpha,permission):
    """One current-state decision; deliberately accepts no case/event schedule."""
    import numpy as np
    folder=Path(folder)
    prediction=file_module('_sttw_boundary_outcome',folder/'preference_outcome_prediction.py')
    coordination=file_module('_sttw_boundary_coordination',folder/'tolerance_coordination.py')
    file=Path(configuration['parameters'])
    if digest(file)!=configuration['parameters_sha256']:raise ValueError('boundary outcome parameters changed')
    with np.load(file) as z:export=dict(z)
    plan=prediction.plan_alpha_intervals(export,observation,configuration['candidate_alphas'],latents,yaw,task_time_s,
        configuration['hypotheses'],interface,no_fall_threshold=configuration['no_fall_threshold'],
        task_threshold=configuration['task_threshold'],max_path_error_m=configuration['max_path_error_m'],
        interval_function=coordination.feasible_alpha_intervals,original_user_alpha=user_alpha)
    decision=coordination.select_from_alpha_range(plan,user_alpha=user_alpha,previous_alpha=user_alpha,
        publication_permitted=permission)
    return dict(plan=plan,decision=decision,actual_future_event_input=False,
        execution='Selected preparation alpha followed by the bound feedback-return policy.' if len(export['mean'])==43 else 'Single causal decision at restored task boundary, held afterwards; not continuous online replanning.')


def validate_outcome_continuation(trained,proposal,folder):
    rule=proposal.get('causal_preference_return')
    if rule:
        expected=dict(kind='sensor_gated_return_v1',parameters={k:v for k,v in rule.items() if k!='user_alpha'},implementation_sha256=digest(Path(folder)/'tolerance_coordination.py'))
        if trained['schema']!='sttw_conditional_outcome43_return_v1' or trained.get('continuation_policy')!=expected:raise ValueError('return-predictor future policy differs from actual controller')
    elif trained['schema']!='sttw_conditional_outcome42_v1':raise ValueError('returning predictor cannot evaluate a held-alpha controller')


def targeted_base_review_inventory(p):
    """Small predeclared mixed-task DEV review of one explicit trained candidate."""
    required=dict(mode=CAPABILITY_MODE,model_id=SELECTION_MODELS['H75'],bank_indices=[0],
        alphas=[0.,.5,1.],horizon_s=10.,expected_cases=12,maximum_lower_ticks=24000,
        optimizer_updates=0,planning_frequency_hz=10.,forces_N=[0.],duration_s=.15,
        planner_points=5001,planner_spacing_m=.02,planner_support_m=100.)
    if any(p.get(k)!=v for k,v in required.items()) or not p.get('candidate_model'):
        raise ValueError('explicit candidate targeted review contract differs')
    expected=[
        dict(route='left90_R0p9_v1p6',family=1,radius_m=.9,angle_deg=90.,speed=1.6,direction=1.,goal_offset_after_turn_m=4.),
        dict(route='left90_R1p3_v2p0',family=1,radius_m=1.3,angle_deg=90.,speed=2.,direction=1.,goal_offset_after_turn_m=4.),
        dict(route='left70_R1p9',family=1,radius_m=1.9,angle_deg=70.,speed=2.6,direction=1.),
        dict(route='S_right_90_R1p8',family=2,radius_m=1.8,angle_deg=90.,speed=2.8,direction=-1.,goal_distance_m=19.65486717224121)]
    if p['route_contexts']!=expected:raise ValueError('four predeclared targeted geometries required')
    identity=p['candidate_model']['registry_id']
    if identity==SELECTION_MODELS['H75']:raise ValueError('targeted candidate must not impersonate frozen H75')
    return [dict(route=c['route'],speed=c['speed'],event='zero',force_N=0.,onset_s=0.,duration_s=.15,
        bank_index=0,path_key=f"bank0_{c['route']}",case_id=f"bank0/{c['route']}/{identity}/a{alpha:g}/zero",
        mode='H75',mode_index=0,model_id=identity,alpha=alpha,query_alpha=alpha,user_alpha=alpha,effective_user_alpha=alpha)
        for c,alpha in itertools.product(p['route_contexts'],p['alphas'])]


def fixed_h75_capability_inventory(p):
    if p.get('capability_variant')=='low_speed_lateral_load':return low_speed_lateral_load_inventory(p)
    if p.get('capability_variant')=='targeted_base_review':return targeted_base_review_inventory(p)
    required=dict(mode=CAPABILITY_MODE,model_id=SELECTION_MODELS['H75'],bank_indices=[6],
        alphas=[0.,.25,.5,.75,1.],horizon_s=10.,expected_cases=35,maximum_lower_ticks=70000,
        optimizer_updates=0,planning_frequency_hz=10.,forces_N=[0.],duration_s=.15,
        planner_points=5001,planner_spacing_m=.02,planner_support_m=100.)
    variant=p.get('capability_variant')
    if variant=='S_goal_extension':required.update(expected_cases=10,maximum_lower_ticks=20000)
    elif variant=='low_speed_tight_turn':required.update(expected_cases=20,maximum_lower_ticks=40000)
    elif variant is not None:raise ValueError('unknown capability variant')
    if any(p.get(k)!=v for k,v in required.items()):raise ValueError('fixed H75 capability contract differs')
    candidate=p.get('candidate_model')
    if candidate is not None:
        if variant!='low_speed_tight_turn':raise ValueError('candidate override only for declared low-speed capability')
        if p.get('base_model_id',SELECTION_MODELS['H75'])!=SELECTION_MODELS['H75']:raise ValueError('candidate base must remain frozen H75')
        if not candidate.get('registry_id') or candidate['registry_id']==SELECTION_MODELS['H75']:raise ValueError('candidate requires a distinct explicit identity')
    contexts=p['route_contexts']
    if variant=='S_goal_extension':
        if p.get('goal_extension_m')!=3. or len(contexts)!=2 or {c['route'] for c in contexts}!={'S_left_90_R1p8','S_right_90_R1p8'} or any(c['family']!=2 or c.get('goal_distance_m')!=p['goal_source']['paths'][c['route']]['new_goal_s'] for c in contexts):raise ValueError('two S-route goal-only extension required')
    elif variant=='low_speed_tight_turn':
        expected={(v,r) for v,r in itertools.product((1.6,2.),(.9,1.3))}
        if len(contexts)!=4 or len({c['route'] for c in contexts})!=4 or {(c['speed'],c['radius_m']) for c in contexts}!=expected or any(c['family']!=1 or c['angle_deg']!=90. or c['direction']!=1. or c.get('goal_offset_after_turn_m')!=4. for c in contexts):raise ValueError('four declared low-speed left tight turns required')
    elif len(contexts)!=7 or len({c['route'] for c in contexts})!=7 or {c['family'] for c in contexts}!={0,1,2}:raise ValueError('seven original route identities required')
    return [dict(route=c['route'],speed=c['speed'],event='zero',force_N=0.,onset_s=0.,duration_s=.15,
        bank_index=6,path_key=f'bank6_{c["route"]}',case_id=f'bank6/{c["route"]}/H75/a{a:g}/zero',
        mode='H75',mode_index=0,model_id=candidate['registry_id'] if candidate else p['model_id'],alpha=a,query_alpha=a,user_alpha=a,effective_user_alpha=a,
        **({k:p['goal_source']['paths'][c['route']][k] for k in ('old_goal_xy','new_goal_xy')}|{'goal_extension_m':3.} if variant=='S_goal_extension' else {}))
        for c,a in itertools.product(contexts,p['alphas'])]


def validate_goal_extension_source(proposal,initial_source):
    """A moved goal uses the exact saved original curve and physical identity."""
    import numpy as np
    source=proposal['goal_source'];run=Path(source['run']);plan=json.loads((run/'plan.json').read_text())
    if digest(run/'plan.json')!=source['plan_sha256'] or json.loads((run/'seal.json').read_text())['plan_sha256']!=source['plan_sha256'] or plan.get('mode')!=CAPABILITY_MODE or plan['initial_source']!=initial_source:raise ValueError('goal source identity differs')
    for c in proposal['route_contexts']:
        r=source['paths'][c['route']];file=Path(r['file'])
        if file.resolve()!=(run/'evaluation'/f"bank6_{c['route']}_path.npz").resolve() or digest(file)!=r['sha256']:raise ValueError('saved goal path changed')
        with np.load(file) as path:
            old=float(path['goal']);new=old+proposal['goal_extension_m']
            point=lambda q:np.array([np.interp(q,path['s'],path['xy'][:,i]) for i in (0,1)])
            if r['old_goal_s']!=old or r['new_goal_s']!=new or new>path['s'][-1] or not np.allclose(point(old),r['old_goal_xy'],atol=1e-9) or not np.allclose(point(new),r['new_goal_xy'],atol=1e-9):raise ValueError('goal extension metadata differs')
            window=(path['s']>=old)&(path['s']<=new)
            if np.any(path['curvature'][window]!=0):raise ValueError('goal extension must stay on existing exit straight')


def authenticate_capability_candidate(candidate, frozen_model, initial_source, native_config):
    """Validate an explicit mixed-training candidate without importing native runtime."""
    import numpy as np
    import torch
    required={'registry_id','actor','actor_sha256','checkpoint','checkpoint_sha256',
              'training_manifest','training_manifest_sha256','source_update'}
    if set(candidate)!=required or not isinstance(candidate['source_update'],int) or candidate['source_update']<=0:
        raise ValueError('explicit complete candidate identity required')
    for name in ('actor','checkpoint','training_manifest'):
        if digest(candidate[name])!=candidate[name+'_sha256']:raise ValueError('candidate artifact changed: '+name)
    manifest=json.loads(Path(candidate['training_manifest']).read_text());cfg=manifest['config']
    canonical=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    identity=manifest['identity']
    if identity['config_sha256']!=canonical(cfg):raise ValueError('candidate training config identity differs')
    if identity['parent_checkpoint_sha256']!=frozen_model['checkpoint_sha256'] or cfg['finetuning']['parent']['checkpoint_sha256']!=frozen_model['checkpoint_sha256']:
        raise ValueError('candidate is not frozen H75 continuation')
    if manifest['source']!=initial_source or identity['lower_actor_sha256']!=initial_source['candidate']['actor_sha256']:
        raise ValueError('candidate physical bank/model/lower differs')
    for key in ('network','timing','residual_interface','path_reward','follower'):
        if cfg[key]!=native_config[key]:raise ValueError('candidate native interface differs: '+key)
    if cfg['network'].get('architecture')!='alpha_modulated_v1' or cfg['network']['actor_dim']!=351:
        raise ValueError('candidate requires original351 alpha-modulated architecture')
    saved=torch.load(candidate['checkpoint'],map_location='cpu',weights_only=False)
    if saved['update']!=candidate['source_update'] or saved['identity']!=identity:raise ValueError('candidate learner identity/update differs')
    # Same numeric Dense-transpose export as path_command_policy.export_torch_actor,
    # without importing modern sttw_control into the sealed native process.
    weights=saved['policy'];values={}
    for i,index in enumerate((0,2,4,6)):
        values[f'Dense_{i}']={'kernel':weights[f'actor.{index}.weight'].detach().cpu().numpy().T,
                              'bias':weights[f'actor.{index}.bias'].detach().cpu().numpy()}
    for key in ('gain_2','shift_2','gain_3','shift_3'):values[key]=weights['actor.'+key].detach().cpu().numpy()
    expected_actor_keys={f'actor.{index}.{field}' for index in (0,2,4,6) for field in ('weight','bias')}|{'actor.'+key for key in ('gain_2','shift_2','gain_3','shift_3')}
    if {key for key in weights if key.startswith('actor.')}!=expected_actor_keys:raise ValueError('candidate actor parameter structure differs')
    with Path(candidate['actor']).open('rb') as stream:actor=pickle.load(stream)
    with Path(frozen_model['actor']).open('rb') as stream:base=pickle.load(stream)
    def compare(actual,expected):
        if isinstance(expected,dict):
            if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError('candidate exported actor structure differs')
            for key in expected:compare(actual[key],expected[key])
        else:
            a,b=np.asarray(actual),np.asarray(expected)
            if a.shape!=b.shape or not np.isfinite(a).all():raise ValueError('candidate actor shape/nonfinite differs')
            np.testing.assert_array_equal(a,b)
    compare(actor,{'params':values})
    def shape(value):
        if isinstance(value,dict):return {key:shape(item) for key,item in value.items()}
        return np.asarray(value).shape
    if shape(actor)!=shape(base):raise ValueError('candidate incompatible with frozen Actor')
    return dict(candidate,checkpoint_kind='PPO_state',role='explicit_training_candidate_not_frozen_H75',
                base_model_id=SELECTION_MODELS['H75'],verification='checkpoint identity, physical source and exact Torch-to-Actor export; zero physics')


def prepare_fixed_h75_capability(transition,root,protocol):
    """Authenticate a completed fresh-physics bank; no integration or old bank reuse."""
    import jax
    import numpy as np
    transition=Path(transition).resolve();root=Path(root).resolve();protocol=Path(protocol).resolve();storage_guard(root)
    proposal=json.loads(protocol.read_text());rows=fixed_h75_capability_inventory(proposal)
    previous=json.loads((transition/'plan.json').read_text());status=json.loads((transition/'status.json').read_text())
    if digest(transition/'plan.json')!=json.loads((transition/'seal.json').read_text())['plan_sha256'] or previous.get('mode')!='native_physics_transition':raise ValueError('sealed physics transition required')
    if status.get('phase')!='completed' or status.get('actual_preparation_lower_ticks')!=5600 or status.get('old_prepared_state_reused') is not False or previous.get('old_bank_reuse') is not False:raise ValueError('completed fresh eight-state preparation required')
    for file,expected in {**previous['inputs'],**previous['copied_files_sha256']}.items():
        if digest(file)!=expected:raise ValueError('transition identity drift '+file)
    source=json.loads((transition/'initial_source.json').read_text());native=Path(previous['native_runtime_root']).resolve()
    if Path(source['prepared_bank']).resolve()!=transition/'prepared_bank.pkl' or digest(source['prepared_bank'])!=source['prepared_bank_sha256']:raise ValueError('fresh bank identity differs')
    xml=native.parents[1]/'mujoco_ros_ws20260313/model/scalebike_scene_matlab.xml'
    if digest(xml)!=source['model']['source_xml_sha256'] or source['model']==previous['old_model_identity']:raise ValueError('new physical model identity differs')
    if proposal.get('capability_variant')=='S_goal_extension':
        originals={c['route']:c for c in previous['contexts']}
        if any({k:v for k,v in c.items() if k!='goal_distance_m'}!=originals.get(c['route']) for c in proposal['route_contexts']):raise ValueError('original S curve/speed changed')
        validate_goal_extension_source(proposal,source)
    elif proposal.get('capability_variant') in ('low_speed_tight_turn','low_speed_lateral_load','targeted_base_review'):
        if previous['config']['path']['goal_section_offset_after_turn_m']!=4. or previous['config']['residual_interface']['speed_reference_bounds_m_s']!=[1.5,3.5]:raise ValueError('low-speed screen preserves exit goal and frozen speed authority')
    elif previous['contexts']!=proposal['route_contexts']:raise ValueError('original routes differ')
    if previous['models']['H75']['registry_id']!=proposal['model_id']:raise ValueError('frozen H75 identity differs')
    model=previous['models']['H75']
    for key in ('actor','checkpoint'):
        if digest(model[key])!=model[key+'_sha256']:raise ValueError('frozen H75 checkpoint differs')
    if proposal.get('candidate_model'):
        model=authenticate_capability_candidate(proposal['candidate_model'],model,source,previous['config'])
    if any(name=='sttw_control' or name.startswith('sttw_control.') for name in sys.modules):raise ValueError('prepare in fresh process before modern control package import')
    sys.path.insert(0,str(native))
    with Path(source['prepared_bank']).open('rb') as f:bank=pickle.load(f)
    bank_index=proposal['bank_indices'][0]
    if proposal.get('capability_variant') in ('low_speed_lateral_load','targeted_base_review'):source=dict(source,bank_index=bank_index)
    snapshot=jax.tree.map(lambda x:np.asarray(x[bank_index]),bank)
    if bool(snapshot.failed) or int(snapshot.physical_tick)!=700 or any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(snapshot) if np.issubdtype(np.asarray(x).dtype,np.number)):raise ValueError('fresh bank6 clock/failure invalid; no replacement')
    proposal=dict(proposal,bank_state_fingerprints={str(bank_index):snapshot_fingerprint(snapshot)})
    folder=Path(__file__).resolve().parent
    modules=[folder/n for n in ('native_goal_run.py','native_goal_continuation.py','native_equilibrium_rule.py','contact_only_environment.py','goal_replanning.py','_vendor/pythonrobotics/dubins.py','_vendor/pythonrobotics/angle.py')]+[Path('learning/cli/run_native_goal_confirmation.py').resolve()]
    if proposal.get('range_planner_boundary'):
        modules += [folder/'preference_outcome_prediction.py',folder/'tolerance_coordination.py']
    if proposal.get('causal_preference_return'):
        module=file_module('_sttw_return_rule_prepare',folder/'tolerance_coordination.py')
        module.validate_preference_return_rule(proposal['causal_preference_return'])
        if not proposal.get('range_planner_boundary'):modules.append(folder/'tolerance_coordination.py')
    expected_sources=json.loads((transition/'native_expected_sources.json').read_text())
    for file,expected in expected_sources.items():
        if digest(file)!=expected:raise ValueError('new native source drift')
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    for file in modules:(root/'source'/file.name).write_bytes(file.read_bytes())
    save(root/'inventory.json',rows);save(root/'effective_config.json',previous['config'])
    files=[protocol,transition/'plan.json',transition/'seal.json',transition/'status.json',transition/'initial_source.json',transition/'native_expected_sources.json',Path(source['prepared_bank']),root/'inventory.json',root/'effective_config.json']+[Path(model[k]) for k in ('actor','checkpoint')]
    if proposal.get('candidate_model'):files.append(Path(model['training_manifest']))
    for reused in proposal.get('excluded_completed_cases',[]):
        archive=Path(reused['run']);index=json.loads((archive/'evaluation/case_index.json').read_text());status=json.loads((archive/'evaluation/status.json').read_text())
        matches=[r for r in index if r['case_id']==reused['case_id']]
        if status['phase']!='completed' or len(matches)!=1 or digest(matches[0]['raw_file'])!=reused['raw_sha256'] or matches[0]['raw_sha256']!=reused['raw_sha256']:raise ValueError('reused physical case identity differs')
        archived=json.loads((archive/'plan.json').read_text())
        if archived['config']!=previous['config'] or archived['models']['H75']['actor_sha256']!=model['actor_sha256']:raise ValueError('reused physics/model/config differs')
        route=matches[0]['route'];current_context=next(c for c in proposal['route_contexts'] if c['route']==route)
        old_context=next(c for c in archived['contexts'] if c['route']==route)
        clean=lambda c:{k:v for k,v in c.items() if k!='boundary_expected_alphas'}
        if clean(current_context)!=clean(old_context):raise ValueError('reused complete source/geometry differs')
        if digest(archive/'plan.json')!=json.loads((archive/'seal.json').read_text())['plan_sha256']:raise ValueError('reused source seal differs')
        files += [archive/'plan.json',archive/'seal.json',archive/'evaluation/status.json',archive/'evaluation/case_index.json',Path(matches[0]['raw_file'])]
    if proposal.get('range_planner_boundary'):
        boundary=proposal['range_planner_boundary'];manifest=Path(boundary['manifest']);parameters=Path(boundary['parameters'])
        if digest(manifest)!=boundary['manifest_sha256'] or digest(parameters)!=boundary['parameters_sha256']:raise ValueError('boundary predictor identity differs')
        trained=json.loads(manifest.read_text())
        if trained['parameters_sha256']!=boundary['parameters_sha256'] or trained['source_model']['actor_sha256']!=model['actor_sha256']:raise ValueError('boundary predictor middle differs')
        grid=boundary['candidate_alphas']
        if (not {0.,.25,.5,.75,1.}.issubset(set(grid)) or grid!=sorted(set(grid)) or any(not math.isfinite(a) or not 0<=a<=1 for a in grid) or not boundary['hypotheses']):raise ValueError('declared sorted grid retaining all original five settings required')
        validate_outcome_continuation(trained,proposal,folder)
        files += [manifest,parameters]
    if proposal.get('frozen_prediction'):
        frozen=proposal['frozen_prediction'];file=Path(frozen['manifest'])
        if digest(file)!=frozen['manifest_sha256']:raise ValueError('prospective prediction changed before prepare')
        record=json.loads(file.read_text());data=Path(record['query_data'])
        if digest(data)!=record['query_data_sha256'] or [c['case_id'] for c in record['cases']]!=[c['case_id'] for c in rows]:raise ValueError('prospective query identity differs')
        trained_file=Path(record['training'])/'manifest.json'
        if digest(trained_file)!=record['training_manifest_sha256']:raise ValueError('prospective fitted model changed')
        validate_outcome_continuation(json.loads(trained_file.read_text()),proposal,folder)
        parameters_file=trained_file.parent/'parameters.npz'
        if digest(parameters_file)!=record['parameters_sha256']:raise ValueError('prospective fitted weights changed')
        files += [file,data,trained_file,parameters_file]
    for context in proposal['route_contexts']:
        if 'state_source' in context:
            ss=context['state_source'];files += [Path(ss['run'])/'plan.json',Path(ss['run'])/'seal.json',Path(ss['carry_file']),Path(ss['path_file'])]
    files=list(dict.fromkeys(files))
    if proposal.get('capability_variant')=='S_goal_extension':
        files += [Path(proposal['goal_source']['run'])/'plan.json']+[Path(r['file']) for r in proposal['goal_source']['paths'].values()]
    source_ticks={c['state_source']['source_task_tick'] for c in proposal['route_contexts'] if 'state_source' in c}
    if len(source_ticks)>1:raise ValueError('one declared source task clock per batched continuation required')
    remaining_s=10.-next(iter(source_ticks))*.005 if source_ticks else 10.
    steps=planning_chunk_steps(10.,remaining_s)
    p=dict(project='STTW_CONTROL',experiment_id=proposal['experiment_id'],mode=CAPABILITY_MODE,source=str(transition),
        native_runtime_root=str(native),native_expected_sources=expected_sources,initial_source=source,models={'H75':model},
        config=previous['config'],contexts=proposal['route_contexts'],protocol=proposal,cases=len(rows),horizon_s=10.,goal_radius_m=1.,chunks=len(steps),steps_per_chunk=5,chunk_upper_steps=steps,
        maximum_active_lower_ticks=proposal['maximum_lower_ticks'],maximum_allocated_lower_slots=proposal['maximum_lower_ticks'],optimizer_updates=0,
        initialization=("Authenticated full native source carries; only candidate alpha changes, original task clock retained. " if any('state_source' in c for c in proposal['route_contexts']) else f"Fresh new-physics prepared bank{bank_index}, original native task reset; ")+f"{model['registry_id']} declared alpha={proposal['alphas']}; no q or additional baseline.",data_role=proposal['data_role'],
        replanning=dict(period_s=.1,radius_m=1.8,departure_m=1.,maximum_per_case=100,continuous_replanning=True,information='Current causal reconnection, no force/event information'),
        stop=proposal['stop_rule'],scoring=proposal['scoring'],input_sha256={**previous['inputs'],**previous['copied_files_sha256'],**{str(file):digest(file) for file in files}},source_sha256={str(file):digest(file) for file in modules})
    if proposal.get('lower_transfer'):
        transfer=proposal['lower_transfer'];file=Path(transfer['manifest'])
        if digest(file)!=transfer['manifest_sha256']:raise ValueError('lower transfer manifest changed')
        record=json.loads(file.read_text());candidate=record['candidate']
        if record['parent_checkpoint_sha256']!=source['candidate']['checkpoint_sha256']:
            raise ValueError('lower transfer parent differs')
        if any('state_source' in c for c in proposal['route_contexts']) or proposal.get('excluded_completed_cases') or proposal.get('range_planner_boundary'):
            raise ValueError('lower transfer requires fresh paired tasks without old outcome predictor or cached cases')
        if candidate['normalizer']!=source['candidate']['normalizer']:
            raise ValueError('lower transfer normalizer differs')
        for key,hash_key in [('actor','actor_sha256'),('checkpoint','checkpoint_sha256'),('config','config_file_sha256')]:
            if digest(candidate[key])!=candidate[hash_key]:raise ValueError('lower transfer artifact changed')
            p['input_sha256'][candidate[key]]=candidate[hash_key]
        p['input_sha256'][str(file)]=transfer['manifest_sha256']
        p['middle_training_initial_source']=source
        p['initial_source']=dict(source,candidate=candidate)
        p['lower_transfer']=record
        p['initialization']+=' Explicit frozen-middle/lower-candidate zero-shot transfer; original physical preparation retained.'
    save(root/'plan.json',p);save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')));save(root/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))


PREFIX_MODE='native_h75_source_prefix'


def native_source_prefix_inventory(p):
    """Seven declared nominal native sources, with H10 task and six-second sampling."""
    required=dict(mode=PREFIX_MODE,model_id=SELECTION_MODELS['H75'],bank_indices=[4],alpha=.5,
        horizon_s=10.,collection_s=6.,capture_task_s=[1.,2.,4.,6.],expected_cases=7,
        maximum_lower_ticks=8400,optimizer_updates=0,planning_frequency_hz=10.)
    if any(p.get(k)!=v for k,v in required.items()):raise ValueError('bounded native prefix contract differs')
    contexts=p['route_contexts']
    if len(contexts)!=7 or len({c['route'] for c in contexts})!=7 or {c['family'] for c in contexts}!={0,1,2}:raise ValueError('seven distinct straight/single/S routes required')
    if set(p['bank_state_fingerprints'])!={'4'}:raise ValueError('prefix bank identity required')
    if any(p.get(k)!=v for k,v in dict(planner_points=5001,planner_spacing_m=.02,planner_support_m=100.).items()):raise ValueError('native prefix path support differs')
    # The inherited contexts are checked byte-for-byte against the sealed native
    # selection source at prepare, rather than inventing a second route library.
    return [dict(route=c['route'],speed=c['speed'],event='zero',force_N=0.,onset_s=0.,duration_s=.15,
        bank_index=4,initial_fingerprint=p['bank_state_fingerprints']['4'],path_key=f'bank4_{c["route"]}',
        case_id=f'bank4/{c["route"]}/H75/a0.5/zero',mode='H75',mode_index=0,
        model_id=p['model_id'],alpha=.5,query_alpha=.5,user_alpha=.5,effective_user_alpha=.5) for c in contexts]


def prepare_native_source_prefix(source,root,protocol):
    """Prepare seven inference-only source episodes; no integration here."""
    source=Path(source).resolve();root=Path(root).resolve();protocol=Path(protocol).resolve();storage_guard(root)
    proposal=json.loads(protocol.read_text());rows=native_source_prefix_inventory(proposal)
    old=json.loads((source/'plan.json').read_text())
    if old.get('mode')!=SELECTION_MODE or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed native selection interface source required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=old['cases']:raise ValueError('completed source required')
    if proposal['route_contexts']!=old['contexts']:raise ValueError('prefix cannot change the source route definitions')
    if old['models']['H75']['registry_id']!=proposal['model_id'] or old['replanning']['period_s']!=.1 or not old['replanning'].get('continuous_replanning'):raise ValueError('frozen H75/current replanning required')
    files=[protocol,source/'plan.json',source/'seal.json',source/'evaluation/status.json',Path(old['initial_source']['prepared_bank'])]
    for file,sha in {**old['input_sha256'],**old['native_expected_sources']}.items():
        if digest(file)!=sha:raise ValueError('native source input drift '+file)
    for key in ('actor','checkpoint'):
        model=old['models']['H75']
        if digest(model[key])!=model[key+'_sha256']:raise ValueError('frozen H75 drift')
        files.append(Path(model[key]))
    if digest(old['initial_source']['prepared_bank'])!=old['initial_source']['prepared_bank_sha256']:raise ValueError('prepared bank drift')
    folder=Path(__file__).resolve().parent
    modules=[folder/n for n in ('native_goal_run.py','native_goal_continuation.py','native_equilibrium_rule.py','contact_only_environment.py','goal_replanning.py','_vendor/pythonrobotics/dubins.py','_vendor/pythonrobotics/angle.py')]+[Path('learning/cli/run_native_goal_confirmation.py').resolve()]
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    for file in modules:(root/'source'/file.name).write_bytes(file.read_bytes())
    save(root/'inventory.json',rows);save(root/'effective_config.json',old['config']);files += [root/'inventory.json',root/'effective_config.json']
    steps=planning_chunk_steps(proposal['planning_frequency_hz'],proposal['collection_s'])
    p=dict(project='STTW_CONTROL',experiment_id=proposal['experiment_id'],mode=PREFIX_MODE,source=str(source),
        native_runtime_root=old['native_runtime_root'],native_expected_sources=old['native_expected_sources'],initial_source=old['initial_source'],
        config=old['config'],models={'H75':old['models']['H75']},contexts=proposal['route_contexts'],protocol=proposal,
        cases=len(rows),horizon_s=10.,collection_s=6.,goal_radius_m=1.,steps_per_chunk=5,chunks=len(steps),chunk_upper_steps=steps,
        maximum_active_lower_ticks=8400,maximum_allocated_lower_slots=8400,optimizer_updates=0,replanning=old['replanning'],
        data_role=proposal['data_role'],initialization='One authenticated bank4 native reset per route; no baseline or optimizer. Save complete source carry/path/host, retain failures without replacement.',
        scoring='Six-second source prefix only, H10 task semantics retained; no H10 success inference.',stop=proposal['stop_rule'],
        input_sha256={str(file):digest(file) for file in files},source_sha256={str(file):digest(file) for file in modules})
    save(root/'plan.json',p);save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))


FIVE_ONLINE_MODE='native_five_alpha_online'


ROUTE_ONSET_ALGORITHM="SHA256 UTF8 f'{seed}:{route}', first8bytes bigendian modulo61+80, divided50"


def deterministic_route_onsets(routes,seed):
    """Frozen causal-experiment metadata; never an input to the coordinator."""
    if type(seed) is not int or seed<0 or any(type(route) is not str or not route for route in routes) or len(set(routes))!=len(routes):raise ValueError('integer onset seed and unique route names required')
    return {route:(int.from_bytes(hashlib.sha256(f'{seed}:{route}'.encode('utf-8')).digest()[:8],'big')%61+80)/50 for route in routes}


def five_alpha_online_inventory(p):
    required=dict(mode=FIVE_ONLINE_MODE,model_id=SELECTION_MODELS['H75'],bank_indices=[6],user_alpha=.5,
        horizon_s=10.,expected_cases=14,maximum_lower_ticks=28000,optimizer_updates=0,
        planning_frequency_hz=10.,onset_s=2.2,duration_s=.15,forces_N=[-64.,64.],force_frame='worldY',
        candidate_alphas=[0.,.25,.5,.75,1.],arm=0,hypothetical_duration_s=.15,
        task_aware_fallback=False)
    if type(p.get('guard_fallback',False)) is not bool:raise ValueError('guard_fallback must be an explicit boolean')
    if p.get('curvature_task_mode','hard') not in ('hard','soft_penalty'):raise ValueError('unknown curvature task mode')
    nominal=p.get('panel_kind')=='nominal_five_user'
    disturbed_users=p.get('panel_kind')=='disturbed_five_user'
    if p.get('panel_kind','disturbed_single_user') not in ('disturbed_single_user','nominal_five_user','disturbed_five_user'):raise ValueError('unknown online panel kind')
    if nominal:
        required.pop('user_alpha');required.update(expected_cases=35,maximum_lower_ticks=70000,forces_N=[0.],user_alphas=[0.,.25,.5,.75,1.],alpha_dwell_s=.2)
    if disturbed_users:
        required.pop('user_alpha');required.pop('onset_s')
        required.update(expected_cases=70,maximum_lower_ticks=140000,user_alphas=[0.,.25,.5,.75,1.],alpha_dwell_s=.2,guard_fallback=True)
        if 'onset_s' in p:raise ValueError('route-timed panel cannot declare a global onset')
        if any(type(a) not in (int,float) or not math.isfinite(a) for key in ('candidate_alphas','user_alphas') for a in p[key]):raise ValueError('five-alpha grids require finite numeric entries')
        expected_onsets=deterministic_route_onsets([c['route'] for c in p['route_contexts']],p['onset_seed'])
        if p.get('onset_algorithm')!=ROUTE_ONSET_ALGORITHM or p.get('onset_by_route_s')!=expected_onsets or any(type(v) not in (int,float) or not math.isfinite(v) for v in p['onset_by_route_s'].values()):raise ValueError('declared deterministic route onset map differs')
    if any(p.get(k)!=v for k,v in required.items()):raise ValueError('bounded five-alpha online contract differs')
    if isinstance(p.get('alpha_dwell_s',0.),bool) or p.get('alpha_dwell_s',0.) not in (0.,.2):raise ValueError('online dwell must be opt-in .2s')
    if any(p.get(k)!=v for k,v in dict(planner_points=5001,planner_spacing_m=.02,planner_support_m=100.).items()):raise ValueError('online path support differs')
    if set(p['bank_state_fingerprints'])!={'6'} or len(p['route_contexts'])!=7 or len({c['route'] for c in p['route_contexts']})!=7:raise ValueError('seven route/bank identities required')
    if disturbed_users:
        return [dict(route=c['route'],speed=c['speed'],bank_index=6,initial_fingerprint=p['bank_state_fingerprints']['6'],path_key=f'bank6_{c["route"]}',case_id=f'bank6/{c["route"]}/coordinator/a{user:g}/t{onset:g}_F{force:+g}',mode='H75',mode_index=0,model_id=p['model_id'],alpha=user,query_alpha=user,user_alpha=user,effective_user_alpha=user,event=f't{onset:g}_F{force:+g}',force_N=force,onset_s=onset,duration_s=p['duration_s']) for c,user,force in itertools.product(p['route_contexts'],p['user_alphas'],p['forces_N']) for onset in [p['onset_by_route_s'][c['route']]]]
    if nominal:
        return [dict(route=c['route'],speed=c['speed'],bank_index=6,initial_fingerprint=p['bank_state_fingerprints']['6'],path_key=f'bank6_{c["route"]}',case_id=f'bank6/{c["route"]}/coordinator/a{user:g}/zero',mode='H75',mode_index=0,model_id=p['model_id'],alpha=user,query_alpha=user,user_alpha=user,effective_user_alpha=user,event='zero',force_N=0.,onset_s=p['onset_s'],duration_s=p['duration_s']) for c,user in itertools.product(p['route_contexts'],p['user_alphas'])]
    return [dict(route=c['route'],speed=c['speed'],bank_index=6,initial_fingerprint=p['bank_state_fingerprints']['6'],
        path_key=f'bank6_{c["route"]}',case_id=f'bank6/{c["route"]}/coordinator/a0.5/F{force:+g}',
        mode='H75',mode_index=0,model_id=p['model_id'],alpha=.5,query_alpha=.5,user_alpha=.5,effective_user_alpha=.5,
        event=f't2.2_F{force:+g}',force_N=force,onset_s=2.2,duration_s=.15) for c,force in itertools.product(p['route_contexts'],p['forces_N'])]


def audit_five_alpha_online_initial(initial,cases,physical_tick,*,panel_kind=None,expected_route_count=7,task_tick=0):
    """Check original reset leaves; only user alpha is neutralized for pairing."""
    import jax
    import jax.numpy as j
    import numpy as np
    alphas=j.asarray([c['alpha'] for c in cases])
    if not bool(j.all((initial.state.tick==task_tick)&(initial.state.alpha==alphas)&(initial.state.physical.physical_tick==physical_tick))):raise ValueError('online initial clocks/user differ')
    neutral=initial.replace(state=initial.state.replace(alpha=j.zeros_like(initial.state.alpha)))
    leaves=jax.tree.leaves(jax.device_get(neutral))
    groups={}
    for i,c in enumerate(cases):
        key=(c['bank_index'],c['route'],c['event']) if panel_kind in ('disturbed_two_user','disturbed_declared_user') else (c['bank_index'],c['route'])
        groups.setdefault(key,[]).append(i)
    if panel_kind in ('disturbed_two_user','disturbed_declared_user','disturbed_declared_subset'):expected_route_count*=len({c['event'] for c in cases})
    expected_group_size={'nominal_five_user':5,'disturbed_five_user':10,'targeted_three_user':3}.get(panel_kind,2)
    if panel_kind=='disturbed_declared_user':expected_group_size=len({c['alpha'] for c in cases})
    if panel_kind=='boundary_selected_user':
        expected_group_size=1
        groups={}
        for i,c in enumerate(cases):groups.setdefault((c['bank_index'],c['route'],c['event']),[]).append(i)
        expected_route_count*=len({c['event'] for c in cases})
    if panel_kind=='disturbed_declared_subset':
        groups={}
        for i,c in enumerate(cases):groups.setdefault((c['bank_index'],c['route'],c['event']),[]).append(i)
    if len(groups)!=expected_route_count or (panel_kind!='disturbed_declared_subset' and any(len(ids)!=expected_group_size for ids in groups.values())) or any(any(not np.array_equal(x[ids[0]],x[i],equal_nan=True) for i in ids[1:]) for ids in groups.values() for x in leaves):raise ValueError('online per-route initial states differ after user alpha neutralization')
    return dict(authenticated_native_online=True,paired_reset_all_leaves_exact=True,native_leaf_count=len(leaves),alpha_neutralized_only=True,initial_user_alphas=[c['alpha'] for c in cases],initial_previous_alpha='declared user',initial_hold_until_tick=0)


def prepare_five_alpha_online(source,root,protocol):
    source=Path(source).resolve();root=Path(root).resolve();protocol=Path(protocol).resolve();storage_guard(root)
    proposal=json.loads(protocol.read_text());rows=five_alpha_online_inventory(proposal);old=json.loads((source/'plan.json').read_text())
    if old.get('mode')!=SELECTION_MODE or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed selection interface required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=old['cases']:raise ValueError('completed native selection required')
    if proposal['route_contexts']!=old['contexts'] or proposal['bank_state_fingerprints']!=old['protocol']['bank_state_fingerprints']:raise ValueError('online route/bank differs')
    if old['models']['H75']['registry_id']!=proposal['model_id'] or old['replanning']['period_s']!=.1 or not old['replanning'].get('continuous_replanning'):raise ValueError('native H75/current planner differs')
    for file,expected in {**old['input_sha256'],**old['native_expected_sources']}.items():
        if digest(file)!=expected:raise ValueError('native source input drift '+file)
    folder=Path(__file__).resolve().parent;coordinator=file_module('_sttw_five_online_auth',folder/'tolerance_coordination.py')
    coordinator.load_five_alpha_export(proposal['predictor_training'],expected_export_sha256=proposal['export_manifest_sha256'],candidate_alphas=tuple(proposal['candidate_alphas']),hypothetical_duration_s=proposal['hypothetical_duration_s'])
    training=Path(proposal['predictor_training']);selection=json.loads((training/'selection_manifest.json').read_text())
    files=[protocol,source/'plan.json',source/'seal.json',Path(old['initial_source']['prepared_bank'])]+[training/n for n in ('export_manifest.json','parameters.npz','selection_manifest.json','selection_seal.json','data_receipt.json')]+[Path(r['file']) for r in selection['selected']]+[Path(selection['config'])]
    for key in ('actor','checkpoint'):
        model=old['models']['H75']
        if digest(model[key])!=model[key+'_sha256']:raise ValueError('H75 identity differs')
        files.append(Path(model[key]))
    modules=[folder/n for n in ('native_goal_run.py','native_goal_continuation.py','native_equilibrium_rule.py','contact_only_environment.py','goal_replanning.py','geometric_preference_schedule.py','tolerance_coordination.py','_vendor/pythonrobotics/dubins.py','_vendor/pythonrobotics/angle.py')]+[Path('learning/cli/run_native_goal_confirmation.py').resolve()]
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    for file in modules:(root/'source'/file.name).write_bytes(file.read_bytes())
    save(root/'inventory.json',rows);save(root/'effective_config.json',old['config']);files += [root/'inventory.json',root/'effective_config.json']
    steps=planning_chunk_steps(10.,10.)
    p=dict(project='STTW_CONTROL',experiment_id=proposal['experiment_id'],mode=FIVE_ONLINE_MODE,source=str(source),
        native_runtime_root=old['native_runtime_root'],native_expected_sources=old['native_expected_sources'],initial_source=old['initial_source'],
        config=old['config'],models={'H75':old['models']['H75']},contexts=proposal['route_contexts'],protocol=proposal,
        cases=len(rows),horizon_s=10.,goal_radius_m=1.,steps_per_chunk=5,chunks=100,chunk_upper_steps=steps,
        maximum_active_lower_ticks=proposal['maximum_lower_ticks'],maximum_allocated_lower_slots=proposal['maximum_lower_ticks'],optimizer_updates=0,replanning=old['replanning'],
        data_role=proposal['data_role'],initialization='Authenticated native bank6 reset, unchanged complete native carry plus separate latch and previous alpha. Initial previous alpha equals each declared user; optional hold deadline0.',
        scoring=proposal['scoring'],stop=proposal['stop_rule'],input_sha256={str(file):digest(file) for file in files},source_sha256={str(file):digest(file) for file in modules})
    save(root/'plan.json',p);save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))


LABEL_MODE='five_alpha_tolerance_labels'


def snapshot_block_for_task(source_plan, task_s):
    """Locate a saved AFTER-chunk / BEFORE-next-planner task boundary."""
    target=task_s*200
    if not math.isfinite(target) or target!=round(target):raise ValueError('source must be an exact lower-tick boundary')
    cumulative=0
    for block,steps in enumerate(source_plan['chunk_upper_steps']):
        cumulative+=steps*4
        if cumulative==target:return block
        if cumulative>target:break
    raise ValueError('source task time is not a saved chunk boundary')


def label_collection_steps(protocol, source_task_s):
    """Stop collection on an upper boundary, preserving the native H10 task."""
    if 'label_collection_duration_s' in protocol and 'collection_end_s' in protocol:raise ValueError('choose relative or absolute label collection endpoint')
    duration=protocol.get('label_collection_duration_s')
    if duration is not None and (isinstance(duration,bool) or not isinstance(duration,(int,float)) or not math.isfinite(duration) or duration<=0):raise ValueError('invalid label collection duration')
    end=source_task_s+duration if duration is not None else protocol.get('collection_end_s',protocol['horizon_s'])
    if isinstance(end,bool) or not isinstance(end,(int,float)) or not math.isfinite(end) or not source_task_s<end<=protocol['horizon_s']:
        raise ValueError('invalid label collection endpoint')
    upper=(end-source_task_s)*50
    if abs(upper-round(upper))>1e-9:raise ValueError('collection endpoint must be an upper boundary')
    count=round(upper);whole,tail=divmod(count,5)
    return [5]*whole+([tail] if tail else [])


def five_alpha_label_inventory(p, source_cases=None, task_s=None, source_plan=None):
    """Declarative related-DEV branching, authenticated against source metadata."""
    required=dict(mode=LABEL_MODE,model_id='STTW_MIDDLE_FULLTAIL_PPO75_U77',
        candidate_alphas=[0.,.25,.5,.75,1.],preparation_s=[.2,.6],forces_N=[-64.,64.],
        force_frame='worldY',duration_s=.15,horizon_s=10.,primary_after_pulse_end_s=3.,
        planning_frequency_hz=10.,maximum_new_prefix_lower_ticks=0,optimizer_updates=0)
    if any(p.get(k)!=v for k,v in required.items()):raise ValueError('five-alpha label contract differs')
    if source_cases is None:source_cases=json.loads((Path(p['source'])/'inventory.json').read_text())
    if source_plan is None:source_plan=json.loads((Path(p['source'])/'plan.json').read_text())
    groups=p['source_groups'];count=p['expected_source_states']
    if len(groups)!=count or [g['source_index'] for g in groups]!=list(range(count)):raise ValueError('exact source groups required')
    times=sorted({g['source_task_s'] for g in groups})
    if task_s is not None and task_s not in times:raise ValueError('undeclared source group clock')
    nominal={i for i,c in enumerate(source_cases) if c.get('mode')=='H75' and c.get('alpha')==.5 and c.get('force_N')==0.}
    if 'source_routes' in p:
        routes=p['source_routes']
        available={source_cases[i]['route'] for i in nominal}
        if not isinstance(routes,list) or not routes or len(set(routes))!=len(routes) or not set(routes)<=available:raise ValueError('explicit unique available source routes required')
        nominal={i for i in nominal if source_cases[i]['route'] in routes}
    if not nominal or any({g['source_lane'] for g in groups if g['source_task_s']==t}!=nominal for t in times):raise ValueError('all nominal source lanes required at each clock')
    if len({(g['source_task_s'],g['source_lane']) for g in groups})!=len(groups):raise ValueError('duplicate complete source group')
    budget=sum(sum(label_collection_steps(p,g['source_task_s']))*4*20 for g in groups)
    if p['expected_cases']!=count*20 or p['maximum_suffix_lower_ticks']!=budget:raise ValueError('declared source-derived suffix budget differs')
    rows=[]
    for g in groups:
        idx=g['source_index'];t=g['source_task_s'];block=snapshot_block_for_task(source_plan,t)
        if t<=0 or t+max(p['preparation_s'])+p['duration_s']+p['primary_after_pulse_end_s']>p['horizon_s']:raise ValueError('source label window outside horizon')
        lane=g['source_lane'];c=source_cases[lane]
        expected=dict(source_block=block,source_user_alpha=.5,complete_carry=f'evaluation/complete_carry_{block:02d}.pkl',
            executed_paths=f'evaluation/executed_paths_{block:02d}.pkl',host_state=f'evaluation/host_state_{block:02d}.npz',expected_task_tick=round(t*200),
            source_case_id=c['case_id'],route=c['route'],path_key=case_path_key(c),bank_index=c['bank_index'])
        if any(g.get(k)!=v for k,v in expected.items()):raise ValueError('source lane metadata differs '+str(idx))
        if p.get('require_ongoing_sources',False) is not False and (p['require_ongoing_sources'] is not True or g['expected_ever_arrived']):raise ValueError('ongoing-only sources required')
        if t+max(p['preparation_s'])+p['duration_s']+p['primary_after_pulse_end_s']>t+sum(label_collection_steps(p,t))*.02+1e-9:raise ValueError('primary label window outside collection')
        if not isinstance(g['expected_ever_arrived'],bool) or not isinstance(g['expected_replanned'],bool):raise ValueError('explicit boolean source host flags required')
        if task_s is not None and t!=task_s:continue
        identity=f"{Path(p['source']).resolve()}@{p['source_plan_sha256']}/tasktick{g['expected_task_tick']}/{g['source_case_id']}"
        for a,prep,force in itertools.product(p['candidate_alphas'],p['preparation_s'],p['forces_N']):
            rows.append(dict(g,source_group_identity=identity,source_origin='saved_nominal_half',speed=c['speed'],mode='H75',mode_index=0,
                alpha=.5,user_alpha=.5,candidate_alpha=a,query_alpha=a,preparation_s=prep,onset_s=t+prep,
                duration_s=.15,force_N=force,primary_window_start_s=t,primary_window_end_s=t+prep+.15+3.,
                case_id=f'source{idx}/alpha{a:g}/prep{prep:g}/F{force:+g}'))
    return rows


def import_five_alpha_sources(source, cases):
    """Import complete pre-boundary carry/path/host; never integrate or reset."""
    import jax
    import jax.numpy as j
    import numpy as np
    source=Path(source);cache={};car=[];paths=[];changed=[];arrived=[];goals=[]
    for c in cases:
        key=c['complete_carry']
        if key not in cache:
            with (source/key).open('rb') as f:carry=pickle.load(f)
            with (source/c['executed_paths']).open('rb') as f:path=pickle.load(f)
            with np.load(source/c['host_state']) as h:host={k:h[k].copy() for k in h.files}
            cache[key]=(carry,path,host)
        carry,path,host=cache[key];i=c['source_lane']
        selected=jax.tree.map(lambda x:j.asarray(x[i]),carry)
        expected=jax.tree.leaves(jax.tree.map(lambda x:np.asarray(x[i]),carry))
        actual=jax.tree.leaves(jax.device_get(selected))
        if any(a.dtype!=b.dtype or not np.array_equal(a,b,equal_nan=True) for a,b in zip(actual,expected)):raise ValueError('complete source copy changed leaf dtype/value')
        if hasattr(selected,'native'):raise ValueError('native selection source must not have online wrapper')
        st=selected.state
        if (int(st.tick)!=c['expected_task_tick'] or int(st.physical.physical_tick)!=c['expected_physical_tick'] or
            float(st.alpha)!=.5 or bool(st.fault) or bool(st.physical.failed)):
            raise ValueError('saved source clock/user/failure differs')
        if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(selected) if np.issubdtype(np.asarray(x).dtype,np.number)):raise ValueError('nonfinite source carry')
        if bool(host['replanned'][i])!=c['expected_replanned'] or bool(host['ever_arrived'][i])!=c['expected_ever_arrived']:raise ValueError('saved source host flags differ')
        car.append(selected);paths.append(jax.tree.map(lambda x:j.asarray(x[i]),path))
        changed.append(host['replanned'][i]);arrived.append(host['ever_arrived'][i]);goals.append(host['original_goals_xy'][i])
    stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
    return stack(car),stack(paths),dict(replanned=np.asarray(changed),ever_arrived=np.asarray(arrived),original_goals_xy=np.asarray(goals))


def validate_five_alpha_source_hosts(source,groups):
    """Authenticate declared host flags and physical origin before preparation."""
    source=Path(source)
    import numpy as np
    manifest=json.loads((source/'evaluation/manifest.json').read_text())
    physical_origin_tick=round(manifest['physical_start_s']*200)
    for g in groups:
        if g['expected_physical_tick']!=physical_origin_tick+g['expected_task_tick']:raise ValueError('source physical origin differs')
        with np.load(source/g['host_state']) as host:
            i=g['source_lane']
            if bool(host['replanned'][i])!=g['expected_replanned'] or bool(host['ever_arrived'][i])!=g['expected_ever_arrived']:raise ValueError('source host declaration differs')


def prepare_five_alpha_labels(source,root,protocol):
    source=Path(source).resolve();root=Path(root).resolve();protocol=Path(protocol).resolve();storage_guard(root)
    proposal=json.loads(protocol.read_text());old=json.loads((source/'plan.json').read_text());cs=json.loads((source/'inventory.json').read_text())
    rows=five_alpha_label_inventory(proposal,cs,source_plan=old)
    if 'physics_identity' in proposal and proposal['physics_identity']!=old['initial_source']['model']:raise ValueError('label source physics identity differs')
    if str(source)!=proposal['source'] or digest(source/'plan.json')!=proposal['source_plan_sha256'] or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('selection source identity differs')
    status=json.loads((source/'evaluation/status.json').read_text())
    if old.get('mode') not in (SELECTION_MODE,PREFIX_MODE,CAPABILITY_MODE) or status.get('phase')!='completed' or status.get('completed_cases')!=old['cases']:raise ValueError('completed selection source required')
    if old['models']['H75']['registry_id']!=proposal['model_id'] or not old['replanning'].get('continuous_replanning') or old['replanning']['period_s']!=.1:raise ValueError('frozen native H75/10Hz identity differs')
    for file,sha in {**old['input_sha256'],**old['native_expected_sources']}.items():
        if digest(file)!=sha:raise ValueError('source input/native drift '+file)
    for file,sha in old['source_sha256'].items():
        if digest(source/'source'/Path(file).name)!=sha:raise ValueError('archived source drift')
    validate_five_alpha_source_hosts(source,proposal['source_groups'])
    folder=Path(__file__).resolve().parent
    modules=[folder/n for n in ('native_goal_run.py','native_goal_continuation.py','native_equilibrium_rule.py','contact_only_environment.py','goal_replanning.py','_vendor/pythonrobotics/dubins.py','_vendor/pythonrobotics/angle.py')]+[Path('learning/cli/run_native_goal_confirmation.py').resolve()]
    files=[protocol,source/'plan.json',source/'seal.json',source/'inventory.json',source/'evaluation/status.json',source/'evaluation/manifest.json']
    files += [source/name for name in {g[k] for g in proposal['source_groups'] for k in ('complete_carry','executed_paths','host_state')}]
    files += [source/'evaluation'/(key+'_path.npz') for key in sorted({g['path_key'] for g in proposal['source_groups']})]
    for k in ('actor','checkpoint'):
        model=old['models']['H75']
        if digest(model[k])!=model[k+'_sha256']:raise ValueError('H75 drift')
        files.append(Path(model[k]))
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    for file in modules:(root/'source'/file.name).write_bytes(file.read_bytes())
    common=dict(initialization=proposal['initialization'],stop=proposal['stop_rule'],project='STTW_CONTROL',experiment_id=proposal['experiment_id'],mode=LABEL_MODE,
        source=str(source),native_runtime_root=old['native_runtime_root'],native_expected_sources=old['native_expected_sources'],
        initial_source=old['initial_source'],config=old['config'],models={'H75':old['models']['H75']},
        contexts=old['contexts'],horizon_s=10.,goal_radius_m=1.,steps_per_chunk=5,optimizer_updates=0,
        protocol={**proposal,'planner_points':5001,'planner_spacing_m':.02,'planner_support_m':100.,
                  'bank_indices':sorted({g['bank_index'] for g in proposal['source_groups']}),'bank_state_fingerprints':old['protocol']['bank_state_fingerprints']},
        replanning=old['replanning'],data_role=proposal['data_role'],scoring=proposal['label'],
        input_sha256={str(file):digest(file) for file in files},source_sha256={str(file):digest(file) for file in modules})
    children=[]
    for t in sorted({g['source_task_s'] for g in proposal['source_groups']}):
        child=root/f'task{t:g}';child.mkdir();(child/'source').mkdir()
        for file in modules:(child/'source'/file.name).write_bytes(file.read_bytes())
        rr=five_alpha_label_inventory(proposal,cs,t,old);save(child/'inventory.json',rr);save(child/'effective_config.json',old['config'])
        steps=label_collection_steps(proposal,t);budget=sum(steps)*4*len(rr)
        plan=dict(common,source_task_s=t,collection_end_s=t+sum(steps)*.02,cases=len(rr),chunks=len(steps),chunk_upper_steps=steps,
            maximum_active_lower_ticks=budget,maximum_allocated_lower_slots=budget)
        plan['input_sha256']={**common['input_sha256'],str(child/'inventory.json'):digest(child/'inventory.json'),str(child/'effective_config.json'):digest(child/'effective_config.json')}
        save(child/'plan.json',plan);save(child/'seal.json',dict(plan_sha256=digest(child/'plan.json')))
        save(child/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0));children.append(str(child))
    save(root/'inventory.json',rows)
    parent=dict(common,label_parent=True,children=children,cases=len(rows),maximum_active_lower_ticks=proposal['maximum_suffix_lower_ticks'],maximum_allocated_lower_slots=proposal['maximum_suffix_lower_ticks'])
    parent['input_sha256']={**common['input_sha256'],str(root/'inventory.json'):digest(root/'inventory.json'),**{str(Path(c)/n):digest(Path(c)/n) for c in children for n in ('plan.json','seal.json')}}
    save(root/'plan.json',parent);save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))


def run_five_alpha_label_groups(root,shape_only=False):
    """Declared clock groups; execution is never retried implicitly."""
    root=Path(root);p=verify(root)
    if not p.get('label_parent'):raise ValueError('label parent required')
    if not shape_only and json.loads((root/'status.json').read_text())['phase']!='prepared':raise ValueError('prepared only')
    active=allocated=0
    try:
        if not shape_only:save(root/'status.json',dict(phase='running',pid=os.getpid(),active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))
        for child in p['children']:
            run(child,shape_only=shape_only)
            if not shape_only:
                st=json.loads((Path(child)/'status.json').read_text());active+=st['active_lower_ticks'];allocated+=st['allocated_lower_slots']
                if active>p['maximum_active_lower_ticks'] or allocated>p['maximum_allocated_lower_slots']:raise RuntimeError('label campaign budget exceeded')
                save(root/'status.json',dict(phase='running',pid=os.getpid(),active_lower_ticks=active,allocated_lower_slots=allocated,optimizer_updates=0))
        if shape_only:
            save(root/'shape_receipt.json',dict(phase='completed',physics_steps=0,groups=p['children'],lanes_per_group=[json.loads((Path(c)/'plan.json').read_text())['cases'] for c in p['children']]))
        else:save(root/'status.json',dict(phase='completed',active_lower_ticks=active,allocated_lower_slots=allocated,optimizer_updates=0,completed_cases=p['cases']))
    except BaseException as exc:
        if not shape_only:
            # Include a failed child's last returned block, without claiming inflight work was zero.
            statuses=[json.loads((Path(c)/'status.json').read_text()) for c in p['children']]
            save(root/'status.json',dict(phase='error',error=repr(exc),active_lower_ticks=sum(x.get('active_lower_ticks',0) for x in statuses),allocated_lower_slots=sum(x.get('allocated_lower_slots',0) for x in statuses),optimizer_updates=0,inflight_cost_unknown=True))
        raise


def inventory():
    events=[dict(event='zero',onset_s=1.6,force_N=0.)]
    events += [dict(event=f't{t:g}_F{f:+g}',onset_s=t,force_N=f)
               for t,f in itertools.product((1.6,2.6),(-32.,32.))]
    return [dict(case_id=f'{side}70_R2p1_v2p2/a{a:g}/{mode}/{e["event"]}',
                 route=side+'70_R2p1',builder=side+'90_R2',speed=2.2,alpha=a,
                 mode=mode,mode_index=i,duration_s=.1,**e)
            for side,a,(i,mode),e in itertools.product(
                ('left','right'),(0.,1.),enumerate(('H75','current','zero','equilibrium')),events)]



def response_protocol():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/h75_channel_response.json').read_text())


def response_inventory(protocol=None):
    p=response_protocol() if protocol is None else protocol
    validate_response_protocol(p)
    events=[dict(event='zero',onset_s=p['zero_onset_s'],force_N=0.)]
    events += [dict(event=f't{t:g}_F{f:+g}',onset_s=t,force_N=f)
               for t,f in itertools.product(p['onsets_s'],p['forces_N'])]
    rows=[]
    for route,(sa,ta),event in itertools.product(p['routes'],p['mappings'],events):
        mode='H75' if sa==ta else 'hybrid';alpha=sa if sa==ta else 1.
        rows.append(dict(case_id=f'{route["route"]}/a{alpha:g}_{mode}/{event["event"]}',
            **route,alpha=alpha,mode=mode,mode_index=0,speed_alpha=sa,steer_alpha=ta,
            duration_s=p['duration_s'],**event))
    if len(rows)!=p['expected_cases']:raise ValueError('response inventory size mismatch')
    return rows


def validate_response_protocol(p):
    # Bounded authorized contract; descriptive metadata may be enriched freely.
    required=dict(horizon_s=10.,expected_cases=54,maximum_lower_ticks=108000,optimizer_updates=0,
        force_frame='worldY',initial_bank=3,duration_s=.1,goal_radius_m=1.,replanning_radius_m=1.8,
        departure_m=1.,maximum_replans=1,replan_cadence_s=1.,planner_support_m=100.,planner_spacing_m=.02,planner_points=5001)
    for key,value in required.items():
        if p.get(key)!=value:raise ValueError('response contract mismatch '+key)
    if (p.get('forces_N')!=[-64.,-32.,32.,64.] or p.get('onsets_s')!=[1.6,4.] or
        p.get('mappings')!=[[.5,.5],[1.,1.],[1.,.5]] or p.get('zero_onset_s')!=1.6 or
        {(r['route'],r['speed']) for r in p.get('routes',[])}!={('S_left_90_R1p8',2.8),('S_right_90_R1p8',2.8)}):
        raise ValueError('response route/mapping/event contract mismatch')





def switching_protocol():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/h75_channel_switching.json').read_text())


def switching_inventory(p):
    base=dict(p,expected_cases=54,maximum_lower_ticks=108000,forces_N=[-64.,-32.,32.,64.],onsets_s=[1.6,4.])
    validate_response_protocol(base)
    if (p['expected_cases']!=36 or p['maximum_lower_ticks']!=72000 or
        p['forces_N']!=[-64.,64.] or p['onsets_s']!=[1.6] or p['switch_times_s']!=[1.,1.4,1.6] or
        p['destination_pair']!=[.5,.5] or p['capture_task_s']!=1. or
        p['origins']!=[dict(origin='fixed1',speed_alpha=1.,steer_alpha=1.),dict(origin='hybrid',speed_alpha=1.,steer_alpha=.5)]):
        raise ValueError('bounded switching contract mismatch')
    if p['prefix_comparison'].get('absolute_tolerance')!=1e-5 or p['prefix_comparison'].get('relative_tolerance')!=1e-6:
        raise ValueError('switch cached-prefix tolerance contract mismatch')
    events=[dict(event='zero',force_N=0.),*[dict(event=f't1.6_F{f:+g}',force_N=f) for f in p['forces_N']]]
    return [dict(**route,**origin,**event,case_id=f'{route["route"]}/{origin["origin"]}/switch{t:g}/{event["event"]}',
        mode='switch_'+origin['origin'],mode_index=0,alpha=1.,switch_s=t,onset_s=p['onsets_s'][0],duration_s=p['duration_s'])
        for route,origin,t,event in itertools.product(p['routes'],p['origins'],p['switch_times_s'],events)]




def online_inventory(p):
    base=dict(p,expected_cases=54,maximum_lower_ticks=108000,mappings=[[.5,.5],[1.,1.],[1.,.5]])
    rows=response_inventory(base)
    if p['expected_cases']!=18 or p['maximum_lower_ticks']!=36000 or p['rule']!=dict(preview_m=1.5,engage_accel_m_s2=3.,release_accel_m_s2=2.7,active_alpha=.5):
        raise ValueError('bounded online scalar contract mismatch')
    return [dict(c,mode='online_scalar',case_id=c['case_id'].replace('/a1_H75/','/online_scalar/')) for c in rows if c['alpha']==1. and c['mode']=='H75']


def prepare_online(source,root,protocol):
    return prepare_switching(source,root,protocol,online=True)



def samestate_inventory(p):
    if (p['expected_cases']!=72 or p['horizon_s']!=10. or p['new_horizon_s']!=9. or p['maximum_lower_ticks']!=129600 or
        p['optimizer_updates']!=0 or p['source_task_s']!=1. or p['candidate_alphas']!=[0.,.5,1.] or
        p['preparation_s']!=[0.,.2,.6] or p['forces_N']!=[-64.,64.] or p['duration_s']!=.1 or
        p['primary_after_pulse_end_s']!=3. or [r['source_index'] for r in p['source_rows']]!=[0,9,18,27]):
        raise ValueError('bounded same-state72 contract mismatch')
    rows=[]
    for src,a,prep,force in itertools.product(p['source_rows'],p['candidate_alphas'],p['preparation_s'],p['forces_N']):
        onset=p['source_task_s']+prep
        rows.append(dict(route=src['route'],speed=2.8,source_index=src['source_index'],source_origin=src['origin'],source_fingerprint=src['fingerprint'],
            case_id=f'source{src["source_index"]}/alpha{a:g}/prep{prep:g}/F{force:+g}',mode='samestate_scalar',mode_index=0,
            alpha=1.,candidate_alpha=a,query_alpha=a,preparation_s=prep,onset_s=onset,duration_s=p['duration_s'],force_N=force,
            primary_window_start_s=p['source_task_s'],primary_window_end_s=onset+p['duration_s']+p['primary_after_pulse_end_s']))
    return rows




def new_confirmation_sources(p):
    return p.get('source_initialization')=='prepared_bank_all_prefixes' or (p.get('mode') in ('h75_late_state_tolerance48','h75_user_late_state192') and p.get('source_initialization')=='saved_task6_snapshot')


def late_state_inventory(p):
    if p.get('mode')=='h75_user_late_state192':return user_late_inventory(p)
    if (p['expected_cases']!=48 or p['source_task_s']!=6. or p['horizon_s']!=10. or p['new_horizon_s']!=4. or
        p['maximum_prefix_lower_ticks']!=0 or p['maximum_lower_ticks']!=38400 or p['maximum_total_lower_ticks']!=38400 or
        p['candidate_alphas']!=[0.,.5,1.] or p['preparation_s']!=[.2,.6] or p['forces_N']!=[-64.,64.] or
        p['duration_s']!=.1 or p['primary_after_pulse_end_s']!=3. or p['optimizer_updates']!=0 or p['bank_indices']!=[2,5]):
        raise ValueError('bounded late48 contract mismatch')
    groups=p['source_groups']
    if len(groups)!=4 or [g['source_index'] for g in groups]!=list(range(4)) or {(g['bank_index'],g['route'],g['origin']) for g in groups}!={(b,r,'fixed_half') for b in (2,5) for r in ('S_left_90_R1p8','S_right_90_R1p8')}:
        raise ValueError('four fixed-half late source groups required')
    rows=[]
    for g,a,prep,force in itertools.product(groups,p['candidate_alphas'],p['preparation_s'],p['forces_N']):
        onset=p['source_task_s']+prep
        rows.append(dict(g,source_origin=g['origin'],speed=2.8,mode='samestate_scalar',mode_index=0,alpha=1.,candidate_alpha=a,query_alpha=a,preparation_s=prep,onset_s=onset,duration_s=.1,force_N=force,primary_window_start_s=6.,primary_window_end_s=onset+.1+3.,case_id=f'{g["source_group_id"]}/alpha{a:g}/prep{prep:g}/F{force:+g}'))
    return rows


def user_late_inventory(p):
    expected={(b,r,origin) for b in (4,6) for r in ('S_left_90_R1p8','S_right_90_R1p8','S_left_60_R2p2','S_right_60_R2p2') for origin in ('user0_last','userhalf_last')}
    required=dict(expected_cases=192,source_task_s=6.,horizon_s=10.,new_horizon_s=4.,maximum_prefix_lower_ticks=0,maximum_lower_ticks=153600,maximum_total_lower_ticks=153600,candidate_alphas=[0.,.5,1.],preparation_s=[.2,.6],forces_N=[-64.,64.],duration_s=.1,primary_after_pulse_end_s=3.,optimizer_updates=0,bank_indices=[4,6],source_initialization='saved_task6_snapshot',force_frame='worldY',replan_cadence_s=1.,replanning_radius_m=1.8,departure_m=1.,maximum_replans=1,goal_radius_m=1.,planner_support_m=100.,planner_spacing_m=.02,planner_points=5001)
    if any(p.get(k)!=v for k,v in required.items()):raise ValueError('bounded user late192 contract mismatch')
    groups=p['source_groups']
    if len(groups)!=16 or [g['source_index'] for g in groups]!=list(range(16)) or len({g['source_group_id'] for g in groups})!=16 or {(g['bank_index'],g['route'],g['origin']) for g in groups}!=expected:raise ValueError('all sixteen nominal user late sources required')
    rows=[]
    for g,a,prep,force in itertools.product(groups,p['candidate_alphas'],p['preparation_s'],p['forces_N']):
        user=0. if g['origin']=='user0_last' else .5
        speed=2.8 if '90_R1p8' in g['route'] else 2.6
        if g['source_user_alpha']!=user or g['speed']!=speed:raise ValueError('source user/speed metadata mismatch')
        onset=6.+prep
        rows.append(dict(g,source_origin=g['origin'],mode='samestate_scalar',mode_index=0,alpha=user,user_alpha=user,effective_user_alpha=user,candidate_alpha=a,query_alpha=a,preparation_s=prep,onset_s=onset,duration_s=.1,force_N=force,primary_window_start_s=6.,primary_window_end_s=onset+.1+3.,case_id=f'{g["source_group_id"]}/alpha{a:g}/prep{prep:g}/F{force:+g}'))
    return rows


def late_source_rows(cases,groups):
    rows=[]
    for group in groups:
        matches=[i for i,c in enumerate(cases) if c['bank_index']==group['bank_index'] and c['route']==group['route'] and c['mode']==group['origin'] and c['force_N']==0.]
        if len(matches)!=1:raise ValueError('one fixed-half nominal saved task6 source required')
        rows.append(matches[0])
    return rows


def prepare_late_state(source,root,protocol):
    source=Path(source).resolve();p=json.loads((source/'plan.json').read_text())
    if p.get('mode')!=('user_preference120' if json.loads(Path(protocol).read_text()).get('mode')=='h75_user_late_state192' else 'tolerance_online120') or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:
        raise ValueError('sealed online120 frozen q ancestor required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=120:raise ValueError('completed online120 required')
    proposal=json.loads(Path(protocol).read_text());late_state_inventory(proposal)
    selected_rows=late_source_rows(json.loads((source/'inventory.json').read_text()),proposal['source_groups'])
    if len(set(selected_rows))!=len(proposal['source_groups']):raise ValueError('late source lanes not unique')
    identity={str(source/name):digest(source/name) for name in ('plan.json','seal.json','inventory.json','evaluation/status.json','evaluation/complete_carry_05.pkl','evaluation/executed_paths_05.pkl','evaluation/host_state_05.npz')}
    for f,sha in p['input_sha256'].items():
        if digest(f)!=sha:raise ValueError('online120 input drift '+f)
        identity[f]=sha
    for f,sha in p['source_sha256'].items():
        archived=source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('online120 archived source drift')
        identity[str(archived)]=sha
    prepare_switching(p['stay_source'],root,protocol,samestate_source=dict(source=str(source),identity=identity,bank144=True,late=True))


def bank144_inventory(p):
    if p.get('mode') in ('h75_late_state_tolerance48','h75_user_late_state192'):return late_state_inventory(p)
    fresh=new_confirmation_sources(p)
    if (p['expected_cases']!=144 or p['maximum_lower_ticks']!=259200 or p['maximum_prefix_lower_ticks']!=(1600 if fresh else 800) or
        p['maximum_total_lower_ticks']!=(260800 if fresh else 260000) or p['horizon_s']!=10. or p['new_horizon_s']!=9. or
        p['source_task_s']!=1. or p['candidate_alphas']!=[0.,.5,1.] or p['preparation_s']!=[0.,.2,.6] or
        p['forces_N']!=[-64.,64.] or p['duration_s']!=.1 or p['primary_after_pulse_end_s']!=3. or p['optimizer_updates']!=0):
        raise ValueError('bounded bank144 contract mismatch')
    groups=p['source_groups']
    if len(groups)!=8 or [g['source_index'] for g in groups]!=list(range(8)):raise ValueError('bank144 source indices mismatch')
    expected={(b,r,o) for b in ((2,5) if fresh else (0,1)) for r in ('S_left_90_R1p8','S_right_90_R1p8') for o in ('fixed1','hybrid')}
    if {(g['bank_index'],g['route'],g['origin']) for g in groups}!=expected:raise ValueError('bank144 source group mismatch')
    if fresh and (p.get('bank_indices')!=[2,5] or set(p.get('bank_state_fingerprints',{}))!={'2','5'}):raise ValueError('new confirmation bank identity missing')
    rows=[]
    for group,a,prep,force in itertools.product(groups,p['candidate_alphas'],p['preparation_s'],p['forces_N']):
        if group['split']!=('new_confirmation' if fresh else 'development' if group['bank_index']==0 else 'label_heldout'):raise ValueError('bank group split mismatch')
        onset=1.+prep
        rows.append(dict(group,source_origin=group['origin'],speed=2.8,mode='samestate_scalar',mode_index=0,alpha=1.,
            candidate_alpha=a,query_alpha=a,preparation_s=prep,onset_s=onset,duration_s=.1,force_N=force,
            primary_window_start_s=1.,primary_window_end_s=onset+.1+3.,case_id=f'{group["source_group_id"]}/alpha{a:g}/prep{prep:g}/F{force:+g}'))
    return rows


def bank144_prefix_cases(p):
    fresh=new_confirmation_sources(p)
    if p.get('mode')=='h75_user_late_state192':return [dict(g,source_origin=g['origin'],mode='hybrid_prefix',mode_index=0,alpha=1.,speed_alpha=g['source_user_alpha'],steer_alpha=g['source_user_alpha'],onset_s=0.,duration_s=.1,force_N=0.,case_id=g['source_group_id']+'/prefix') for g in p['source_groups']]
    return [dict(g,source_origin=g['origin'],speed=2.8,mode='hybrid_prefix',mode_index=0,alpha=1.,speed_alpha=.5 if g['origin']=='fixed_half' else 1.,steer_alpha=.5 if g['origin'] in ('hybrid','fixed_half') else 1.,
        onset_s=0.,duration_s=.1,force_N=0.,case_id=g['source_group_id']+'/prefix') for g in p['source_groups'] if fresh or g['origin']=='hybrid']


def prepare_bank144(source,root,protocol):
    source=Path(source).resolve();p=json.loads((source/'plan.json').read_text())
    if p.get('mode')!='h75_online_confirmation100' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:
        raise ValueError('sealed confirmation100 required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=100:raise ValueError('completed100 required')
    identity={}
    for f,sha in p['source_sha256'].items():
        archived=source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('confirmation source snapshot drift')
        identity[str(archived)]=sha
    for f,sha in p['input_sha256'].items():
        if digest(f)!=sha:raise ValueError('confirmation input drift '+f)
        identity[f]=sha
    proposal=json.loads(Path(protocol).read_text());bank144_inventory(proposal)
    old=json.loads((source/'inventory.json').read_text())
    for group in ([] if new_confirmation_sources(proposal) else proposal['source_groups']):
        c=old[group['confirmation_row']]
        if c['bank_index']!=group['bank_index'] or c['route']!=group['route'] or c['mode']!='fixed1' or c['force_N']!=0.:
            raise ValueError('fixed metadata confirmation source selection mismatch')
    names=['plan.json','seal.json'] if new_confirmation_sources(proposal) else ['plan.json','seal.json','inventory.json','evaluation/status.json','evaluation/initial_carry.pkl','evaluation/complete_carry_00.pkl','evaluation/host_state_00.npz','evaluation/executed_paths_00.pkl']
    names += ['evaluation/'+g['path_key']+'_path.npz' for g in proposal['source_groups'] if g['origin']=='fixed1' and not new_confirmation_sources(proposal)]
    for name in names:
        f=source/name;identity[str(f)]=digest(f)
    prepare_switching(p['stay_source'],root,protocol,samestate_source=dict(source=str(source),identity=identity,bank144=True))

def gather_source_carry(saved,indices):
    """Copy existing complete state, without reconstructing/resetting any leaf."""
    import jax
    import jax.numpy as j
    import numpy as np
    result=jax.tree.map(lambda x:j.asarray(x)[j.asarray(indices)],saved)
    aa=jax.tree.leaves(jax.device_get(result));bb=jax.tree.leaves(saved)
    if len(aa)!=len(bb) or any(a.dtype!=np.asarray(b).dtype or any(not np.array_equal(a[i],np.asarray(b)[index],equal_nan=True) for i,index in enumerate(indices)) for a,b in zip(aa,bb)):
        raise ValueError('complete source-copy leaves changed')
    return result

def prepare_samestate(source,root,protocol):
    source=Path(source).resolve();p=json.loads((source/'plan.json').read_text())
    if p.get('mode')!='h75_switching_continuation' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:
        raise ValueError('sealed same-state source required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=36:raise ValueError('completed36 source required')
    identity={}
    for f,sha in p['source_sha256'].items():
        archived=source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('same-state archived source changed')
        identity[str(archived)]=sha
    for f,sha in p['input_sha256'].items():
        if digest(f)!=sha:raise ValueError('same-state source input changed '+f)
        identity[f]=sha
    proposal=json.loads(Path(protocol).read_text());rows=samestate_inventory(proposal)
    original=json.loads((source/'inventory.json').read_text())
    for c in proposal['source_rows']:
        old=original[c['source_index']]
        if old['route']!=c['route'] or old['origin']!=c['origin'] or old['force_N']!=0. or old['switch_s']!=1.:
            raise ValueError('fixed metadata source selection mismatch')
    import numpy as np
    with np.load(source/'evaluation/task1_host_state.npz') as host:
        indices=[c['source_index'] for c in proposal['source_rows']]
        if np.any(host['replanned'][indices]) or np.any(host['ever_arrived'][indices]):raise ValueError('source original-path/unarrived contract differs')
    files=['plan.json','seal.json','inventory.json','evaluation/status.json','evaluation/task1_complete_carry.pkl','evaluation/task1_host_state.npz','evaluation/reused_source_prefix.npz']
    files += ['evaluation/'+r+'_path.npz' for r in sorted({c['route'] for c in rows})]
    for name in files:
        f=source/name;identity[str(f)]=digest(f)
    prepare_switching(p['stay_source'],root,protocol,samestate_source=dict(source=str(source),identity=identity))

def broaden_inventory(p):
    expected_routes=[dict(route=f'S_{side}_{angle}_R{radius}',speed=speed) for angle,radius,speed in [(90,'1p8',2.8),(60,'2p2',2.6)] for side in ('left','right')]
    arms=['temporal_rank_last','task_guard_only','fixed_half','fixed1','speedcap']
    if any(p.get(k)!=v for k,v in dict(expected_cases=120,maximum_lower_ticks=240000,bank_indices=[4,6],onsets_s=[2.2],forces_N=[-64.,64.],horizon_s=10.,duration_s=.1,arms=arms,optimizer_updates=0,maximum_replans=1,force_frame='worldY',routes=expected_routes,zero_onset_s=2.2,replan_cadence_s=1.,departure_m=1.,replanning_radius_m=1.8,goal_radius_m=1.,planner_points=5001,planner_spacing_m=.02,planner_support_m=100.).items()):
        raise ValueError('bounded broaden120 contract mismatch')
    if set(p['bank_state_fingerprints'])!={'4','6'} or p['arm_to_export_index']!={a:1 for a in arms}:raise ValueError('broaden frozen identities differ')
    # Retain the exact coordinator gates independently of new inventory geometry.
    events=[dict(event='zero',force_N=0.,onset_s=2.2)]+[dict(event=f't2.2_F{force:+g}',force_N=force,onset_s=2.2) for force in (-64.,64.)]
    return [dict(**route,**event,bank_index=b,initial_fingerprint=p['bank_state_fingerprints'][str(b)],path_key=f'bank{b}_'+route['route'],case_id=f'bank{b}/{route["route"]}/{arm}/{event["event"]}',mode=arm,mode_index=i,alpha=1.,query_alpha=.5 if arm=='fixed_half' else 1.,duration_s=.1) for b,route,event,(i,arm) in itertools.product(p['bank_indices'],p['routes'],events,enumerate(arms))]


def prepare_broaden(source,root,protocol):
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text())
    if old.get('mode')!='tolerance_online40' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed temporal40 source required')
    proposal=json.loads(Path(protocol).read_text());broaden_inventory(proposal)
    if proposal['predictor_manifest']!=old['protocol']['predictor_manifest']:raise ValueError('last frozen predictor source changed')
    for key in ('coordination','rule'):
        if proposal[key]!=old['protocol'][key]:raise ValueError('frozen coordination gates changed')
    identity=dict(old['input_sha256'])
    for file,sha in identity.items():
        if digest(file)!=sha:raise ValueError('ancestor input drift '+file)
    identity[str(source/'plan.json')]=digest(source/'plan.json');identity[str(source/'seal.json')]=digest(source/'seal.json')
    prepare_switching(old['stay_source'],root,protocol,online=True,confirmation_source=dict(source=str(source),identity=identity,tolerance=True))


def user_preference_inventory(p):
    arms=['user0_last','user0_guard','user0_fixed','userhalf_last','userhalf_guard']
    if p['arms']!=arms or p['arm_to_export_index']!={a:1 for a in arms}:raise ValueError('user preference five arm identity differs')
    if p['user_preference']!=dict(supported_users=[0.,.5,1.],speed_weight='.1+.9*user',steer_weight='1-.9*user',preference_weight=.05,switch_weight=.02,initial_previous_alpha='user'):raise ValueError('user proxy contract differs')
    base=dict(p,mode='tolerance_broaden120',arms=['temporal_rank_last','task_guard_only','fixed_half','fixed1','speedcap'])
    base['arm_to_export_index']={a:1 for a in base['arms']}
    rows=broaden_inventory(base)
    for row in rows:
        i=row['mode_index'];user=0. if i<3 else .5;method='fixed' if i==2 else 'guard' if i in (1,4) else 'last'
        row.update(mode=arms[i],alpha=user,user_alpha=user,effective_user_alpha=user,query_alpha=user,method=method)
        row['case_id']=f"bank{row['bank_index']}/{row['route']}/{arms[i]}/{row['event']}"
    return rows


def prepare_user_preference(source,root,protocol):
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text())
    if old.get('mode')!='tolerance_broaden120' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed broaden120 source required')
    proposal=json.loads(Path(protocol).read_text());user_preference_inventory(proposal)
    for key in ('predictor_manifest','coordination','rule','routes','bank_indices','bank_state_fingerprints','onsets_s','forces_N','duration_s','force_frame','horizon_s','goal_radius_m','replanning_radius_m','departure_m','maximum_replans','replan_cadence_s','planner_support_m','planner_spacing_m','planner_points'):
        if proposal[key]!=old['protocol'][key]:raise ValueError('user cache condition differs '+key)
    identity=dict(old['input_sha256'])
    for file,sha in identity.items():
        if digest(file)!=sha:raise ValueError('ancestor input drift '+file)
    identity[str(source/'plan.json')]=digest(source/'plan.json');identity[str(source/'seal.json')]=digest(source/'seal.json')
    prepare_switching(old['stay_source'],root,protocol,online=True,confirmation_source=dict(source=str(source),identity=identity,tolerance=True))


def prior_online_inventory(p):
    arms=['user0_prior','userhalf_prior','user1_prior']
    if p['expected_cases']!=72 or p['maximum_lower_ticks']!=144000 or p['arms']!=arms or p['arm_to_export_index']!={a:0 for a in arms}:raise ValueError('fixed prior online72 contract mismatch')
    base=dict(p,expected_cases=120,maximum_lower_ticks=240000,arms=['user0_last','user0_guard','user0_fixed','userhalf_last','userhalf_guard'])
    base['arm_to_export_index']={a:1 for a in base['arms']}
    conditions={}
    for row in user_preference_inventory(base):conditions.setdefault((row['bank_index'],row['route'],row['event']),row)
    rows=[]
    for row in conditions.values():
        for i,user in enumerate((0.,.5,1.)):
            own=dict(row,mode=arms[i],mode_index=i,alpha=user,user_alpha=user,effective_user_alpha=user,query_alpha=user,method='prior_last')
            own['case_id']=f"bank{row['bank_index']}/{row['route']}/{arms[i]}/{row['event']}";rows.append(own)
    return rows


def prepare_prior_online(source,root,protocol):
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text());proposal=json.loads(Path(protocol).read_text());prior_online_inventory(proposal)
    if old.get('mode')!='user_preference120' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed user preference120 source required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=120:raise ValueError('completed prior comparator required')
    for key in ('coordination','rule','routes','bank_indices','bank_state_fingerprints','onsets_s','forces_N','duration_s','force_frame','horizon_s','goal_radius_m','replanning_radius_m','departure_m','maximum_replans','replan_cadence_s','planner_support_m','planner_spacing_m','planner_points','user_preference'):
        if proposal[key]!=old['protocol'][key]:raise ValueError('prior confirmation changes matched condition '+key)
    identity=dict(old['input_sha256'])
    for file,sha in identity.items():
        if digest(file)!=sha:raise ValueError('prior comparator input changed '+file)
    identity[str(source/'plan.json')]=digest(source/'plan.json');identity[str(source/'seal.json')]=digest(source/'seal.json')
    coordinator=file_module('_sttw_prior_freeze',Path(__file__).with_name('tolerance_coordination.py'))
    coordinator.freeze_user_mix_last(proposal['predictor_training'],proposal['predictor_prepared'],proposal['predictor_manifest'])
    prepare_switching(old['stay_source'],root,protocol,online=True,confirmation_source=dict(source=str(source),identity=identity,tolerance=True))


def task_aware_inventory(p):
    arms=['user0_taskaware','userhalf_taskaware','user1_taskaware']
    if p['arms']!=arms or p['arm_to_export_index']!={a:0 for a in arms} or p.get('task_aware_fallback') is not True:raise ValueError('task aware fallback identity differs')
    base=dict(p,arms=['user0_prior','userhalf_prior','user1_prior']);base['arm_to_export_index']={a:0 for a in base['arms']}
    rows=prior_online_inventory(base)
    for row in rows:
        row.update(mode=arms[row['mode_index']],method='task_aware')
        row['case_id']=f"bank{row['bank_index']}/{row['route']}/{row['mode']}/{row['event']}"
    return rows


def prepare_task_aware(source,root,protocol):
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text());proposal=json.loads(Path(protocol).read_text());task_aware_inventory(proposal)
    if old.get('mode')!='user_prior_online72' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed prior online72 comparator required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=72:raise ValueError('completed prior72 comparator required')
    for key in ('coordination','rule','routes','bank_indices','bank_state_fingerprints','onsets_s','forces_N','duration_s','force_frame','horizon_s','goal_radius_m','replanning_radius_m','departure_m','maximum_replans','replan_cadence_s','planner_support_m','planner_spacing_m','planner_points','user_preference','predictor_training','predictor_prepared'):
        if proposal[key]!=old['protocol'][key]:raise ValueError('fallback changes matched condition '+key)
    identity=dict(old['input_sha256'])
    for file,sha in identity.items():
        if digest(file)!=sha:raise ValueError('prior comparator input changed '+file)
    identity[str(source/'plan.json')]=digest(source/'plan.json');identity[str(source/'seal.json')]=digest(source/'seal.json')
    if Path(proposal['predictor_manifest']).resolve()!=(source/'q/original_candidate_manifest.json').resolve():raise ValueError('fallback must reuse unchanged prior candidate')
    prepare_switching(old['stay_source'],root,protocol,online=True,confirmation_source=dict(source=str(source),identity=identity,tolerance=True))


def frozen_confirmation_variant(p):
    variant=p.get('variant')
    if variant not in (None,'geometry_duration_extension_v1'):raise ValueError('unknown frozen confirmation variant')
    return variant


def frozen_confirmation_inventory(p):
    variant=frozen_confirmation_variant(p)
    users=[('user0',0.),('userhalf',.5),('user1',1.)]
    methods=[('taskaware','task_aware',0),('guard','guard',1),('fixed','fixed',2)]
    declarations=[(f'{name}_{suffix}',user,method,code) for name,user in users for suffix,method,code in methods]+[('user1_speedcap',1.,'speedcap',3)]
    arms=[x[0] for x in declarations]
    routes=[dict(route=f'S_{side}_90_R1p8',speed=2.8) for side in ('left','right')]
    cases,budget,onsets,zero,duration=200,400000,[1.6,3.4],1.6,.1
    if variant=='geometry_duration_extension_v1':
        contexts=[dict(route=f'S_{side}_{angle}_R2p2',angle_deg=angle,radius_m=2.2,speed=2.6,direction=direction) for angle in (60,120) for side,direction in (('left',1.),('right',-1.))]
        if p.get('route_contexts')!=contexts:raise ValueError('explicit four geometry contexts differ')
        routes=[dict(route=c['route'],speed=c['speed']) for c in contexts]
        cases,budget,onsets,zero,duration=240,480000,[2.6],2.6,.15
    expected=dict(expected_cases=cases,maximum_lower_ticks=budget,horizon_s=10.,optimizer_updates=0,bank_indices=[1,7],forces_N=[-64.,64.],onsets_s=onsets,zero_onset_s=zero,duration_s=duration,force_frame='worldY',routes=routes,goal_radius_m=1.,replanning_radius_m=1.8,departure_m=1.,maximum_replans=1,replan_cadence_s=1.,planner_support_m=100.,planner_spacing_m=.02,planner_points=5001,arms=arms)
    if any(p.get(k)!=v for k,v in expected.items()) or set(p['bank_state_fingerprints'])!={'1','7'} or p['arm_to_export_index']!={a:0 for a in arms}:raise ValueError('frozen confirmation contract differs')
    events=[dict(event='zero',force_N=0.,onset_s=zero)]+[dict(event=f't{t:g}_F{force:+g}',force_N=force,onset_s=t) for t,force in itertools.product(onsets,(-64.,64.))]
    return [dict(**route,**event,bank_index=b,initial_fingerprint=p['bank_state_fingerprints'][str(b)],path_key=f'bank{b}_'+route['route'],case_id=f'bank{b}/{route["route"]}/{arm}/{event["event"]}',mode=arm,mode_index=i,alpha=user,user_alpha=user,effective_user_alpha=user,query_alpha=user,method=method,method_index=code,duration_s=duration) for b,route,event,(i,(arm,user,method,code)) in itertools.product(p['bank_indices'],routes,events,enumerate(declarations))]


def apply_route_context_variant(plan):
    if plan['mode']=='frozen_transfer_evaluation':
        frozen_transfer_inventory(plan['protocol']);plan['contexts']=copy.deepcopy(plan['protocol']['route_contexts'])
        return plan
    if plan['mode'] in ('frozen_confirmation200','guard_fallback_evaluation','frozen_transfer_evaluation') and frozen_confirmation_variant(plan['protocol'])=='geometry_duration_extension_v1':
        (guard_fallback_inventory if plan['mode']=='guard_fallback_evaluation' else frozen_confirmation_inventory)(plan['protocol'])
        plan['contexts']=copy.deepcopy(plan['protocol']['route_contexts'])
        plan['geometry_protocol_variant']='geometry_duration_extension_v1'
    return plan


def frozen_control_core_identity(current,archived):
    """Compare deployed control functions plus recursively called local helpers."""
    import ast
    def functions(path):
        tree=ast.parse(Path(path).read_text());return {node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
    roots={'infer_probabilities','native_offsets','candidate_features','candidate_probabilities','preview_curvature','choose_alpha','coordinate'}
    maps=[functions(current),functions(archived)];names=set(roots)
    for funcs in maps:
        pending=list(roots);visited=set()
        while pending:
            name=pending.pop()
            if name in visited:continue
            if name not in funcs:raise ValueError('frozen control function missing '+name)
            visited.add(name)
            for node in ast.walk(funcs[name]):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in funcs and node.func.id not in visited:pending.append(node.func.id)
        names|=visited
    digests=[{name:hashlib.sha256(ast.dump(funcs[name],include_attributes=False).encode()).hexdigest() for name in sorted(names)} for funcs in maps]
    if digests[0]!=digests[1]:raise ValueError('frozen executable control core differs')
    return dict(function_ast_sha256=digests[0],current_file_sha256=digest(current),archived_file_sha256=digest(archived),whole_files_equal=digest(current)==digest(archived),scope='control functions and recursively called local helpers; exporter-only file differences permitted')


TRANSFER_METHOD_CODES={'guard_degraded':0,'guard':1,'fixed':2,'speedcap':3}


def transfer_events(p):
    events=([dict(event='zero',force_N=0.,onset_s=p['zero_onset_s'])] if p.get('include_nominal',True) else [])
    return events+[dict(event=f't{t:g}_F{force:+g}',force_N=force,onset_s=t) for t,force in itertools.product(p['onsets_s'],p['forces_N'])]


def scheduled_chunk_steps(p):
    """Split only the ordinary host interval containing a declared timing probe."""
    t=p.get('scheduled_replan_s')
    if t is None:return None
    if p.get('mode')!='frozen_transfer_evaluation' or not math.isfinite(t) or not 0<t<p['horizon_s'] or abs(t/.02-round(t/.02))>1e-8:
        raise ValueError('scheduled replanning must be a finite interior upper tick in transfer evaluation')
    ends=sorted(set([round(v/.02) for v in range(1,int(p['horizon_s'])+1)]+[round(t/.02)]))
    return [b-a for a,b in zip([0]+ends[:-1],ends)]


def frozen_transfer_inventory(p):
    """Explicit cartesian protocol; no source episodes are cached as new data."""
    declarations=p['method_declarations'];arms=p['arms'];routes=p['routes'];banks=p['bank_indices']
    if not declarations or [d['arm'] for d in declarations]!=arms or len(set(arms))!=len(arms):raise ValueError('unique declared transfer arms required')
    if len(declarations)!=4 or {d['method'] for d in declarations}!=set(TRANSFER_METHOD_CODES) or any(d['user_alpha']!=1. for d in declarations):raise ValueError('declared user1 four-method transfer required')
    if len(set(banks))!=len(banks) or not banks or set(p['bank_state_fingerprints'])!={str(b) for b in banks}:raise ValueError('unique authenticated transfer banks required')
    if not routes or len({r['route'] for r in routes})!=len(routes):raise ValueError('unique declared transfer routes required')
    contexts=p['route_contexts']
    if [(c['route'],c['speed']) for c in contexts]!=[(r['route'],r['speed']) for r in routes] or any(c['direction'] not in (-1.,1.) or c['angle_deg']<=0 or c['radius_m']<=0 or c['speed']<=0 for c in contexts):raise ValueError('explicit transfer geometry required')
    events=transfer_events(p);scheduled_chunk_steps(p)
    if len({(e['force_N'],e['onset_s']) for e in events})!=len(events) or any(e['onset_s']<0 or e['onset_s']+p['duration_s']>p['horizon_s'] for e in events):raise ValueError('unique finite transfer events within horizon required')
    count=len(banks)*len(routes)*len(events)*len(arms)
    if p['horizon_s']!=10. or p['expected_cases']!=count or p['maximum_lower_ticks']!=count*2000 or p['optimizer_updates']!=0 or p['force_frame']!='worldY' or p['arm_to_export_index']!={a:0 for a in arms}:raise ValueError('transfer cartesian count/budget/q identity differs')
    if any(p[k]!=v for k,v in dict(goal_radius_m=1.,replanning_radius_m=1.8,departure_m=1.,maximum_replans=1,replan_cadence_s=1.,planner_points=5001,planner_spacing_m=.02,planner_support_m=100.).items()):raise ValueError('frozen transfer planner/endpoint contract differs')
    return [dict(**route,**event,bank_index=b,initial_fingerprint=p['bank_state_fingerprints'][str(b)],path_key=f'bank{b}_'+route['route'],case_id=f"bank{b}/{route['route']}/{d['arm']}/{event['event']}",mode=d['arm'],mode_index=i,alpha=d['user_alpha'],user_alpha=d['user_alpha'],effective_user_alpha=d['user_alpha'],query_alpha=d['user_alpha'],method=d['method'],method_index=TRANSFER_METHOD_CODES[d['method']],duration_s=p['duration_s']) for b,route,event,(i,d) in itertools.product(banks,routes,events,enumerate(declarations))]


def declared_confirmation_methods(cases,arms):
    """Generate controller dispatch from authenticated inventory, including legacy."""
    mapping={}
    for row in cases:
        pair=(row['method'],row['method_index'])
        if row['mode'] in mapping and mapping[row['mode']]!=pair:raise ValueError('inconsistent inventory method declaration')
        mapping[row['mode']]=pair
    if set(mapping)!=set(arms):raise ValueError('inventory methods do not cover declared arms')
    codes={'task_aware':0,**TRANSFER_METHOD_CODES}
    if any(method not in codes or codes[method]!=code for method,code in mapping.values()):raise ValueError('unsupported inventory method dispatch')
    return [mapping[a][1] for a in arms]


def prepared_bank_fingerprints(plan,indices):
    """Read full native snapshots in an isolated process; zero environment steps."""
    import subprocess
    code="""import sys,pickle,json,hashlib,numpy as np
sys.path.insert(0,sys.argv[1]);import jax
with open(sys.argv[2],'rb') as f:bank=pickle.load(f)
result={}
for index in json.loads(sys.argv[3]):
 selected=jax.tree.map(lambda x:np.asarray(x)[index],bank)
 if bool(selected.failed):raise ValueError('prepared snapshot failed')
 h=hashlib.sha256()
 for path,value in jax.tree_util.tree_flatten_with_path(selected)[0]:
  a=np.asarray(value);h.update(jax.tree_util.keystr(path).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
 result[str(index)]=h.hexdigest()
print(json.dumps(result))
"""
    source=plan['initial_source']
    if digest(source['prepared_bank'])!=source['prepared_bank_sha256']:raise ValueError('prepared bank file changed')
    return json.loads(subprocess.check_output([sys.executable,'-B','-c',code,plan['native_runtime_root'],source['prepared_bank'],json.dumps(indices)],text=True))


def prepare_frozen_transfer(source,root,protocol):
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text());proposal=json.loads(Path(protocol).read_text());frozen_transfer_inventory(proposal)
    if old.get('mode')!='guard_fallback_evaluation' or old['protocol'].get('stage')!='extension' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed current guard-fallback extension source required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=old['cases']:raise ValueError('completed frozen control source required')
    if old['protocol'].get('guard_fallback') is not True:raise ValueError('source guard fallback is not enabled')
    for key in ('coordination','rule','force_frame','horizon_s','goal_radius_m','replanning_radius_m','departure_m','maximum_replans','replan_cadence_s','planner_support_m','planner_spacing_m','planner_points','user_preference','predictor_manifest','predictor_training','predictor_prepared'):
        if proposal[key]!=old['protocol'][key]:raise ValueError('transfer changes frozen controller '+key)
    actual=prepared_bank_fingerprints(old,proposal['bank_indices'])
    if actual!=proposal['bank_state_fingerprints']:raise ValueError('transfer complete prepared fingerprints differ')
    archived=source/'source/tolerance_coordination.py'
    entries=[sha for file,sha in old['source_sha256'].items() if Path(file).name=='tolerance_coordination.py']
    if entries!=[digest(archived)]:raise ValueError('source frozen coordinator identity changed')
    core=frozen_control_core_identity(Path(__file__).with_name('tolerance_coordination.py'),archived)
    identity=dict(old['input_sha256'])
    for file,sha in identity.items():
        if digest(file)!=sha:raise ValueError('frozen transfer source input drift '+file)
    for name in ('plan.json','seal.json','evaluation/status.json'):identity[str(source/name)]=digest(source/name)
    prepare_switching(old['stay_source'],root,protocol,online=True,confirmation_source=dict(source=str(source),identity=identity,tolerance=True,frozen_candidate_only=True,frozen_core_identity=core,prepared_bank_fingerprints=actual))


def prepare_frozen_confirmation(source,root,protocol):
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text());proposal=json.loads(Path(protocol).read_text());frozen_confirmation_inventory(proposal)
    if old.get('mode')!='task_aware_fallback72' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed task-aware candidate required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=72:raise ValueError('completed candidate source required')
    kernel_entries=[sha for file,sha in old['source_sha256'].items() if Path(file).name=='tolerance_coordination.py']
    core_identity=None
    if frozen_confirmation_variant(proposal)=='geometry_duration_extension_v1':
        archived=source/'source/tolerance_coordination.py'
        if len(kernel_entries)!=1 or digest(archived)!=kernel_entries[0]:raise ValueError('archived frozen control source changed')
        core_identity=frozen_control_core_identity(Path(__file__).with_name('tolerance_coordination.py'),archived)
    elif len(kernel_entries)!=1 or digest(Path(__file__).with_name('tolerance_coordination.py'))!=kernel_entries[0]:raise ValueError('frozen candidate kernel changed before confirmation')
    frozen_keys=('coordination','rule','force_frame','horizon_s','goal_radius_m','replanning_radius_m','departure_m','maximum_replans','replan_cadence_s','planner_support_m','planner_spacing_m','planner_points','user_preference','predictor_training','predictor_prepared')
    if frozen_confirmation_variant(proposal) is None:frozen_keys+=('duration_s',)
    for key in frozen_keys:
        if proposal[key]!=old['protocol'][key]:raise ValueError('frozen confirmation changes controller contract '+key)
    identity=dict(old['input_sha256'])
    for file,sha in identity.items():
        if digest(file)!=sha:raise ValueError('candidate input changed '+file)
    for name in ('plan.json','seal.json','evaluation/status.json'):
        identity[str(source/name)]=digest(source/name)
    prepare_switching(old['stay_source'],root,protocol,online=True,confirmation_source=dict(source=str(source),identity=identity,tolerance=True,frozen_candidate_only=True,frozen_core_identity=core_identity))


def select_confirmation_arm(method,decision,permission,prev_alpha,prev_latch,user_alpha):
    import jax.numpy as j
    fixed=method>=2
    alpha=j.where(fixed,j.where(permission,user_alpha,prev_alpha),decision['alpha'])
    latch=j.where(fixed,j.where(permission,False,prev_latch),decision['latched'])
    return alpha,latch


def speed_query_inventory(p):
    arms=['user0_speedquery','userhalf_speedquery','user1_speedquery']
    if p['expected_cases']!=60 or p['maximum_lower_ticks']!=120000 or p['arms']!=arms or p['arm_to_export_index']!={a:0 for a in arms} or p.get('allow_masked_export') is not True or p.get('feature_mask_keep_indices')!=[4]+list(range(34,41)):raise ValueError('bounded speed+query online60 contract differs')
    base=dict(p,expected_cases=200,maximum_lower_ticks=400000,arms=[f'{u}_{m}' for u in ('user0','userhalf','user1') for m in ('taskaware','guard','fixed')]+['user1_speedcap'])
    base['arm_to_export_index']={a:0 for a in base['arms']}
    rows=[]
    for row in frozen_confirmation_inventory(base):
        if row['method']!='task_aware':continue
        i=int(row['alpha']*2);row.update(mode=arms[i],mode_index=i,method='speed_query',method_index=0)
        row['case_id']=f"bank{row['bank_index']}/{row['route']}/{row['mode']}/{row['event']}";rows.append(row)
    return rows


def prepare_speed_query(source,root,protocol):
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text());proposal=json.loads(Path(protocol).read_text());speed_query_inventory(proposal)
    if old.get('mode')!='frozen_confirmation200' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('sealed full-input frozen200 comparator required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=200:raise ValueError('completed frozen200 required')
    for key in ('coordination','rule','routes','bank_indices','bank_state_fingerprints','forces_N','onsets_s','zero_onset_s','duration_s','force_frame','horizon_s','goal_radius_m','replanning_radius_m','departure_m','maximum_replans','replan_cadence_s','planner_support_m','planner_spacing_m','planner_points','user_preference'):
        if proposal[key]!=old['protocol'][key]:raise ValueError('input ablation changes controller/task contract '+key)
    identity=dict(old['input_sha256'])
    for file,sha in identity.items():
        if digest(file)!=sha:raise ValueError('frozen200 input changed '+file)
    for name in ('plan.json','seal.json','evaluation/status.json'):identity[str(source/name)]=digest(source/name)
    coordinator=file_module('_sttw_masked_freeze',Path(__file__).with_name('tolerance_coordination.py'))
    coordinator.freeze_user_mix_last(proposal['predictor_training'],proposal['predictor_prepared'],proposal['predictor_manifest'],label='speed_query_rank_last',allow_masked_export=True)
    prepare_switching(old['stay_source'],root,protocol,online=True,confirmation_source=dict(source=str(source),identity=identity,tolerance=True))


def guard_fallback_inventory(p):
    stage=p.get('stage');extension=stage=='extension'
    if stage not in ('extension','regression'):raise ValueError('unknown guard fallback stage')
    arms=['user0_guardfallback','userhalf_guardfallback','user1_guardfallback']
    count,budget=(72,144000) if extension else (60,120000)
    if p['expected_cases']!=count or p['maximum_lower_ticks']!=budget or p['arms']!=arms or p['arm_to_export_index']!={a:0 for a in arms} or p.get('guard_fallback') is not True or p.get('maximum_campaign_lower_ticks')!=264000:raise ValueError('bounded staged guard fallback contract differs')
    if p.get('variant')!=('geometry_duration_extension_v1' if extension else None):raise ValueError('guard fallback stage geometry differs')
    base=dict(p,expected_cases=240 if extension else 200,maximum_lower_ticks=480000 if extension else 400000,arms=[f'{u}_{m}' for u in ('user0','userhalf','user1') for m in ('taskaware','guard','fixed')]+['user1_speedcap'])
    base['arm_to_export_index']={a:0 for a in base['arms']}
    rows=[]
    for row in frozen_confirmation_inventory(base):
        if row['method']!='task_aware':continue
        i=int(row['alpha']*2);row.update(mode=arms[i],mode_index=i,method='guard_degraded',method_index=0)
        row['case_id']=f"bank{row['bank_index']}/{row['route']}/{row['mode']}/{row['event']}";rows.append(row)
    return rows


def prepare_guard_fallback(source,root,protocol):
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text());proposal=json.loads(Path(protocol).read_text());guard_fallback_inventory(proposal)
    extension=proposal['stage']=='extension';source_count=240 if extension else 200
    if old.get('mode')!='frozen_confirmation200' or old['protocol'].get('variant')!=proposal.get('variant') or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('matching frozen guard fallback comparator required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=source_count:raise ValueError('complete frozen comparator required')
    for key in ('coordination','rule','routes','bank_indices','bank_state_fingerprints','forces_N','onsets_s','zero_onset_s','duration_s','force_frame','horizon_s','goal_radius_m','replanning_radius_m','departure_m','maximum_replans','replan_cadence_s','planner_support_m','planner_spacing_m','planner_points','user_preference','predictor_training','predictor_prepared'):
        if proposal[key]!=old['protocol'][key]:raise ValueError('guard fallback changes fixed contract '+key)
    if extension and proposal['route_contexts']!=old['protocol']['route_contexts']:raise ValueError('guard fallback route contexts differ')
    identity=dict(old['input_sha256'])
    for file,sha in identity.items():
        if digest(file)!=sha:raise ValueError('guard fallback frozen input changed '+file)
    for name in ('plan.json','seal.json','evaluation/status.json'):identity[str(source/name)]=digest(source/name)
    stage_gate=None
    if not extension:
        helper=file_module('_sttw_guard_stage_gate',Path('learning/cli/audit_native_goal_initial.py').resolve())
        stage_gate=helper.guard_fallback_stage1_gate(proposal['stage1_run'])
        if not stage_gate['allowed']:raise ValueError('stage1 has no goal/contact rescue; stage2 not authorized by declared stop rule')
        identity.update(stage_gate['identity'])
    prepare_switching(old['stay_source'],root,protocol,online=True,confirmation_source=dict(source=str(source),identity=identity,tolerance=True,stage1_gate=stage_gate,intentional_control_change='guard_fallback only; historical default remainsFalse'))


def tolerance_inventory(p):
    """Preregistered online DEV panel; independent of historical100 identity."""
    if p.get('mode')=='frozen_transfer_evaluation':return frozen_transfer_inventory(p)
    if p.get('mode')=='guard_fallback_evaluation':return guard_fallback_inventory(p)
    if p.get('mode')=='speed_query_online60':return speed_query_inventory(p)
    if p.get('mode')=='frozen_confirmation200':return frozen_confirmation_inventory(p)
    if p.get('mode')=='task_aware_fallback72':return task_aware_inventory(p)
    if p.get('mode')=='user_prior_online72':return prior_online_inventory(p)
    if p.get('mode')=='user_preference120':return user_preference_inventory(p)
    if p.get('mode')=='tolerance_broaden120':return broaden_inventory(p)
    temporal=p.get('mode')=='tolerance_online40'
    expected_arms=['temporal_rank_selected','temporal_rank_last'] if temporal else ['q_rank','q_bce','task_guard_only','existing_geometry','fixed_half','fixed1']
    if (p['expected_cases']!=(40 if temporal else 120) or p['maximum_lower_ticks']!=(80000 if temporal else 240000) or
        p['bank_indices']!=[2,5] or p['onsets_s']!=[1.6,3.4] or
        p['forces_N']!=[-64.,64.] or p['horizon_s']!=10. or p['duration_s']!=.1 or
        p['arms']!=expected_arms or
        p['optimizer_updates']!=0 or p['maximum_replans']!=1 or p['force_frame']!='worldY'):
        raise ValueError('bounded tolerance120 contract mismatch')
    if (set(p['bank_state_fingerprints'])!={'2','5'} or
        p['routes']!=[dict(route='S_left_90_R1p8',speed=2.8),dict(route='S_right_90_R1p8',speed=2.8)] or
        p['replan_cadence_s']!=1. or p['departure_m']!=1. or p['replanning_radius_m']!=1.8 or
        p['goal_radius_m']!=1. or p['planner_points']!=5001 or p['planner_spacing_m']!=.02 or
        p['rule']!=dict(preview_m=1.5,engage_accel_m_s2=3.,release_accel_m_s2=2.7,active_alpha=.5)):
        raise ValueError('tolerance120 source identity mismatch')
    expected_coordination=dict(frequency_hz=50,candidate_alphas=[0.,.5,1.],
        hypothetical_forces_N=[-64.,64.],hypothetical_preparation_s=[.2,.6],hypothetical_duration_s=.1,
        aggregation='minimum of four hypotheses after equal mean of three seed sigmoid probabilities',
        task_preview_m=6.,task_accel_limit_m_s2=3.,q_enter=.8,q_restore=.85,fallback_alpha=.5,
        initial_alpha=1.,outer_alpha_slew=False,dwell_s=0.,proxy_cost_weights=[1.,.1,.05,.02])
    if p['coordination']!=expected_coordination:raise ValueError('frozen coordination contract differs')
    if temporal and p['arm_to_export_index']!={'temporal_rank_selected':0,'temporal_rank_last':1}:raise ValueError('two rank ensemble identity mismatch')
    events=[dict(event='zero',force_N=0.,onset_s=p['zero_onset_s'])]
    events += [dict(event=f't{t:g}_F{f:+g}',force_N=f,onset_s=t) for t,f in itertools.product(p['onsets_s'],p['forces_N'])]
    return [dict(**route,**event,bank_index=b,initial_fingerprint=p['bank_state_fingerprints'][str(b)],
        path_key=f'bank{b}_'+route['route'],case_id=f'bank{b}/{route["route"]}/{arm}/{event["event"]}',
        mode=arm,mode_index=i,alpha=1.,query_alpha=.5 if arm=='fixed_half' else 1.,duration_s=.1)
        for b,route,event,(i,arm) in itertools.product(p['bank_indices'],p['routes'],events,enumerate(p['arms']))]


def select_tolerance_arm(mode,decision,geometric_alpha,geometric_latch,permission,prev_alpha,prev_latch):
    """Select only scalar preference; original native publication remains external."""
    import jax.numpy as j
    alpha=j.where(mode<3,decision['alpha'],j.where(mode==3,geometric_alpha,j.where(mode==4,.5,1.)))
    latch=j.where(mode<3,decision['latched'],j.where(mode==3,geometric_latch,False))
    return j.where(permission,alpha,prev_alpha),j.where(permission,latch,prev_latch)


def select_broaden_arm(mode,decision,permission,prev_alpha,prev_latch):
    """Five explicit protocol indices: last, guard, half, one, speedcap."""
    import jax.numpy as j
    proposed=j.where(mode<2,decision['alpha'],j.where(mode==2,.5,1.))
    latch=j.where(mode<2,decision['latched'],False)
    return j.where(permission,proposed,prev_alpha),j.where(permission,latch,prev_latch)


def validate_temporal_comparison_contract(proposal,cached,config,initial_source,models,contexts):
    if cached.get('mode')!='tolerance_online120' or cached['cases']!=120:
        raise ValueError('original online120 six-arm comparison source required')
    if cached['config']!=config or cached['initial_source']!=initial_source or cached['models']!=models or cached['contexts']!=contexts:
        raise ValueError('cached physics/controller/initial/model/path identity differs')
    for key in ('bank_indices','bank_state_fingerprints','routes','forces_N','onsets_s','zero_onset_s','duration_s','force_frame','horizon_s','goal_radius_m','replanning_radius_m','departure_m','maximum_replans','replan_cadence_s','planner_support_m','planner_spacing_m','planner_points','rule','coordination'):
        if proposal[key]!=cached['protocol'][key]:raise ValueError('cached comparison condition differs '+key)
    return True


def prepare_temporal_online(source,root,protocol):
    """Two new rank ensembles; authenticate existing120 without new baseline physics."""
    source=Path(source).resolve();p=json.loads((source/'plan.json').read_text())
    proposal=json.loads(Path(protocol).read_text());tolerance_inventory(proposal)
    if proposal.get('mode')!='tolerance_online40' or Path(proposal['comparison_source']).resolve()!=source:
        raise ValueError('explicit temporal40 protocol/comparison source required')
    validate_temporal_comparison_contract(proposal,p,p['config'],p['initial_source'],p['models'],p['contexts'])
    if p.get('mode')!='tolerance_online120' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:
        raise ValueError('sealed online120 comparison source required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=120:raise ValueError('completed online120 required')
    identity=dict(p['input_sha256'])
    for f,sha in p['source_sha256'].items():
        archived=source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('comparison archived source drift')
        identity[str(archived)]=sha
    records=json.loads((source/'evaluation/case_index.json').read_text())
    if len(records)!=120 or {r['mode'] for r in records}!=set(p['protocol']['arms']):raise ValueError('complete original six comparison methods required')
    for r in records:identity[r['raw_file']]=r['raw_sha256']
    for name in ('plan.json','seal.json','inventory.json','evaluation/status.json','evaluation/case_index.json','analysis/results.json','analysis/receipt.json','evaluation/replanning_decisions.json'):
        identity[str(source/name)]=digest(source/name)
    for route in {c['path_key'] for c in records}:
        identity[str(source/'evaluation'/(route+'_path.npz'))]=digest(source/'evaluation'/(route+'_path.npz'))
    for path in (source/'evaluation').glob('replanned_path_*.npz'):identity[str(path)]=digest(path)
    for file,sha in identity.items():
        if digest(file)!=sha:raise ValueError('cached comparison identity drift '+file)
    ancestor=Path(p['confirmation_source']['source'])
    ancestor_identity=dict(p['confirmation_source']['identity'])
    for name in ('plan.json','prefix/initial_carry.pkl'):ancestor_identity[str(ancestor/name)]=digest(ancestor/name)
    comparison=dict(source=str(source),expected_cases=120,cached_methods=p['protocol']['arms'],new_methods=proposal['arms'],total_records=160,identity=identity,reuse_physics_only=True)
    prepare_switching(p['stay_source'],root,protocol,online=True,confirmation_source=dict(source=str(ancestor),identity={**ancestor_identity,**identity},tolerance=True,comparison=comparison))


def prepare_tolerance(source,root,protocol):
    """Authenticate new-bank ancestor; never reuse its task1 suffix physics."""
    source=Path(source).resolve();p=json.loads((source/'plan.json').read_text())
    if p.get('mode')!='h75_samestate_bank144' or not new_confirmation_sources(p['protocol']):
        raise ValueError('new-source confirmation144 ancestor required')
    if digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:
        raise ValueError('confirmation144 seal mismatch')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=144:
        raise ValueError('completed confirmation144 required')
    proposal=json.loads(Path(protocol).read_text());tolerance_inventory(proposal)
    if any(proposal[k]!=p['protocol'][k] for k in ('bank_indices','bank_state_fingerprints','routes')):
        raise ValueError('new source bank/path identity differs')
    identity=dict(p['input_sha256'])
    for f,sha in p['source_sha256'].items():
        archived=source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('confirmation archived source mismatch')
        identity[str(archived)]=sha
    for f,sha in identity.items():
        if digest(f)!=sha:raise ValueError('confirmation input drift '+f)
    for name in ('plan.json','seal.json','evaluation/status.json','prefix/initial_carry.pkl'):
        f=source/name;identity[str(f)]=digest(f)
    prepare_switching(p['stay_source'],root,protocol,online=True,
        confirmation_source=dict(source=str(source),identity=identity,tolerance=True))


def confirmation_inventory(p):
    base=dict(p,expected_cases=18,maximum_lower_ticks=36000,forces_N=[-64.,-32.,32.,64.],onsets_s=[1.6,4.],zero_onset_s=1.6)
    online_inventory(base)
    if (p['expected_cases']!=100 or p['maximum_lower_ticks']!=200000 or p['bank_indices']!=[0,1] or
        p['forces_N']!=[-64.,64.] or p['onsets_s']!=[2.2,3.4] or p['zero_onset_s']!=2.2 or
        p['arms']!=['online_scalar','fixed0','fixed_half','fixed1','zero_with_existing_speed_floor'] or p['existing_speed_floor_m_s']!=1.5):
        raise ValueError('bounded confirmation100 contract mismatch')
    events=[dict(event='zero',force_N=0.,onset_s=p['zero_onset_s'])]
    events += [dict(event=f't{t:g}_F{f:+g}',force_N=f,onset_s=t) for t,f in itertools.product(p['onsets_s'],p['forces_N'])]
    return [dict(**route,**event,bank_index=b,initial_fingerprint=p['bank_state_fingerprints'][str(b)],
        path_key=f'bank{b}_'+route['route'],case_id=f'bank{b}/{route["route"]}/{arm}/{event["event"]}',
        mode=arm,mode_index=i,alpha=1.,query_alpha=(0. if i==1 else .5 if i==2 else 1.),duration_s=p['duration_s'])
        for b,route,event,(i,arm) in itertools.product(p['bank_indices'],p['routes'],events,enumerate(p['arms']))]


def snapshot_fingerprint(snapshot):
    import jax
    import numpy as np
    sha=hashlib.sha256()
    for path,value in jax.tree_util.tree_flatten_with_path(snapshot)[0]:
        a=np.asarray(value);sha.update(jax.tree_util.keystr(path).encode());sha.update(str(a.dtype).encode());sha.update(a.tobytes())
    return sha.hexdigest()


def existing_floor_latent(nominal,config,floor=1.5):
    """Exact existing profile7 zero-neural support mapping, native scales only."""
    import jax.numpy as j
    desired=j.maximum(0.,floor-nominal[0])
    scale=config['residual_interface']['speed_positive_scale_m_s']
    bounded=j.clip(desired/scale,-1.+1e-6,1.-1e-6)
    latent=j.stack([j.arctanh(bounded),j.asarray(0.)])
    realized=bounded*scale
    return latent,dict(support_floor=j.asarray(floor),support_requested_offset=desired,
        support_realized_target_offset=realized,support_active=realized>0.)


def case_path_key(case):
    return case.get('path_key',case['route'])



def load_confirmation_environment(cfg,source):
    """Native constructor/reset inputs, one bank read; no physical integration."""
    import jax
    import jax.numpy as j
    import numpy as np
    from dataclasses import asdict
    from sttw_control.path_command_env import PathCommandEnv
    from sttw_control.frozen_local_lower import FrozenLocalLower
    from sttw_control.lower_command import SCALES
    from sttw_control.path_command_training import config_hash
    candidate=source['candidate']
    if digest(candidate['config'])!=candidate['config_file_sha256']:raise ValueError('lower config changed')
    np.testing.assert_array_equal(np.asarray(SCALES),np.asarray(candidate['normalizer']['scales'],np.float32))
    for name in ('actor','checkpoint'):
        if digest(candidate[name])!=candidate[name+'_sha256']:raise ValueError('lower model identity changed')
    with Path(candidate['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
    spec=json.loads((Path(source['upper_run'])/'alpha0/frozen_config_stage52.json').read_text())
    env=PathCommandEnv(cfg,spec,FrozenLocalLower(params),'terminal_contract_v2')
    for key,live in [('model',env.physics.bundle.identity),('controller',asdict(env.cc)),('actuator',asdict(env.ac))]:
        if config_hash(live)!=config_hash(source[key]):raise ValueError('native physical identity mismatch '+key)
    if digest(source['prepared_bank'])!=source['prepared_bank_sha256']:raise ValueError('bank identity changed')
    with Path(source['prepared_bank']).open('rb') as f:bank=pickle.load(f)
    return env,bank

def prepare_confirmation(source,root,protocol):
    source=Path(source).resolve();p=json.loads((source/'plan.json').read_text())
    if p.get('mode')!='h75_online_scalar' or digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:
        raise ValueError('sealed completedonline18 source required')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=18:raise ValueError('completedonline18 source required')
    identity=dict(p['input_sha256'])
    for f,sha in p['source_sha256'].items():
        archived=source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('online archived source mismatch')
        identity[str(archived)]=sha
    for f,sha in identity.items():
        if digest(f)!=sha:raise ValueError('online source input mismatch '+f)
    for name in ('plan.json','seal.json','evaluation/status.json'):
        f=source/name;identity[str(f)]=digest(f)
    confirmation_inventory(json.loads(Path(protocol).read_text()))
    # Preserve the authenticated native ancestor chain, without reusing trajectories.
    prepare_switching(p['stay_source'],root,protocol,online=True,confirmation_source=dict(source=str(source),identity=identity))

def continuation_inventory(p):
    original=dict(p,maximum_lower_ticks=72000)
    rows=switching_inventory(original)
    if (p['maximum_lower_ticks']!=64800 or p['total_campaign_lower_ticks']!=79200 or
        p['consumed_source_lower_ticks']!=14400 or p['continuation_task_s']!=1. or p['new_horizon_s']!=9. or
        p['prefix_comparison'].get('field_absolute_tolerance')!={'phi_dot':1e-4} or not p['prefix_comparison'].get('floating_descriptive_only')):
        raise ValueError('bounded continuation amendment mismatch')
    return rows


def prefix_lane_divergence(logs,cases):
    import numpy as np
    groups={};maxima={}
    for i,c in enumerate(cases):groups.setdefault((c['route'],c['origin']),[]).append(i)
    for indices in groups.values():
        for name,value in logs.items():
            a=np.asarray(value)[:,indices];b=np.asarray(value)[:,indices[0],None]
            if np.issubdtype(a.dtype,np.number):maxima[name]=max(maxima.get(name,0.),float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0.)))
    return maxima


def audit_scheduled_pairs(logs,cases):
    import numpy as np
    for i,c in enumerate(cases):
        active=np.asarray(logs['active_tick'])[:,i]
        upper_tick=(np.rint(np.asarray(logs['time'])[:,i]/.005).astype(int)//4)*4
        switched=upper_tick>=round(c['switch_s']/.005)
        pair=np.where(switched[:,None],np.array([.5,.5]),np.array([c['speed_alpha'],c['steer_alpha']]))
        if not np.array_equal(logs['channel_alpha_pair'][active,i],pair[active]) or not np.array_equal(logs['switch_active'][active,i],switched[active]):
            raise ValueError('scheduled channel pair/timing mismatch')
        if 'applied_force_impulse' in logs:
            impulse=np.asarray(logs['applied_force_impulse'])[:,i]
            force=c['force_N'];time=np.asarray(logs['time'])[:,i]
            if np.any(impulse[:,[0,2]]!=0.) or (force==0. and np.any(impulse!=0.)):
                raise ValueError('worldY pulse vector mismatch')
            if force:
                nonzero=np.any(impulse!=0.,axis=-1)
                # Native substep clock may straddle one lower boundary.
                inside=(time>=c['onset_s']-.005-1e-6)&(time<c['onset_s']+c['duration_s']+.005+1e-6)
                if np.any(nonzero&~inside) or np.any(impulse[:,1]*force<0.) or np.any(np.abs(impulse[:,1])>abs(force)*.005+1e-5):
                    raise ValueError('pulse sign/timing/impulse mismatch')
    return dict(actual_pairs_and_switch_timing_verified=True,worldY_force_vector_verified='applied_force_impulse' in logs)


def prepare_continuation(source,root,protocol,*,charged_source=None):
    source=Path(source).resolve();p=json.loads((source/'plan.json').read_text())
    if p.get('mode')!='h75_channel_switching':raise ValueError('failed switching source required')
    if digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('continuation source seal mismatch')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='error' or status.get('active_lower_ticks')!=7200 or status.get('allocated_lower_slots')!=7200:raise ValueError('exact7200 consumed source required')
    identity=dict(p['input_sha256'])
    for f,sha in p['source_sha256'].items():
        archived=source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('continuation archived source mismatch')
        identity[str(archived)]=sha
    for f,sha in identity.items():
        if digest(f)!=sha:raise ValueError('continuation source input mismatch '+f)
    for name in ('plan.json','seal.json','inventory.json','evaluation/status.json','evaluation/task1_complete_carry.pkl','evaluation/task1_host_state.npz','evaluation/chunk_00.npz'):
        f=source/name;identity[str(f)]=digest(f)
    if charged_source is None:raise ValueError('prior failed continuation accounting source required')
    charged_source=Path(charged_source).resolve();charged=json.loads((charged_source/'plan.json').read_text())
    charged_status=json.loads((charged_source/'evaluation/status.json').read_text())
    if (digest(charged_source/'plan.json')!=json.loads((charged_source/'seal.json').read_text())['plan_sha256'] or
        charged.get('mode')!='h75_switching_continuation' or Path(charged['continuation_source']['source']).resolve()!=source or
        charged_status.get('phase')!='error' or charged_status.get('active_lower_ticks')!=7200 or charged_status.get('allocated_lower_slots')!=7200):
        raise ValueError('charged failed continuation identity/count mismatch')
    for f,sha in charged['input_sha256'].items():
        if digest(f)!=sha:
            # The explicit amended protocol supersedes this one configuration
            # input. Authenticate its old bytes from the sealed embedded JSON.
            embedded=(json.dumps(charged['protocol'],indent=2)+'\n').encode()
            if Path(f).resolve()!=Path(protocol).resolve() or hashlib.sha256(embedded).hexdigest()!=sha:
                raise ValueError('charged continuation input drift '+f)
    for f,sha in charged['source_sha256'].items():
        archived=charged_source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('charged continuation archived source drift')
        identity[str(archived)]=sha
    for name in ('plan.json','seal.json','evaluation/status.json','evaluation/chunk_01.npz'):
        f=charged_source/name;identity[str(f)]=digest(f)
    proposed=json.loads(Path(protocol).read_text());rows=continuation_inventory(proposed)
    if rows!=json.loads((source/'inventory.json').read_text()):raise ValueError('original36 continuation inventory differs')
    import numpy as np
    with np.load(source/'evaluation/chunk_00.npz') as raw:
        logs={k:raw[k] for k in raw.files}
    audit_scheduled_pairs(logs,rows)
    if logs['active_tick'].shape!=(200,36) or not np.all(logs['active_tick']) or np.any(logs['switch_active']) or np.any(logs['applied_force_impulse']):
        raise ValueError('source prefix active/switch/force contract mismatch')
    for name in ('physical_failure','contact_failure','nonfinite','policy_fault','lower_fault','continuation_geometry_fault'):
        if np.any(logs[name]):raise ValueError('source prefix fault/physical event '+name)
    prepare_switching(p['stay_source'],root,protocol,continuation_source=dict(source=str(source),identity=identity,prefix_lane_maximum_absolute_difference=prefix_lane_divergence(logs,rows),charged_source=str(charged_source)))


def switch_mapping(task_tick,origin_speed,origin_steer,switch_tick,destination):
    """A query mapping change only; never edits any closed-loop state leaf."""
    import jax.numpy as j
    active=task_tick>=switch_tick
    return j.where(active,destination[0],origin_speed),j.where(active,destination[1],origin_steer),active


def assert_shared_switch_carry(carry,cases,*,lanes_per_group=9):
    import jax
    import numpy as np
    leaves=jax.tree.leaves(jax.device_get(carry));groups={}
    for i,c in enumerate(cases):groups.setdefault((c['route'],c['origin']),[]).append(i)
    if len(groups)!=4 or any(len(v)!=lanes_per_group for v in groups.values()):raise ValueError('switch capture groups mismatch')
    for indices in groups.values():
        for leaf in leaves:
            if any(not np.array_equal(leaf[indices[0]],leaf[i],equal_nan=True) for i in indices[1:]):
                raise ValueError('complete task1.0 carry differs within origin group')
    return dict(groups=4,lanes_per_group=lanes_per_group,leaf_count=len(leaves),all_leaves_exact=True,task_s=1.)


def audit_stay_prefix(logs,cases,cached,protocol,*,own_switch=False):
    import numpy as np
    matched={};maxima={};counts=[];exceeded={}
    contract=protocol['prefix_comparison']
    atol=contract.get('absolute_tolerance',1e-5);rtol=contract.get('relative_tolerance',1e-6)
    for i,c in enumerate(cases):
        key=(c['route'],c['origin'],c['force_N'])
        record=next(r for r in cached if (r['route'],r['origin'],r['force_N'])==key)
        count=round(c['switch_s']/.005) if own_switch else len(logs['time'])
        if len(logs['time'])<count:raise ValueError('insufficient own-switch prefix')
        counts.append(count)
        with np.load(record['raw_file'],allow_pickle=False) as old:
            for name in contract['fields']:
                a=np.asarray(logs[name])[:count,i];b=old[name][:count]
                floating=np.issubdtype(a.dtype,np.floating)
                difference=float(np.max(np.abs(a.astype(float)-b.astype(float)),initial=0.)) if a.shape==b.shape else float('inf')
                maxima[name]=max(maxima.get(name,0.),difference)
                equal=(np.allclose(a,b,atol=contract.get('field_absolute_tolerance',{}).get(name,atol),rtol=rtol,equal_nan=False) if floating else np.array_equal(a,b)) if a.shape==b.shape else False
                if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
                    raise ValueError('nonfinite stay-origin prefix '+name)
                if a.dtype!=b.dtype or a.shape!=b.shape:raise ValueError('stay prefix dtype/shape mismatch '+name)
                if not equal:
                    exceeded[name]=exceeded.get(name,0)+1
                    if not (floating and contract.get('floating_descriptive_only',False)):
                        raise ValueError('stay-origin prefix mismatch '+str(key)+'/'+name)
        matched[key]=record['raw_file']
    return dict(matched_stay_cases=len(matched),new_lanes=len(cases),prefix_lower_ticks=counts,
        own_switch=own_switch,all_fields_within_tolerance=not bool(exceeded),floating_descriptive_only=contract.get('floating_descriptive_only',False),exceeded_lane_field_counts=exceeded,discrete_exact=True,
        absolute_tolerance=atol,relative_tolerance=rtol,field_absolute_tolerance=contract.get('field_absolute_tolerance',{}),maximum_absolute_difference=maxima)


def prepare_switching(source,root,protocol,*,continuation_source=None,online=False,confirmation_source=None,samestate_source=None):
    source=Path(source).resolve();stay=json.loads((source/'plan.json').read_text())
    if stay.get('mode')!='h75_channel_response':raise ValueError('sealed54 response source required')
    if digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:raise ValueError('stay source seal mismatch')
    status=json.loads((source/'evaluation/status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=54:raise ValueError('complete54 stay source required')
    identity=dict(stay['input_sha256'])
    for f,sha in stay['source_sha256'].items():
        archived=source/'source'/Path(f).name
        if digest(archived)!=sha:raise ValueError('stay archived module mismatch')
        identity[str(archived)]=sha
    for f,sha in identity.items():
        if digest(f)!=sha:raise ValueError('stay input mismatch '+f)
    records=json.loads((source/'evaluation/case_index.json').read_text());selected=[]
    for r in records:
        if online:
            if r['mode']!='H75' or r['alpha'] not in (.5,1.):continue
        elif r['alpha']!=1. or r['mode'] not in ('H75','hybrid') or r['force_N'] not in (0.,-64.,64.) or r['onset_s']!=1.6:continue
        r=dict(r,origin=('fixed'+format(r['alpha'],'g')) if online else 'fixed1' if r['mode']=='H75' else 'hybrid')
        if digest(r['raw_file'])!=r['raw_sha256']:raise ValueError('stay raw mismatch')
        selected.append(r);identity[r['raw_file']]=r['raw_sha256']
    if online:
        if len(selected)!=36 or len({(r['route'],r['alpha'],r['event']) for r in selected})!=36:raise ValueError('exact36 fixed-alpha cache required')
    elif len(selected)!=12 or len({(r['route'],r['origin'],r['force_N']) for r in selected})!=12:raise ValueError('exact12 matching stays required')
    for name in ('plan.json','seal.json','evaluation/status.json','evaluation/case_index.json'):
        f=source/name;identity[str(f)]=digest(f)
    prepare_response(stay['source'],root,protocol,stay_source=dict(source=str(source),plan=stay,records=[] if confirmation_source is not None or samestate_source is not None else selected,identity=identity,online=online),continuation_source=continuation_source,confirmation_source=confirmation_source,samestate_source=samestate_source)

def extend_straight_support(path,target_points,spacing_m):
    """Keep all original geometry samples; extend only a straight terminal ray."""
    import jax.numpy as j
    import numpy as np
    extra=target_points-len(path.s)
    if extra<0:raise ValueError('source path exceeds planner support')
    if not extra:return path
    if not np.allclose(np.asarray(path.curvature)[-2:],0.,atol=1e-8):
        raise ValueError('only straight source support may extend')
    tail=j.arange(1,extra+1)*spacing_m
    return path.replace(s=j.concatenate([path.s,path.s[-1]+tail]),
        xy=j.concatenate([path.xy,path.xy[-1]+tail[:,None]*j.array([j.cos(path.heading[-1]),j.sin(path.heading[-1])])]),
        heading=j.concatenate([path.heading,j.full(extra,path.heading[-1])]),
        curvature=j.concatenate([path.curvature,j.zeros(extra)]))

def force_pulses(cases,physical_start):
    """Archived teleop pulse is a world vector, never rotated with route/body."""
    return [[physical_start+c['onset_s'],physical_start+c['onset_s']+c['duration_s'],
             *[c['force_N']*x for x in c.get('force_direction_world',[0.,1.,0.])]] for c in cases]

def channel_queries(obs,apply,speed_alpha,steer_alpha):
    """Same observation except native alpha336; one composed publication."""
    import jax.numpy as j
    half=apply(obs.at[336].set(.5));full=apply(obs.at[336].set(1.))
    return j.stack([j.where(speed_alpha==.5,half[0],full[0]),
                    j.where(steer_alpha==.5,half[1],full[1])]),half,full


def prepare_response(source,root,protocol,*,stay_source=None,continuation_source=None,confirmation_source=None,samestate_source=None):
    source=Path(source).resolve();root=Path(root).resolve();protocol=Path(protocol).resolve();storage_guard(root)
    old=json.loads((source/'plan.json').read_text())
    def authenticate(directory,plan,expected_cases):
        if digest(directory/'plan.json')!=json.loads((directory/'seal.json').read_text())['plan_sha256']:
            raise ValueError('source seal changed')
        if json.loads((directory/'evaluation/status.json').read_text()).get('phase')!='completed' or json.loads((directory/'evaluation/status.json').read_text()).get('completed_cases')!=expected_cases:
            raise ValueError('completed source required')
        for f,sha in {**plan['input_sha256'],**plan['native_source_sha256']}.items():
            if digest(f)!=sha:raise ValueError('source identity changed '+f)
        for f,sha in plan['runner_sha256'].items():
            archived=directory/'source'/Path(f).name
            if digest(archived)!=sha:raise ValueError('archived runner changed '+str(archived))
    authenticate(source,old,8)
    if old.get('mode')!='native_channel_factorial_v1':raise ValueError('native channel factorial source required')
    cached=Path(old['cached_source']);native=json.loads((cached/'plan.json').read_text());authenticate(cached,native,16)
    if native.get('mode')!='native_capability_v1':raise ValueError('native S capability required')
    proposal=json.loads(protocol.read_text())
    online=stay_source is not None and stay_source.get('online',False)
    if samestate_source is not None:
        (bank144_inventory if samestate_source.get('bank144') else samestate_inventory)(proposal)
    elif confirmation_source is not None:(tolerance_inventory if proposal.get("mode") in ("tolerance_online120","tolerance_online40","tolerance_broaden120","user_preference120","user_prior_online72","task_aware_fallback72","frozen_confirmation200","speed_query_online60","guard_fallback_evaluation","frozen_transfer_evaluation") else confirmation_inventory)(proposal)
    elif online:online_inventory(proposal)
    elif stay_source is None:validate_response_protocol(proposal)
    elif continuation_source is None:switching_inventory(proposal)
    else:continuation_inventory(proposal)
    if old['config']!=native['config'] or old['initial_source']!=native['initial_source'] or old['h75']!=native['h75']:
        raise ValueError('native source/config/initial/model mismatch')
    if old['initial_source']['bank_index']!=3:raise ValueError('native bank3 required')
    files={**old['input_sha256'],**native['input_sha256'],**old['native_source_sha256']}
    for directory in (source,cached):
        for name in ('plan.json','seal.json','evaluation/status.json','evaluation/initial_carry.pkl'):
            f=directory/name;files[str(f)]=digest(f)
    for c in native['contexts']:
        f=cached/'evaluation'/(c['route']+'_path.npz');files[str(f)]=digest(f)
    for owner in (old['h75'],old['initial_source']['candidate']):
        for key in ('actor','checkpoint'):
            if digest(owner[key])!=owner[key+'_sha256']:raise ValueError('model changed')
            files[owner[key]]=owner[key+'_sha256']
    folder=Path(__file__).resolve().parent
    modules=[folder/n for n in ('native_goal_run.py','native_goal_continuation.py','native_equilibrium_rule.py','contact_only_environment.py','goal_replanning.py','_vendor/pythonrobotics/dubins.py','_vendor/pythonrobotics/angle.py')]
    modules.append(Path('learning/cli/run_native_goal_confirmation.py').resolve())
    if proposal.get('mode') in ('user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation'):modules.append(Path('learning/cli/audit_native_goal_initial.py').resolve())
    if online or (samestate_source is not None and samestate_source.get('bank144')):modules.append(folder/'geometric_preference_schedule.py')
    if proposal.get('mode') in ('tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation'):modules.append(folder/'tolerance_coordination.py')
    if stay_source is not None:
        if any(stay_source['plan'][k]!=v for k,v in [('config',old['config']),('initial_source',old['initial_source']),('models',{'H75':old['h75']})]):raise ValueError('stay source full interface identity differs')
        if stay_source['plan']['native_runtime_root']!=old['native_runtime_root'] or stay_source['plan']['native_expected_sources']!=old['native_source_sha256']:
            raise ValueError('stay native runtime identity differs')
        files.update(stay_source['identity'])
    if continuation_source is not None:files.update(continuation_source['identity'])
    if confirmation_source is not None:files.update(confirmation_source['identity'])
    if samestate_source is not None:files.update(samestate_source['identity'])
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    for f in modules:(root/'source'/f.name).write_bytes(f.read_bytes())
    if stay_source is not None:
        save(root/'cached_stay_records.json',stay_source['records']);files[str(root/'cached_stay_records.json')]=digest(root/'cached_stay_records.json')
    if proposal.get('mode') in ('tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','speed_query_online60'):
        coordinator=file_module('_sttw_tolerance_export',folder/'tolerance_coordination.py')
        coordinator.export_predictors(proposal['predictor_manifest'],root/'q',allow_masked_export=proposal.get('mode')=='speed_query_online60')
        files.update({str(f.resolve()):digest(f) for f in (root/'q').rglob('*') if f.is_file()})
    save(root/'inventory.json',(bank144_inventory(proposal) if samestate_source.get('bank144') else samestate_inventory(proposal)) if samestate_source is not None else (tolerance_inventory(proposal) if proposal.get("mode") in ("tolerance_online120","tolerance_online40","tolerance_broaden120","user_preference120","user_prior_online72","task_aware_fallback72","frozen_confirmation200","speed_query_online60","guard_fallback_evaluation","frozen_transfer_evaluation") else confirmation_inventory(proposal)) if confirmation_source is not None else online_inventory(proposal) if online else continuation_inventory(proposal) if continuation_source is not None else response_inventory(proposal) if stay_source is None else switching_inventory(proposal));save(root/'effective_config.json',old['config'])
    if proposal.get('mode') in ('h75_late_state_tolerance48','h75_user_late_state192','frozen_confirmation200','guard_fallback_evaluation','frozen_transfer_evaluation'):
        import shutil
        original_q=Path(confirmation_source['source'] if proposal.get('mode') in ('frozen_confirmation200','guard_fallback_evaluation','frozen_transfer_evaluation') else samestate_source['source'])/'q'
        receipt=json.loads((original_q/'receipt.json').read_text())
        for name,sha in receipt['files_sha256'].items():
            if digest(original_q/name)!=sha:raise ValueError('frozen online q bundle drift')
        shutil.copytree(original_q,root/'q')
        files.update({str(f.resolve()):digest(f) for f in (root/'q').rglob('*') if f.is_file()})
    files.update({str(f.resolve()):digest(f) for f in (protocol,root/'inventory.json',root/'effective_config.json')})
    plan=dict(project='STTW_CONTROL',experiment_id=proposal['experiment_id'],mode=proposal['mode'] if proposal.get('mode') in ('tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation','h75_late_state_tolerance48','h75_user_late_state192') else ('h75_samestate_bank144' if samestate_source.get('bank144') else 'h75_samestate_tolerance72') if samestate_source is not None else 'h75_online_confirmation100' if confirmation_source is not None else 'h75_online_scalar' if online else 'h75_switching_continuation' if continuation_source is not None else 'h75_channel_response' if stay_source is None else 'h75_channel_switching',
        source=str(source),cached_source=str(cached),native_runtime_root=old['native_runtime_root'],
        native_expected_sources=old['native_source_sha256'],initial_source=old['initial_source'],
        models={'H75':old['h75']},config=old['config'],contexts=native['contexts'],cases=proposal['expected_cases'],horizon_s=proposal['horizon_s'],goal_radius_m=proposal['goal_radius_m'],
        chunks=int(proposal.get('new_horizon_s',proposal['horizon_s'])/proposal['replan_cadence_s']),steps_per_chunk=50,maximum_active_lower_ticks=proposal['maximum_lower_ticks'],maximum_allocated_lower_slots=proposal['maximum_lower_ticks'],
        optimizer_updates=0,protocol=proposal,samestate_source=samestate_source,confirmation_source=confirmation_source,continuation_source=continuation_source,stay_source=None if stay_source is None else stay_source['source'],data_role=proposal['data_role'],question=proposal['question'],
        replanning=dict(period_s=proposal['replan_cadence_s'],radius_m=proposal['replanning_radius_m'],departure_m=proposal['departure_m'],maximum_per_case=proposal['maximum_replans'],information=proposal['replanning'],ordering=proposal['ordering']),
        scoring=proposal['scoring'],stop=proposal['stop_rule'],normalization=proposal['normalization'],initialization=proposal.get('initialization','Authenticated original native prepared bank/reset'),
        input_sha256=files,source_sha256={str(f.resolve()):digest(f) for f in modules})
    apply_route_context_variant(plan)
    if proposal.get('scheduled_replan_s') is not None:
        plan['chunk_upper_steps']=scheduled_chunk_steps(proposal);plan['chunks']=len(plan['chunk_upper_steps'])
    if proposal.get('mode') in ('tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation','h75_late_state_tolerance48','h75_user_late_state192'):
        plan['frozen_predictor_export']=dict(directory=str(root/'q'),receipt_sha256=digest(root/'q/receipt.json'),
            candidate_manifest_sha256=digest(root/'q/original_candidate_manifest.json'),
            arm_to_export_index=proposal.get('arm_to_export_index',dict(q_rank=1,q_bce=0)),seed_aggregation='equal mean of three sigmoid probabilities')
    if proposal.get('mode')=='tolerance_online40':
        comparison=confirmation_source['comparison'];cached_plan=json.loads((Path(comparison['source'])/'plan.json').read_text())
        validate_temporal_comparison_contract(proposal,cached_plan,plan['config'],plan['initial_source'],plan['models'],plan['contexts'])
        plan['comparison_source']=comparison
        receipt=json.loads((root/'q/receipt.json').read_text())
        if receipt['arm_order']!=proposal['arms']:raise ValueError('frozen export rank labels/slots differ')
    if proposal.get('mode')=='user_preference120':
        cache_helper=file_module('_sttw_user_preference_cache',Path('learning/cli/audit_native_goal_initial.py').resolve())
        plan['comparison_source']=cache_helper.validate_user_preference_cache(confirmation_source['source'],config=plan['config'],models=plan['models'],protocol=proposal,initial_source=plan['initial_source'],replanning=plan['replanning'])
        for file,sha in plan['comparison_source']['identity'].items():plan['input_sha256'][file]=sha
    if proposal.get('mode')=='user_prior_online72':
        cache_helper=file_module('_sttw_prior_online_cache',Path('learning/cli/audit_native_goal_initial.py').resolve())
        plan['comparison_source']=cache_helper.validate_prior_online_cache(confirmation_source['source'],config=plan['config'],models=plan['models'],protocol=proposal,initial_source=plan['initial_source'],replanning=plan['replanning'])
        for file,sha in plan['comparison_source']['identity'].items():plan['input_sha256'][file]=sha
    if proposal.get('mode')=='task_aware_fallback72':
        cache_helper=file_module('_sttw_task_aware_cache',Path('learning/cli/audit_native_goal_initial.py').resolve())
        plan['comparison_source']=cache_helper.validate_task_aware_cache(confirmation_source['source'],config=plan['config'],models=plan['models'],protocol=proposal,initial_source=plan['initial_source'],replanning=plan['replanning'])
        for file,sha in plan['comparison_source']['identity'].items():plan['input_sha256'][file]=sha
    if proposal.get('mode') in ('frozen_confirmation200','frozen_transfer_evaluation'):
        plan['frozen_candidate_source']=dict(source=confirmation_source['source'],plan_sha256=digest(Path(confirmation_source['source'])/'plan.json'),control_source_sha256=json.loads((Path(confirmation_source['source'])/'plan.json').read_text())['source_sha256'],q_parameters_sha256=digest(root/'q/parameters.npz'),new_comparison_physics_only=True,frozen_core_identity=confirmation_source.get('frozen_core_identity'))
    if proposal.get('mode')=='speed_query_online60':
        cache_helper=file_module('_sttw_speed_query_cache',Path('learning/cli/audit_native_goal_initial.py').resolve())
        plan['comparison_source']=cache_helper.validate_speed_query_cache(confirmation_source['source'],config=plan['config'],models=plan['models'],protocol=proposal,initial_source=plan['initial_source'],replanning=plan['replanning'])
        for file,sha in plan['comparison_source']['identity'].items():plan['input_sha256'][file]=sha
    if proposal.get('mode')=='guard_fallback_evaluation':
        helper=file_module('_sttw_guard_fallback_cache',Path('learning/cli/audit_native_goal_initial.py').resolve())
        plan['comparison_source']=helper.validate_guard_fallback_cache(confirmation_source['source'],config=plan['config'],models=plan['models'],protocol=proposal,initial_source=plan['initial_source'],replanning=plan['replanning'])
        plan['stage1_gate']=confirmation_source.get('stage1_gate')
        plan['intentional_control_change']=confirmation_source['intentional_control_change']
        for file,sha in plan['comparison_source']['identity'].items():plan['input_sha256'][file]=sha
    save(root/'plan.json',plan);save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))


def storage_guard(root):
    mount=Path('/media/qy/6A7E3DB07E3D75C3')
    if not os.path.ismount(mount) or not Path(root).resolve().is_relative_to(mount):
        raise ValueError('large evaluation outputs require the mounted data volume')
    stat=os.statvfs('/home/qy/STTW_CONTROL')
    if stat.f_bavail*stat.f_frsize<50*1024**3:
        raise ValueError('system volume must retain at least50GiB available')


def prepare(source,root):
    source=Path(source).resolve();root=Path(root).resolve();storage_guard(root)
    previous=json.loads((source/'plan.json').read_text())
    if digest(source/'plan.json')!=json.loads((source/'seal.json').read_text())['plan_sha256']:
        raise ValueError('native zero source plan identity changed')
    for f,sha in previous['input_sha256'].items():
        if digest(f)!=sha:raise ValueError('native source metadata changed '+f)
    status=json.loads((source/'evaluation/status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=8:
        raise ValueError('completed native zero source required')
    native_source=Path(previous['source'])
    original=json.loads((native_source/'plan.json').read_text())
    registry=Path('learning/configs/frozen_middle_registry.json').resolve()
    h75=json.loads(registry.read_text())['records']['STTW_MIDDLE_FULLTAIL_PPO75_U77']
    models=dict(H75={k:h75[k] for k in ('actor','actor_sha256','checkpoint','checkpoint_sha256')},
                current=original['current_model'])
    files=[source/'plan.json',source/'seal.json',source/'evaluation/status.json',native_source/'plan.json',registry]
    for model in models.values():
        for key in ('actor','checkpoint'):
            if digest(model[key])!=model[key+'_sha256']:raise ValueError('frozen model changed '+key)
            files.append(Path(model[key]))
    cfg=copy.deepcopy(previous['config'])
    cfg['training_future_phase_C']['episode_s']=20.
    cfg['path'].update(turn_radius_m=2.1,turn_angle_rad=math.radians(70))
    end=cfg['path']['straight_prefix_m']+cfg['path']['curvature_transition_length_m']+math.radians(70)*2.1
    cfg['path']['straight_tail_m']=80.-end
    if cfg['path_reward'].get('tolerance',{}).get('weight',0):raise ValueError('old analytic shaping must remain off')
    if cfg['residual_interface']['speed_reference_bounds_m_s']!=[1.5,3.5]:raise ValueError('native permission mismatch')
    own=Path(__file__).resolve();folder=own.parent
    modules=[own,folder/'native_goal_continuation.py',folder/'native_equilibrium_rule.py',folder/'contact_only_environment.py',
             folder/'goal_replanning.py',folder/'_vendor/pythonrobotics/dubins.py',folder/'_vendor/pythonrobotics/angle.py',
             Path('learning/cli/run_native_goal_confirmation.py').resolve()]
    files += [folder/'_vendor/pythonrobotics/LICENSE',folder/'_vendor/pythonrobotics/UPSTREAM.json']
    root.mkdir(exist_ok=False);(root/'source').mkdir()
    for f in modules:(root/'source'/f.name).write_bytes(f.read_bytes())
    save(root/'inventory.json',inventory());save(root/'effective_config.json',cfg)
    files += [root/'inventory.json',root/'effective_config.json']
    plan=dict(project='STTW_CONTROL',experiment_id='STTW-NATIVE-GOAL-CONFIRMATION-V1',owner='/root',
        question='Do frozen preference policies retain useful task or physical-response gains beyond zero and a fixed rule under a common goal-continuation contract?',
        source=str(source),native_source=str(native_source),native_runtime_root=previous['native_runtime_root'],
        native_expected_sources=previous['native_source_sha256'],initial_source=previous['initial_source'],models=models,
        config=cfg,cases=80,horizon_s=20.,goal_radius_m=1.,steps_per_chunk=50,chunks=20,
        maximum_active_lower_ticks=320000,maximum_allocated_lower_slots=320000,optimizer_updates=0,
        initialization='weights_only inference; original complete prepared bank snapshot plus native task reset, no synthetic physics state',
        normalization='Original native frame/context scales; archived Actor architecture, no learned observation normalizer',
        data_role='Prospective fixed related DEV contexts, shared complete ancestor; not independent training repeats or final TEST',
        rule=dict(budget_rad=cfg['path_reward']['working_roll_start_rad'],grid_points=65),
        replanning=dict(period_s=1.,radius_m=1.8,departure_m=1.,maximum_per_case=1,
            information='Own current geometry at ordinary boundaries; no event schedule, force or pulse-end input'),
        domain_contract='Native follower command50Hz; readonly geometry validity200Hz. Same mask on all nonzero targets. Reentry after16 complete clean upper blocks. Zero targets use original filters; no history/offset reset or extra speed support.',
        scoring='Original worldgoal1m before contact/fault at10/20s; whole-window contact separately. Original-route errors, original requested speed/time and reference cost separately. Current-path V3 is diagnostic after replanning.',
        stop='Fixed80 only; contact/fault/20s. Errors preserve raw partials and cost, no implicit restart, force/radius/time/threshold sweep or model adoption. All-pass is uninformative for binary response, not permission to escalate.',
        input_sha256={str(f.resolve()):digest(f) for f in files},source_sha256={str(f.resolve()):digest(f) for f in modules})
    save(root/'plan.json',plan);save(root/'seal.json',dict(plan_sha256=digest(root/'plan.json')))
    save(root/'status.json',dict(phase='prepared',active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0))


def verify(root):
    root=Path(root).resolve();storage_guard(root)
    p=json.loads((root/'plan.json').read_text())
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('changed plan')
    for f,sha in {**p['source_sha256'],**p['input_sha256']}.items():
        if digest(f)!=sha:raise ValueError('changed sealed source/input '+f)
    if json.loads((root/'inventory.json').read_text())!=(fixed_h75_capability_inventory(p['protocol']) if p.get('mode')==CAPABILITY_MODE else five_alpha_online_inventory(p['protocol']) if p.get('mode')==FIVE_ONLINE_MODE else native_source_prefix_inventory(p['protocol']) if p.get('mode')==PREFIX_MODE else five_alpha_label_inventory(p['protocol'],task_s=p.get('source_task_s')) if p.get('mode')==LABEL_MODE else middle_selection_inventory(p['protocol']) if p.get('mode')==SELECTION_MODE else bank144_inventory(p['protocol']) if p.get('mode') in ('h75_samestate_bank144','h75_late_state_tolerance48','h75_user_late_state192') else samestate_inventory(p['protocol']) if p.get('mode')=='h75_samestate_tolerance72' else confirmation_inventory(p['protocol']) if p.get('mode')=='h75_online_confirmation100' else online_inventory(p['protocol']) if p.get('mode')=='h75_online_scalar' else continuation_inventory(p['protocol']) if p.get('mode')=='h75_switching_continuation' else switching_inventory(p['protocol']) if p.get('mode')=='h75_channel_switching' else response_inventory(p['protocol']) if p.get('mode')=='h75_channel_response' else tolerance_inventory(p['protocol']) if p.get('mode') in ('tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation') else inventory()):raise ValueError('fixed inventory changed')
    return p


def load(root,*,prefix_phase=False):
    root=Path(root).resolve();p=verify(root);native=Path(p['native_runtime_root']).resolve()
    for name,module in tuple(sys.modules.items()):
        if name=='sttw_control' or name.startswith('sttw_control.'):
            file=getattr(module,'__file__',None)
            if file and not Path(file).resolve().is_relative_to(native):raise ValueError('modern control package already imported')
    sys.path.insert(0,str(native))
    import jax
    import jax.numpy as j
    import numpy as np
    import sttw_control.geometric_path as geometry
    from sttw_control.path_reference import pursuit
    from sttw_control.path_command_training import load_environment
    from sttw_control.path_command_policy import make_path_actor
    from sttw_control.direct_command_policy import q_ratio
    from sttw_control.path_command_reward import configured_cost_keys,failure_reward
    from sttw_control.runtime import configure_compilation_cache
    folder=Path(__file__).resolve().parent
    adapter=file_module('_sttw_goal_continuation',folder/'native_goal_continuation.py')
    contact=file_module('_sttw_goal_contact',folder/'contact_only_environment.py')
    rule=file_module('_sttw_goal_rule',folder/'native_equilibrium_rule.py')
    # Only planner/vendor modules live in this separate namespace. Bind their
    # Path dependency to the actual native type; do not extend sttw_control.
    from types import ModuleType
    package=ModuleType('_sttw_planner_only');package.__path__=[str(folder)]
    sys.modules[package.__name__]=package
    sys.modules[package.__name__+'.geometric_path']=geometry
    planner=file_module(package.__name__+'.goal_replanning',folder/'goal_replanning.py')
    configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
    labels=p.get('mode')==LABEL_MODE
    native_prefix=p.get('mode')==PREFIX_MODE
    five_online=p.get('mode')==FIVE_ONLINE_MODE
    capability=p.get('mode')==CAPABILITY_MODE
    returning=capability and bool(p['protocol'].get('causal_preference_return'))
    selection=p.get('mode') in (SELECTION_MODE,PREFIX_MODE,FIVE_ONLINE_MODE,CAPABILITY_MODE)
    cfg=p['config'];tolerance=five_online or p.get('mode') in ('tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation');confirmation=p.get('mode')=='h75_online_confirmation100' or tolerance or selection
    fresh=p.get('mode') in ('h75_samestate_bank144','h75_late_state_tolerance48','h75_user_late_state192') and new_confirmation_sources(p['protocol'])
    if confirmation or fresh or labels:
        env,prepared=load_confirmation_environment(cfg,p['initial_source'])
        snapshot=jax.tree.map(lambda x:j.asarray(x[p['protocol']['bank_indices'][0]]),prepared)
    else:env,snapshot=load_environment(cfg,p['initial_source'],'terminal_contract_v2')
    env=adapter.NativeGoalContinuation(env,contact_only_physics=contact.ContactOnlyPhysics,
                                        project=geometry.project,pursuit=pursuit)
    bank144=p.get('mode') in ('h75_samestate_bank144','h75_late_state_tolerance48','h75_user_late_state192')
    cases=bank144_prefix_cases(p['protocol']) if bank144 and prefix_phase else json.loads((root/'inventory.json').read_text());pose=np.asarray(env.physics.helpers.pose(snapshot.data))
    samestate=labels or p.get('mode')=='h75_samestate_tolerance72' or (bank144 and not prefix_phase)
    confirmation=p.get('mode')=='h75_online_confirmation100' or tolerance or selection
    online=five_online or p.get('mode') in ('h75_online_scalar','h75_online_confirmation100') or tolerance
    branch=p.get('mode')=='h75_switching_continuation'
    switching=p.get('mode') in ('h75_channel_switching','h75_switching_continuation')
    response=labels or selection or tolerance or p.get('mode') in ('h75_channel_response','h75_channel_switching','h75_switching_continuation','h75_online_scalar','h75_online_confirmation100','h75_samestate_tolerance72','h75_samestate_bank144','h75_late_state_tolerance48','h75_user_late_state192')
    if (samestate or bank144) and not fresh:
        bank={}
        for route in sorted({case_path_key(c) for c in cases}):
            with np.load(Path(p['source'] if labels else p['samestate_source']['source'])/'evaluation'/(route+'_path.npz')) as rawpath:
                bank[route]=geometry.Path(*(j.asarray(rawpath[k]) for k in ('s','xy','heading','curvature','turn_end','goal')))
    elif confirmation or fresh:
        snapshots={b:jax.tree.map(lambda x:j.asarray(x[b]),prepared) for b in p['protocol']['bank_indices']}
        for b,snap in snapshots.items():
            if snapshot_fingerprint(snap)!=p['protocol']['bank_state_fingerprints'][str(b)]:raise ValueError('new prepared snapshot fingerprint differs')
            if bool(snap.failed):raise ValueError('new prepared snapshot failed')
        from sttw_control.path_command_scenarios import _integrate_route
        bank={}
        for b,snap in snapshots.items():
            ownpose=np.asarray(env.physics.helpers.pose(snap.data))
            for c in p['contexts']:
                if c['route'] not in {r['route'] for r in cases}:continue
                path=_integrate_route(ownpose,cfg['path'],c.get('family',2),c['radius_m'],np.deg2rad(c['angle_deg']),c['speed'],c['direction'],0.,0.).path
                if capability and p['protocol'].get('capability_variant')=='S_goal_extension':
                    record=p['protocol']['goal_source']['paths'][c['route']]
                    with np.load(record['file']) as saved:path=path.replace(**{k:j.asarray(saved[k]) for k in ('s','xy','heading','curvature','turn_end','goal')})
                if selection and 'goal_distance_m' in c:path=path.replace(goal=j.asarray(c['goal_distance_m']))
                bank[f'bank{b}_'+c['route']]=extend_straight_support(path,p['protocol']['planner_points'],p['protocol']['planner_spacing_m'])
    elif response:
        from sttw_control.path_command_scenarios import _integrate_route
        bank={}
        for c in p['contexts']:
            if c['route'] not in {r['route'] for r in cases}:continue
            path=_integrate_route(pose,cfg['path'],2,c['radius_m'],np.deg2rad(c['angle_deg']),c['speed'],c['direction'],0.,0.).path
            with np.load(Path(p['cached_source'])/'evaluation'/(c['route']+'_path.npz')) as old:
                for field in ('s','xy','heading','curvature','turn_end','goal'):
                    if not np.array_equal(np.asarray(getattr(path,field)),old[field]):raise ValueError('original S path mismatch '+field)
            path=extend_straight_support(path,p['protocol']['planner_points'],p['protocol']['planner_spacing_m'])
            bank[c['route']]=path
    else:bank={c['route']:geometry.build_path(c['builder'],pose,cfg) for c in cases}
    target_points=p['protocol']['planner_points'] if response else 4001
    if any(len(path.s)!=target_points for path in bank.values()):raise ValueError('initial/replanned geometry support shape mismatch')
    stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
    paths=stack([bank[case_path_key(c)] for c in cases]);speeds=j.asarray([c['speed'] for c in cases]);alphas=j.asarray([c['alpha'] for c in cases])
    if labels:
        initial,paths,label_host=import_five_alpha_sources(p['source'],cases)
        initial_identity=dict(all_source_copy_leaves_exact=True,native_leaf_count=len(jax.tree.leaves(initial)),source_task_s=p['source_task_s'],source_boundary='before planner',source_host_flags_preserved=True)
    elif bank144:
        schedule=file_module('_sttw_online_scalar',folder/'geometric_preference_schedule.py')
        if prefix_phase and fresh:
            lane_snapshots=stack([snapshots[c['bank_index']] for c in cases])
            initial=jax.jit(jax.vmap(lambda snap,path,v,a:env.initial(env.reset(snap,path,v,a))))(lane_snapshots,paths,speeds,alphas)
            if not bool(j.all((initial.state.tick==0)&(initial.state.physical.physical_tick==700)&(initial.state.alpha==1.))):raise ValueError('new prefix source clock/user mismatch')
            initial_identity=dict(authenticated_prepared_bank=True,bank_state_fingerprints=p['protocol']['bank_state_fingerprints'],native_leaf_count=len(jax.tree.leaves(initial)),prefix_lanes=len(cases),task_initialization='exact confirmation reset once; no reset at suffix')
        elif prefix_phase:
            with (Path(p['samestate_source']['source'])/'evaluation/initial_carry.pkl').open('rb') as f:archived=pickle.load(f)
            initial=gather_source_carry(archived.native,[c['confirmation_row'] for c in cases])
            if not bool(j.all((initial.state.tick==0)&(initial.state.physical.physical_tick==700)&(initial.state.alpha==1.))):raise ValueError('hybrid prefix source clock/user mismatch')
            initial_identity=dict(all_source_copy_leaves_exact=True,native_leaf_count=len(jax.tree.leaves(initial)),prefix_lanes=4)
        else:
            sealed=json.loads((root/'sources/seal.json').read_text())
            if sealed['plan_sha256']!=digest(root/'plan.json'):raise ValueError('generated source plan identity mismatch')
            records=json.loads((root/'sources/records.json').read_text())
            for c in cases:c['source_fingerprint']=records[c['source_index']]['source_fingerprint']
            for f,sha in sealed['files'].items():
                if digest(f)!=sha:raise ValueError('generated source identity drift')
            with (root/'sources/complete_carry.pkl').open('rb') as f:saved=pickle.load(f)
            initial=gather_source_carry(saved,[c['source_index'] for c in cases])
            if p.get('mode') in ('h75_late_state_tolerance48','h75_user_late_state192'):
                with (root/'sources/executed_paths.pkl').open('rb') as f:saved_paths=pickle.load(f)
                paths=gather_source_carry(saved_paths,[c['source_index'] for c in cases])
            if not bool(j.all(initial.state.tick==round(p['protocol']['source_task_s']/env.cc.dt))):raise ValueError('assembled source task clock mismatch')
            initial_identity=dict(all_source_copy_leaves_exact=True,native_leaf_count=len(jax.tree.leaves(initial)),source_groups=len(p['protocol']['source_groups']),copies_per_source=len(cases)//len(p['protocol']['source_groups']),source_receipt=sealed)
    elif samestate:
        with (Path(p['samestate_source']['source'])/'evaluation/task1_complete_carry.pkl').open('rb') as f:saved=pickle.load(f)
        for c in p['protocol']['source_rows']:
            selected=jax.tree.map(lambda x:np.asarray(x)[c['source_index']],saved)
            if snapshot_fingerprint(selected)!=c['fingerprint']:raise ValueError('complete source fingerprint changed')
        initial=gather_source_carry(saved,[c['source_index'] for c in cases])
        aa=jax.tree.leaves(jax.device_get(initial));bb=jax.tree.leaves(saved)
        if any(a.dtype!=b.dtype or any(not np.array_equal(a[i],np.asarray(b)[c['source_index']],equal_nan=True) for i,c in enumerate(cases)) for a,b in zip(aa,bb)):raise ValueError('same-state copy changed source leaves')
        if not bool(j.all((initial.state.tick==200)&(initial.state.alpha==1.))):raise ValueError('source clock/useralpha mismatch')
        initial_identity=dict(all_source_copy_leaves_exact=True,native_leaf_count=len(aa),source_groups=4,copies_per_source=18,source_fingerprints={str(c['source_index']):c['fingerprint'] for c in p['protocol']['source_rows']})
    elif confirmation:
        lane_snapshots=stack([snapshots[c['bank_index']] for c in cases])
        initial=jax.jit(jax.vmap(lambda snap,path,v,a:env.initial(env.reset(snap,path,v,a))))(lane_snapshots,paths,speeds,alphas)
    else:initial=jax.jit(jax.vmap(lambda path,v,a:env.initial(env.reset(snapshot,path,v,a))))(paths,speeds,alphas)
    jax.block_until_ready(initial)
    conditional_receipt=None
    if capability and any('state_source' in c for c in p['contexts']):
        initial,conditional_receipt=restore_conditional_capability_states(p,cases,paths)
    if response and not confirmation and not samestate and not bank144:
        cached=Path(p['cached_source'])
        with (cached/'evaluation/initial_carry.pkl').open('rb') as f:archived=pickle.load(f)
        old_cases=json.loads((cached/'plan.json').read_text())['cases']
        indices=[next(i for i,c in enumerate(old_cases) if c['route']==case['route'] and c['speed']==case['speed'] and c['policy']=='H75' and c['alpha']==case['alpha']) for case in cases]
        expected=jax.tree.map(lambda x:np.asarray(x)[indices],archived)
        # Only the documented continuation domain latch is adapted; full native reset identity is exact.
        raw=jax.jit(jax.vmap(lambda path,v,a:env.reset(snapshot,path,v,a)))(paths,speeds,alphas)
        aa,ta=jax.tree.flatten(jax.device_get(raw));bb,tb=jax.tree.flatten(expected)
        if ta!=tb or len(aa)!=len(bb) or any(a.dtype!=b.dtype or not np.array_equal(a,b,equal_nan=True) for a,b in zip(aa,bb)):
            raise ValueError('cached source native reset mismatch; no compatibility weakening')
        initial_identity=dict(all_native_leaves_exact=True,native_leaf_count=len(aa),dtypes_exact=True,original_paths_exact=True)

    if capability:
        initial_identity=audit_five_alpha_online_initial(initial,cases,initial.state.physical.physical_tick[0] if conditional_receipt else snapshot.physical_tick,panel_kind='boundary_selected_user' if p['protocol'].get('range_planner_boundary') else 'disturbed_declared_subset' if p['protocol'].get('excluded_completed_cases') else 'disturbed_declared_user' if p['protocol'].get('capability_variant')=='low_speed_lateral_load' else 'targeted_three_user' if p['protocol'].get('capability_variant')=='targeted_base_review' else 'nominal_five_user',expected_route_count=len(p['contexts']),task_tick=j.asarray([round(c.get('source_task_start_s',0.)/.005) for c in cases]))
        if conditional_receipt:initial_identity.update(conditional_receipt)
        for key in ('authenticated_native_online','initial_previous_alpha','initial_hold_until_tick'):initial_identity.pop(key)
        initial_identity['authenticated_fixed_h75_capability']=True
    elif five_online:
        initial_identity=audit_five_alpha_online_initial(initial,cases,snapshot.physical_tick,panel_kind=p['protocol'].get('panel_kind'))
    elif native_prefix:
        if not bool(j.all((initial.state.tick==0)&(initial.state.alpha==.5)&(initial.state.physical.physical_tick==snapshot.physical_tick))):raise ValueError('native prefix initialization differs')
        initial_identity=dict(authenticated_native_prefix=True,prefix_lanes=len(cases),native_leaf_count=len(jax.tree.leaves(initial)),new_bank_fingerprints=p['protocol']['bank_state_fingerprints'],collection_s=p['collection_s'],task_horizon_s=p['horizon_s'])
    elif selection:
        groups={}
        for i,c in enumerate(cases):groups.setdefault((c['bank_index'],c['route'],c['event']),[]).append(i)
        neutral=initial.replace(state=initial.state.replace(alpha=j.zeros_like(initial.state.alpha)))
        leaves=jax.tree.leaves(jax.device_get(neutral))
        if any(len(ids)!=16 for ids in groups.values()) or any(any(not np.array_equal(x[ids[0]],x[i],equal_nan=True) for i in ids[1:]) for ids in groups.values() for x in leaves):raise ValueError('selection reset differs across models/preferences')
        initial_identity=dict(all_arm_reset_leaves_exact=True,alpha_neutralized=True,groups=len(groups),native_leaf_count=len(leaves),new_bank_fingerprints=p['protocol']['bank_state_fingerprints'])
    elif confirmation:
        groups={}
        for i,c in enumerate(cases):groups.setdefault((c['bank_index'],c['route'],c['event']),[]).append(i)
        comparison_initial=initial
        if p.get('mode') in ('user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation'):comparison_initial=initial.replace(state=initial.state.replace(alpha=j.ones_like(initial.state.alpha)))
        leaves=jax.tree.leaves(jax.device_get(comparison_initial))
        if len(groups)!=(len(cases)//len(p['protocol']['arms']) if p.get('mode')=='frozen_transfer_evaluation' else 24 if p.get('mode') in ('tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72') or (p.get('mode') in ('frozen_confirmation200','guard_fallback_evaluation','frozen_transfer_evaluation') and p['protocol'].get('variant')=='geometry_duration_extension_v1') else 20) or any(len(ids)!=(len(p['protocol']['arms']) if tolerance else 5) for ids in groups.values()):raise ValueError('newbank reset grouping mismatch')
        if any(any(not np.array_equal(x[ids[0]],x[i],equal_nan=True) for i in ids[1:]) for ids in groups.values() for x in leaves):raise ValueError('five-arm complete physical reset differs')
        initial_identity=dict(user_alpha_neutralized_for_cross_user_reset=p.get('mode') in ('user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation'),new_bank_fingerprints=p['protocol']['bank_state_fingerprints'],all_five_arm_reset_leaves_exact=not tolerance,all_arm_reset_leaves_exact=True,native_leaf_count=len(leaves),groups=len(groups),original_paths_anchored_to_own_pose=True,no_old_trajectory_reuse=True)
    if tolerance and not five_online and p.get('mode') not in ('tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation'):
        ancestor=Path(p['confirmation_source']['source'])
        ancestor_plan=json.loads((ancestor/'plan.json').read_text())
        source_cases=bank144_prefix_cases(ancestor_plan['protocol'])
        with (ancestor/'prefix/initial_carry.pkl').open('rb') as f:saved_initial=pickle.load(f)
        mapping={(c['bank_index'],c['route']):i for i,c in enumerate(source_cases) if c['origin']=='fixed1'}
        expected=gather_source_carry(saved_initial,[mapping[c['bank_index'],c['route']] for c in cases])
        actual_leaves,actual_tree=jax.tree.flatten(jax.device_get(initial));expected_leaves,expected_tree=jax.tree.flatten(expected)
        if actual_tree!=expected_tree or any(a.dtype!=b.dtype or not np.array_equal(a,b,equal_nan=True) for a,b in zip(actual_leaves,expected_leaves)):
            raise ValueError('new online task0 reset differs from saved native prefix initial')
        initial_identity['saved_prefix_initial_all_native_leaves_exact']=True
        initial_identity['saved_prefix_initial_file']=str(ancestor/'prefix/initial_carry.pkl')
    if branch:
        with (Path(p['continuation_source']['source'])/'evaluation/task1_complete_carry.pkl').open('rb') as f:saved=pickle.load(f)
        initial=jax.tree.map(j.asarray,saved)
        aa,ta=jax.tree.flatten(jax.device_get(initial));bb,tb=jax.tree.flatten(saved)
        if ta!=tb or len(aa)!=len(bb) or any(a.dtype!=b.dtype or not np.array_equal(a,b,equal_nan=True) for a,b in zip(aa,bb)):
            raise ValueError('own saved continuation leaf changed during loading')
        initial_identity['continuation_snapshot']=dict(all_saved_leaves_preserved=True,common_origin_state='approximate; user-authorized numerical amendment',prefix_lane_maximum_absolute_difference=p['continuation_source']['prefix_lane_maximum_absolute_difference'])
        if not bool(j.all(initial.state.tick==200)):raise ValueError('continuation task1 tick mismatch')

    if bool(j.any(initial.state.fault|initial.state.physical.failed)):
        raise ValueError('initial native task is invalid; not a physical failure result')
    env.set_log_template(jax.tree.map(lambda x:x[0],initial.state),bank[case_path_key(cases[0])],speeds[0])
    if online:
        schedule=file_module('_sttw_online_scalar',folder/'geometric_preference_schedule.py')
        if tolerance:
            coordination=file_module('_sttw_tolerance_coordination',folder/'tolerance_coordination.py')
            qparams=jax.tree.map(j.asarray,coordination.load_five_alpha_export(p['protocol']['predictor_training'],expected_export_sha256=p['protocol']['export_manifest_sha256'],candidate_alphas=tuple(p['protocol']['candidate_alphas']),hypothetical_duration_s=p['protocol']['hypothetical_duration_s']) if five_online else coordination.load_export(root/'q'))
            initial=coordination.OnlineCarry(initial,j.zeros(len(cases),bool),alphas if five_online or p.get('mode') in ('user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation') else j.ones(len(cases)))
            if five_online and p['protocol'].get('alpha_dwell_s',0.):initial=coordination.DwellOnlineCarry(initial.native,initial.latched,initial.prev_alpha,j.zeros(len(cases),j.int32))
        else:initial=schedule.OnlineCarry(initial,j.zeros(len(cases),bool))
    actors={}
    for name,model in p['models'].items():
        with Path(model['actor']).open('rb') as f:actors[name]=jax.tree.map(j.asarray,pickle.load(f))
    actor=make_path_actor(cfg)
    for params in actors.values():
        if jax.eval_shape(actor.apply,params,j.zeros(351)).shape!=(2,):raise ValueError('native Actor shape mismatch')
    params=stack([actors[c['mode'] if c['mode'] in actors else 'H75'] for c in cases])
    n=len(cases);horizon=p['horizon_s'];speed_alpha=j.asarray([c.get('query_alpha',c.get('speed_alpha',c['alpha'])) for c in cases]);steer_alpha=j.asarray([c.get('query_alpha',c.get('steer_alpha',c['alpha'])) for c in cases])
    if capability and p['protocol'].get('range_planner_boundary'):
        boundary=p['protocol']['range_planner_boundary'];receipts=[];selected=[];by_route={}
        for i,case in enumerate(cases):
            lane=jax.tree.map(lambda x:x[i],initial)
            user=float(case['original_user_alpha']);state=lane.state.replace(alpha=j.asarray(user,lane.state.alpha.dtype))
            path=jax.tree.map(lambda x:x[i],paths)
            obs,_,fault=env.observation(state,path,speeds[i],horizon)
            query_grid=sorted(set(boundary['candidate_alphas']+[user]))
            queries=jax.vmap(lambda a:obs.at[336].set(a))(j.asarray(query_grid))
            latent=jax.vmap(lambda x:actor.apply(actors['H75'],x))(queries)
            yaw=float(env.physics.observe(state.physical.data)[1][2])
            permission=bool(adapter.target_permission(lane)&~fault)
            receipt=query_boundary_alpha(folder,dict(boundary,candidate_alphas=query_grid),np.asarray(obs),np.asarray(latent),yaw,float(state.tick)*env.cc.dt,cfg['residual_interface'],user,permission)
            chosen=receipt['decision']['selected_alpha']
            if chosen!=case['alpha']:raise ValueError('actual NN choice differs from sealed expected candidate; no physical execution')
            signature=dict(plan=receipt['plan'],decision=receipt['decision'])
            if case['route'] in by_route and by_route[case['route']]!=signature:raise ValueError('same-source boundary decisions depend on actual event')
            by_route[case['route']]=signature;receipts.append(dict(case_id=case['case_id'],**receipt));selected.append(chosen)
        speed_alpha=j.asarray(selected);steer_alpha=speed_alpha
        initial=initial.replace(state=initial.state.replace(alpha=speed_alpha))
        save(root/'range_planner_decisions.json',dict(actual_future_event_input=False,same_source_actual_event_decisions_identical=True,decisions=receipts))
    if returning:
        return_module=file_module('_sttw_preference_return',folder/'tolerance_coordination.py')
        return_rule=return_module.validate_preference_return_rule(p['protocol']['causal_preference_return'])
        initial=initial.replace(state=initial.state.replace(alpha=j.full(n,return_rule['user_alpha'],dtype=initial.state.alpha.dtype)))
        initial=return_module.PreferenceReturnCarry(initial,speed_alpha,j.zeros(n,j.int32),initial.state.tick,j.zeros(n,j.int32))
        initial_identity.update(original_user_alpha=return_rule['user_alpha'],initial_executing_alphas=np.asarray(speed_alpha).tolist(),causal_preference_return=True)
    modes=j.asarray([c['mode_index'] for c in cases],j.int32)
    physical_start=float(snapshot.physical_tick)*env.cc.dt
    pulses=j.asarray(force_pulses(cases,physical_start))
    keys=list(configured_cost_keys(cfg))
    keep=('active_tick','time','actual_xy','reference_xy','actual_forward_speed','wheel_speed_proxy','v_user',
          'path_cross_track','path_heading_error','path_progress','phi','phi_dot','actual_delta','peak_roll',
          'physical_failure','contact_failure','nonfinite','domain_exit','policy_fault','lower_fault',
          'governed','nominal','applied_force_impulse','scored_tick_reward','latent_z','offsets','filtered_offset','target_offset',
          'reference_target_permitted','cumulative_domain_exit','clean_blocks_at_publication','clean_blocks_at_end',
          'continuation_geometry_invalid','continuation_geometry_fault')
    switch_ticks=j.asarray([round(c.get('switch_s',1e9)/env.cc.dt) for c in cases],j.int32) if switching else j.zeros(n,j.int32)
    def one(carry,param,path,v,pulse,mode,sa,ta,st):
        if returning:
            sidecar=carry;carry=carry.native
        elif tolerance:
            sidecar=carry;carry=carry.native
        elif online:
            alpha_exec,latched,decision=schedule.advance_scalar(carry.latched,v,path.s,path.curvature,carry.state.projection.progress,p['protocol']['rule'])
            if confirmation:
                alpha_exec=j.where(mode==0,alpha_exec,sa);latched=j.where(mode==0,latched,False)
                decision=dict(decision,rule_alpha_target=alpha_exec)
            carry=carry.native
        confirmation_method=j.asarray(declared_confirmation_methods(cases,p['protocol']['arms']))[mode] if p.get('mode') in ('frozen_confirmation200','frozen_transfer_evaluation') else j.asarray(-1)
        origin_pair=j.stack([sa,ta])
        if switching:sa,ta,switched=switch_mapping(carry.state.tick,sa,ta,st,p['protocol']['destination_pair'])
        if response:
            obs,_,obs_fault=env.observation(carry.state,path,v,horizon)
            if returning:
                measurement,_,true_speed=env.physics.observe(carry.state.physical.data)
                return_permission=adapter.target_permission(carry)&~obs_fault
                returned=return_module.advance_preference_return(sidecar.alpha_exec,sidecar.stable_ticks,sidecar.phase,
                    sidecar.start_tick,carry.state.tick,measurement[0],measurement[1],true_speed,return_permission,return_rule)
                sa=ta=returned['alpha']
            if five_online:
                candidate_alpha=j.asarray(p['protocol']['candidate_alphas'])
                queries=jax.vmap(lambda a:obs.at[336].set(a))(candidate_alpha)
                latents=jax.vmap(lambda x:actor.apply(param,x))(queries)
                yaw=env.physics.observe(carry.state.physical.data)[1][2]
                permission=adapter.target_permission(carry)&~obs_fault
                # No event schedule, direction, force or pulse end enters this call.
                cd=coordination.coordinate(qparams,obs,latents,yaw,carry.state.physical.raw,v,
                    carry.state.projection.progress,path.s,path.curvature,permission,sidecar.latched,sidecar.prev_alpha,
                    cfg['residual_interface'],candidate_alphas=tuple(p['protocol']['candidate_alphas']),
                    hypothetical_duration_s=p['protocol']['hypothetical_duration_s'],arm=0,use_q=True,user_alpha=carry.state.alpha,guard_fallback=p['protocol'].get('guard_fallback',False),task_aware_fallback=False,curvature_task_mode=p['protocol'].get('curvature_task_mode','hard'))
                native_permission=permission
                if p['protocol'].get('curvature_task_mode','hard')=='soft_penalty':
                    # Invalid proposed reference is an engineering fault, not a
                    # safe midpoint fallback. Reuse the native fault/withhold path.
                    carry=carry.replace(state=carry.state.replace(fault=carry.state.fault|cd['candidate_reference_fault']))
                    permission=permission&~cd['candidate_reference_fault']
                alpha_exec,latched=cd['alpha'],cd['latched']
                dwell=None
                if p['protocol'].get('alpha_dwell_s',0.):
                    dwell=coordination.apply_alpha_dwell(cd['alpha'],cd['latched'],cd['task_mask'],permission,sidecar.prev_alpha,sidecar.latched,sidecar.hold_until_tick,carry.state.tick,candidate_alphas=tuple(p['protocol']['candidate_alphas']),dwell_ticks=round(p['protocol']['alpha_dwell_s']/env.cc.dt))
                    alpha_exec,latched=dwell['alpha'],dwell['latched']
                queried=obs.at[336].set(alpha_exec)
                z=latents[j.argmin(j.abs(candidate_alpha-alpha_exec))];half=full=j.zeros_like(z)
                user_index=j.argmin(j.abs(candidate_alpha-carry.state.alpha))
                restore=cd['task_mask'][user_index]&(cd['qmin'][user_index]>=j.where(sidecar.latched,.85,.8))
                eligible=cd['task_mask']&(cd['qmin']>=.8)&(candidate_alpha!=carry.state.alpha)
                decision=dict(rule_preview_max_curvature=cd['max_abs_curvature'],rule_accel_estimate=v*v*cd['max_abs_curvature'],rule_alpha_target=alpha_exec)
                coord_log={'coord_'+k:cd[k] for k in ('probabilities','qmin','targets','task_mask','proxy_cost','degraded','publication_withheld','max_abs_curvature')}
                coord_log.update(coord_prev_alpha=sidecar.prev_alpha,coord_latents=latents,coord_candidate_alpha=candidate_alpha,
                    coord_permission=permission,coord_pre_progress=carry.state.projection.progress,coord_latched_before=sidecar.latched,
                    coord_nominal=carry.state.physical.raw,coord_candidate_clipped=cd['target_clipped'],coord_unbounded_targets=cd['unbounded_targets'],
                    coord_eligible=eligible,coord_restore_requested=restore,coord_restore_published=restore&permission)
                if p['protocol'].get('curvature_task_mode','hard')=='soft_penalty':
                    coord_log.update({'coord_'+key:cd[key] for key in ('historical_hard_task_mask','valid_target_mask','original_proxy_cost','proxy_accel','curvature_excess','curvature_excess_penalty','selected_reference_valid','candidate_reference_fault')})
                    coord_log['coord_native_permission']=native_permission
                if p['protocol'].get('guard_fallback',False):
                    guard_requested=(~restore)&(~j.any(eligible))&j.any(cd['task_mask'])
                    guard_published=cd['guard_fallback_selected']&(j.asarray(True) if dwell is None else ~dwell['held'])&(alpha_exec==cd['alpha'])
                    coord_log.update(coord_guard_fallback=cd['guard_fallback'],coord_guard_fallback_requested=guard_requested,coord_guard_fallback_selected=cd['guard_fallback_selected'],coord_guard_fallback_alpha=cd['alpha'],coord_guard_fallback_published=guard_published)
                if dwell is not None:
                    coord_log.update(coord_raw_alpha=cd['alpha'],coord_raw_latched=cd['latched'],coord_dwell_tick=carry.state.tick,coord_hold_until_before=sidecar.hold_until_tick,coord_hold_until_tick=dwell['hold_until_tick'],coord_hold_remaining_ticks=dwell['remaining_ticks'],coord_hold_release_reason=dwell['release_reason'],coord_held=dwell['held'],coord_alpha_changed=dwell['alpha_changed'])
                    coord_log['coord_restore_published']=restore&permission&~dwell['held']&~latched&(alpha_exec==carry.state.alpha)
            elif selection:
                queried=obs.at[336].set(sa);z=jax.lax.cond(mode==3,lambda _:j.zeros(2),lambda _:actor.apply(param,queried),None);half=full=j.zeros_like(z)
            elif tolerance:
                candidate_alpha=j.asarray(p['protocol']['coordination']['candidate_alphas'])
                queries=jax.vmap(lambda a:obs.at[336].set(a))(candidate_alpha)
                latents=jax.vmap(lambda x:actor.apply(param,x))(queries)
                yaw=env.physics.observe(carry.state.physical.data)[1][2]
                permission=adapter.target_permission(carry)&~obs_fault
                cd=coordination.coordinate(qparams,obs,latents,yaw,carry.state.physical.raw,v,
                    carry.state.projection.progress,path.s,path.curvature,permission,sidecar.latched,sidecar.prev_alpha,
                    cfg['residual_interface'],candidate_alphas=tuple(p['protocol']['coordination']['candidate_alphas']),use_q=(confirmation_method!=1) if p.get('mode') in ('frozen_confirmation200','frozen_transfer_evaluation') else True if p.get('mode') in ('user_prior_online72','task_aware_fallback72','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation') else ((mode!=1)&(mode!=4)) if p.get('mode')=='user_preference120' else (mode!=1) if p.get('mode')=='tolerance_broaden120' else True if p.get('mode')=='tolerance_online40' else mode!=2,arm=(j.asarray([p['protocol']['arm_to_export_index'][a] for a in p['protocol']['arms']])[mode] if p.get('mode') in ('tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation') else j.where(mode==1,0,1)),user_alpha=carry.state.alpha if p.get('mode') in ('user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation') else 1.,task_aware_fallback=(confirmation_method==0) if p.get('mode') in ('frozen_confirmation200','frozen_transfer_evaluation') else p.get('mode') in ('task_aware_fallback72','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation'),guard_fallback=(confirmation_method==0) if p.get('mode')=='frozen_transfer_evaluation' else p.get('mode')=='guard_fallback_evaluation')
                geometric_alpha,geometric_latch,geometric=schedule.advance_scalar(sidecar.latched,v,path.s,path.curvature,carry.state.projection.progress,p['protocol']['rule'])
                alpha_exec,latched=select_tolerance_arm(mode,cd,geometric_alpha,geometric_latch,permission,sidecar.prev_alpha,sidecar.latched)
                if p.get('mode')=='tolerance_broaden120':
                    alpha_exec,latched=select_broaden_arm(mode,cd,permission,sidecar.prev_alpha,sidecar.latched)
                if p.get('mode')=='user_preference120':
                    alpha_exec=j.where(mode==2,j.where(permission,carry.state.alpha,sidecar.prev_alpha),cd['alpha'])
                    latched=j.where(mode==2,j.where(permission,False,sidecar.latched),cd['latched'])
                if p.get('mode') in ('user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation'):alpha_exec,latched=cd['alpha'],cd['latched']
                if p.get('mode') in ('frozen_confirmation200','frozen_transfer_evaluation'):alpha_exec,latched=select_confirmation_arm(confirmation_method,cd,permission,sidecar.prev_alpha,sidecar.latched,carry.state.alpha)
                queried=obs.at[336].set(alpha_exec)
                selected=j.argmin(j.abs(candidate_alpha-alpha_exec));z=latents[selected]
                speedcap_log={}
                if p.get('mode') in ('tolerance_broaden120','frozen_confirmation200','frozen_transfer_evaluation'):
                    capped,caplog=coordination.speedcap_latent(latents[j.argmin(j.abs(candidate_alpha-1.))],carry.state.physical.raw,cd['max_abs_curvature'],cfg['residual_interface'])
                    z=j.where(confirmation_method==3 if p.get('mode') in ('frozen_confirmation200','frozen_transfer_evaluation') else mode==4,capped,z)
                    speedcap_log={f'speedcap_{k}':value for k,value in caplog.items()}
                    speedcap_log['speedcap_active']=(confirmation_method==3 if p.get('mode') in ('frozen_confirmation200','frozen_transfer_evaluation') else mode==4)&permission&caplog['active']
                half=full=j.zeros_like(z)
                decision=dict(rule_preview_max_curvature=geometric['rule_preview_max_curvature'],rule_accel_estimate=geometric['rule_accel_estimate'],rule_alpha_target=alpha_exec)
                coord_log=dict(speedcap_log,**{'coord_'+k:cd[k] for k in ('probabilities','qmin','targets','task_mask','proxy_cost','degraded','publication_withheld','max_abs_curvature')})
                if p.get('mode') in ('task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation'):coord_log.update(coord_task_aware_fallback=cd['task_aware_fallback'],coord_task_aware_selected=cd['task_aware_selected'])
                if p.get('mode') in ('guard_fallback_evaluation','frozen_transfer_evaluation'):coord_log.update(coord_guard_fallback=cd['guard_fallback'],coord_guard_fallback_selected=cd['guard_fallback_selected'])
                coord_log['coord_degraded']=((confirmation_method<2) if p.get('mode') in ('frozen_confirmation200','frozen_transfer_evaluation') else True if p.get('mode') in ('user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation') else (mode!=2) if p.get('mode')=='user_preference120' else mode<2 if p.get('mode')=='tolerance_broaden120' else mode<3)&cd['degraded']
                coord_log.update(coord_prev_alpha=sidecar.prev_alpha,coord_latents=latents,coord_candidate_alpha=candidate_alpha,coord_permission=permission,coord_pre_progress=carry.state.projection.progress,coord_latched_before=sidecar.latched,coord_nominal=carry.state.physical.raw,coord_candidate_clipped=cd['target_clipped'],coord_unbounded_targets=cd['unbounded_targets'])
            elif online:
                queried=obs.at[336].set(alpha_exec)
                z=jax.lax.cond(mode==4,lambda _:j.zeros(2),lambda _:actor.apply(param,queried),None) if confirmation else actor.apply(param,queried)
                half=full=j.zeros_like(z)
                if confirmation:
                    support_z,support=existing_floor_latent(carry.state.physical.raw,cfg,p['protocol']['existing_speed_floor_m_s'])
                    z=j.where(mode==4,support_z,z)
            elif samestate:
                queried=obs.at[336].set(sa);z=actor.apply(param,queried);half=full=j.zeros_like(z)
            else:z,half,full=channel_queries(obs,lambda x:actor.apply(param,x),sa,ta)
        def target(obs,state):
            if response:return z
            return jax.lax.cond(mode==3,
                lambda _:rule.target(state.physical.raw,state.alpha,cfg,lambda x:q_ratio(x,env.cc),p['rule']['budget_rad'],p['rule']['grid_points'])[0],
                lambda _:actor.apply(param,obs),None)
        end,log=env.policy_step(carry,path,v,horizon,target,pulse,zero_mode=False if confirmation or samestate else mode==2)
        if response:
            ticks=lambda x:j.broadcast_to(x,log['active_tick'].shape+j.asarray(x).shape)
            log.update(queried_latent_alpha_half=ticks(half),queried_latent_alpha_one=ticks(full),channel_alpha_pair=ticks(j.stack([sa,ta])),composed_latent=ticks(z),query_observation=ticks(obs),pre_upper_yaw=ticks(env.physics.observe(carry.state.physical.data)[1][2]))
        if samestate:log.update(query_observation=ticks(queried),channel_alpha_pair=ticks(j.stack([sa,sa])),alpha_user=ticks(carry.state.alpha),alpha_exec=ticks(sa))
        if returning:
            log.update(query_observation=ticks(queried),alpha_user=ticks(carry.state.alpha),alpha_exec=ticks(sa),
                return_pre_roll=ticks(measurement[0]),return_pre_roll_rate=ticks(measurement[1]),return_pre_speed=ticks(true_speed),
                return_permission=ticks(return_permission),return_start_tick=ticks(sidecar.start_tick),
                **{'return_'+key:ticks(value) for key,value in returned.items() if key!='alpha'})
            end=return_module.PreferenceReturnCarry(end,sa,returned['stable_ticks'],sidecar.start_tick,returned['phase'])
        if confirmation and not tolerance and not selection:
            log.update(neural_query_active=ticks(mode!=4),support_nominal_reference=ticks(carry.state.physical.raw),rule_pre_upper_progress=ticks(carry.state.projection.progress),**{k:ticks(j.where(mode==4,x,j.zeros_like(x))) for k,x in support.items()})
        if online:
            log.update(alpha_user=ticks(carry.state.alpha if five_online or p.get('mode') in ('user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation') else j.asarray(1.)),alpha_exec=ticks(alpha_exec),rule_latched=ticks(latched),query_observation=ticks(queried),channel_alpha_pair=ticks(j.stack([alpha_exec,alpha_exec])),**{k:ticks(x) for k,x in decision.items()})
            if tolerance:
                log.update(rule_pre_upper_progress=ticks(carry.state.projection.progress),**{k:ticks(x) for k,x in coord_log.items()})
                end=coordination.DwellOnlineCarry(end,latched,alpha_exec,dwell['hold_until_tick']) if five_online and p['protocol'].get('alpha_dwell_s',0.) else coordination.OnlineCarry(end,latched,alpha_exec)
            else:end=schedule.OnlineCarry(end,latched)
        if switching:log.update(origin_channel_alpha_pair=ticks(origin_pair),switch_active=ticks(switched),switch_tick=ticks(st))
        return end,dict({k:log[k] for k in keep},**({k:log[k] for k in ('alpha_user','alpha_exec')} if samestate or returning else {}),**({k:log[k] for k in log if k.startswith('return_')} if returning else {}),**({k:log[k] for k in ('neural_query_active','support_nominal_reference','rule_pre_upper_progress','support_floor','support_requested_offset','support_realized_target_offset','support_active')} if confirmation and not tolerance and not selection else {}),**({k:log[k] for k in ('alpha_user','alpha_exec','rule_latched','rule_preview_max_curvature','rule_accel_estimate','rule_alpha_target')} if online else {}),**({k:log[k] for k in ('origin_channel_alpha_pair','switch_active','switch_tick')} if switching else {}),**({k:log[k] for k in (('channel_alpha_pair','composed_latent','query_observation','pre_upper_yaw','reference_yaw_unwrapped','e_psi_unwrapped') if online or samestate else ('queried_latent_alpha_half','queried_latent_alpha_one','channel_alpha_pair','composed_latent','query_observation','pre_upper_yaw','reference_yaw_unwrapped','e_psi_unwrapped'))} if response else {}),**({k:log[k] for k in (*coord_log,'rule_pre_upper_progress')} if tolerance else {}),effective_costs=j.stack([log['effective_components'][k] for k in keys],axis=-1))
    @partial(jax.jit,static_argnames=('upper_steps',))
    def chunk(carry,paths,upper_steps=50):
        return jax.lax.scan(lambda c,_:jax.vmap(one)(c,params,paths,speeds,pulses,modes,speed_alpha,steer_alpha,switch_ticks),carry,None,length=upper_steps)
    @jax.jit
    def measurements(carry):
        return jax.vmap(lambda d:env.physics.observe(d)[1])(carry.state.physical.data)
    @jax.jit
    def bind(carry,paths,mask):
        if returning:
            new=carry._replace(native=jax.vmap(env.rebind_geometry)(carry.native,paths,speeds))
        elif online:
            native=jax.vmap(env.rebind_geometry)(carry.native,paths,speeds)
            new=(coordination.DwellOnlineCarry(native,carry.latched,carry.prev_alpha,carry.hold_until_tick) if five_online and p['protocol'].get('alpha_dwell_s',0.) else coordination.OnlineCarry(native,carry.latched,carry.prev_alpha)) if tolerance else schedule.OnlineCarry(native,carry.latched)
        else:new=jax.vmap(env.rebind_geometry)(carry,paths,speeds)
        return jax.tree.map(lambda a,b:j.where(mask.reshape((n,)+(1,)*(a.ndim-1)),a,b),new,carry)
    consumed={}
    for module in tuple(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file and Path(file).suffix=='.py' and Path(file).resolve().is_relative_to(native):
            file=str(Path(file).resolve());expected=p['native_expected_sources'].get(file)
            if expected is None or digest(file)!=expected:raise ValueError('native consumed source drift '+file)
            consumed[file]=expected
    # This is a diagnostic consistency check, not a new reward/criterion.
    terminal20=float(failure_reward(1000,cfg));terminal10=float(failure_reward(500,cfg))
    return dict(plan=p,config=cfg,cases=cases,env=env,initial=initial,paths=paths,bank=bank,speeds=speeds,
                chunk=chunk,bind=bind,measurements=measurements,planner=planner,keys=keys,
                physical_start_s=physical_start,consumed=consumed,terminal20=terminal20,terminal10=terminal10,initial_identity=initial_identity if response or capability else None,source_host=label_host if labels else None)


def flatten(logs):
    import numpy as np
    return {k:np.asarray(v).transpose((0,2,1)+tuple(range(3,v.ndim))).reshape((-1,v.shape[1])+v.shape[3:]) for k,v in logs.items()}


def run(root,*,shape_only=False):
    root=Path(root).resolve();p=verify(root)
    if p.get('label_parent'):return run_five_alpha_label_groups(root,shape_only=shape_only)
    if not shape_only and json.loads((root/'status.json').read_text())['phase']!=('prefix_completed' if p.get('mode') in ('h75_samestate_bank144','h75_late_state_tolerance48','h75_user_late_state192') else 'prepared'):raise ValueError('prepared only; no implicit rerun')
    directory=root/('shape' if shape_only else 'evaluation');directory.mkdir(exist_ok=False)
    started=time.monotonic();status=dict(phase='initializing',pid=os.getpid(),active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0)
    def update(**kw):
        status.update(kw,elapsed_s=time.monotonic()-started)
        if p.get('mode') in ('h75_samestate_bank144','h75_late_state_tolerance48','h75_user_late_state192'):
            prefix_ticks=p['protocol']['maximum_prefix_lower_ticks']
            status.update(active_prefix_lower_ticks=prefix_ticks,allocated_prefix_lower_slots=prefix_ticks,total_new_active_lower_ticks=prefix_ticks+status.get('active_lower_ticks',0),total_new_allocated_lower_slots=prefix_ticks+status.get('allocated_lower_slots',0))
        if p.get('mode')=='h75_switching_continuation':
            status.update(prior_consumed_lower_ticks=p['protocol']['consumed_source_lower_ticks'],campaign_active_lower_ticks=p['protocol']['consumed_source_lower_ticks']+status.get('active_lower_ticks',0))
        save(directory/'status.json',status)
        if not shape_only:save(root/'status.json',status)
    update()
    try:
        import jax
        import jax.numpy as j
        import numpy as np
        n=p['cases'];runtime=load(root);carry=runtime['initial'];paths=runtime['paths'];env=runtime['env']
        shaped=jax.eval_shape(runtime['chunk'],carry,paths)
        jax.eval_shape(runtime['bind'],carry,paths,j.zeros(n,bool))
        receipt=dict(active_shape=list(shaped[1]['active_tick'].shape),real_lanes=n,padding=0,physics_steps=0,
                     consumed_native_sources=runtime['consumed'],physical_start_s=runtime['physical_start_s'],
                     terminal_cost20=runtime['terminal20'],terminal_cost10=runtime['terminal10'],
                     log_template_after_wrapper=True,planner_path_type='Explicit native Path alias; only planner/vendor namespace isolated')
        if runtime['initial_identity'] is not None:receipt.update(runtime['initial_identity'])
        save(directory/'shape_receipt.json',receipt)
        if shape_only:update(phase='completed',physics_steps=0);return
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('GPU required for numerical evaluation')
        cases=runtime['cases'];path_list=[runtime['bank'][case_path_key(c)] for c in cases]
        original_path_list=list(path_list)
        if p.get('mode') in (LABEL_MODE,'h75_late_state_tolerance48','h75_user_late_state192'):path_list=[jax.tree.map(lambda x:x[i],paths) for i in range(n)]
        goals=np.array([[np.interp(float(path.goal),np.asarray(path.s),np.asarray(path.xy)[:,axis]) for axis in (0,1)] for path in original_path_list])
        original_goal_headings=np.array([np.interp(float(path.goal),np.asarray(path.s),np.asarray(path.heading)) for path in original_path_list])
        reconnection=p.get('protocol',{}).get('reconnection')
        rejected_replans=[]
        for name,path in runtime['bank'].items():np.savez_compressed(directory/(name+'_path.npz'),**{k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')})
        save(directory/'manifest.json',dict(config=p['config'],source=p['initial_source'],cases=cases,models=p['models'],
            original_goals_xy=goals.tolist(),reward_components=runtime['keys'],physics_identity=env.physics.bundle.identity,
            plan_sha256=digest(root/'plan.json'),clock='time is task pre-tick, actual_xy is post-tick; pulse uses task time plus physical_start_s',
            data_role=p['data_role'],physical_start_s=runtime['physical_start_s'],initialization=p.get('initialization'),prepared_bank_indices=p['protocol'].get('bank_indices') if 'protocol' in p else None,prepared_state_fingerprints=p['protocol'].get('bank_state_fingerprints') if 'protocol' in p else None))
        with (directory/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        all_logs=[];decisions=[];changed=np.zeros(n,bool);arrived=np.zeros(n,bool);actual=0
        bank144=p.get('mode') in ('h75_samestate_bank144','h75_late_state_tolerance48','h75_user_late_state192')
        samestate=p.get('mode') in ('h75_samestate_tolerance72','h75_samestate_bank144','h75_late_state_tolerance48','h75_user_late_state192')
        branch=p.get('mode') in ('h75_switching_continuation','h75_samestate_tolerance72','h75_samestate_bank144','h75_late_state_tolerance48','h75_user_late_state192')
        if branch:
            source=Path(p['samestate_source']['source'] if samestate else p['continuation_source']['source']);indices=[c['source_index'] for c in cases] if samestate else list(range(n))
            if not bank144:
                with np.load(source/'evaluation/reused_source_prefix.npz' if samestate else source/'evaluation/chunk_00.npz') as old:
                    prefix={k:old[k][:,indices] for k in old.files}
                if not samestate:all_logs.append(prefix)
                np.savez_compressed(directory/'reused_source_prefix.npz',**prefix)
            if samestate:
                observations=jax.jit(jax.vmap(lambda state,path,v:env.observation(state,path,v,p['horizon_s'])[0]))(carry.state,paths,runtime['speeds'])
                feature_data={'native_observation' if p.get('mode')=='h75_user_late_state192' else 'native_observation_user1':np.asarray(jax.device_get(observations))}
                if p.get('mode')=='h75_user_late_state192':feature_data['source_user_alpha']=np.asarray(carry.state.alpha)
                np.savez_compressed(directory/'source_features.npz',**feature_data)
            hostfile=root/'sources/host_state.npz' if bank144 else source/'evaluation/task1_host_state.npz'
            with np.load(hostfile) as host:
                changed=host['replanned'][indices].copy();arrived=host['ever_arrived'][indices].copy()
                if not np.array_equal(goals,host['original_goals_xy'][indices]):raise ValueError('continuation original host goals differ')
            if not samestate:
                prefix_audit=audit_stay_prefix(prefix,cases,json.loads((root/'cached_stay_records.json').read_text()),p['protocol'])
                save(directory/'source_prefix_audit.json',prefix_audit)
            save(directory/'prefix_provenance.json',dict(source=str(source),source_indices=indices,reused_lower_slots=round(p['protocol'].get('source_task_s',1.)/env.cc.dt)*n,new_physics_ticks=0,source_consumed_lower_ticks=0 if samestate else p['protocol']['consumed_source_lower_ticks'],total_budget=p['maximum_active_lower_ticks'] if samestate else p['protocol']['total_campaign_lower_ticks'],source_task_s=p['protocol'].get('source_task_s',1.),warning='Same complete source copied without physics; suffix clock remains original task time' if samestate else p['protocol']['prefix_provenance']))
            save(directory/'task1_capture_receipt.json',runtime['initial_identity'] if samestate else runtime['initial_identity']['continuation_snapshot'])
            with (directory/'task1_complete_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
            np.savez_compressed(directory/'task1_host_state.npz',replanned=changed,ever_arrived=arrived,original_goals_xy=goals)

        labels=p.get('mode')==LABEL_MODE
        if labels:
            host=runtime['source_host']
            changed=host['replanned'].copy();arrived=host['ever_arrived'].copy()
            if not np.array_equal(goals,host['original_goals_xy']):raise ValueError('imported original goals differ')
            np.savez_compressed(directory/'source_host_state.npz',**host)
            source_records=[]
            for source_index in sorted({c['source_index'] for c in cases}):
                ids=[i for i,c in enumerate(cases) if c['source_index']==source_index];i=ids[0]
                state=jax.tree.map(lambda x:x[i],carry)
                source_records.append(dict(source_index=source_index,case_indices=ids,source_lane=cases[i]['source_lane'],source_task_s=p['source_task_s'],pre_boundary_fingerprint=snapshot_fingerprint(jax.device_get(state)),ever_arrived=bool(arrived[i]),replanned=bool(changed[i]),original_goal_xy=goals[i].tolist()))

        for block in range(p['chunks']):
            if block==1 and p.get('mode')=='h75_channel_switching':
                with (directory/'task1_complete_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
                if not np.all(np.asarray(jax.device_get(carry.state.tick))==round(p['protocol']['capture_task_s']/env.cc.dt)):
                    raise ValueError('capture is not at declared task1.0 boundary')
                np.savez_compressed(directory/'task1_host_state.npz',replanned=changed,ever_arrived=arrived,original_goals_xy=goals)
                capture=assert_shared_switch_carry(carry,cases)
                cached=json.loads((root/'cached_stay_records.json').read_text())
                capture['stay_prefix']=audit_stay_prefix(all_logs[0],cases,cached,p['protocol'])
                save(directory/'task1_capture_receipt.json',capture)
            if labels or block or (branch and p.get('mode') not in ('h75_late_state_tolerance48','h75_user_late_state192')):
                poses= jax.device_get(runtime['measurements'](carry))
                cross,invalid,failed,fault,ticks=jax.device_get((carry.state.projection.cross_track,carry.state.follower.invalid,carry.state.physical.failed,carry.state.fault,carry.state.tick))
                mask=np.zeros(n,bool)
                for i in range(n):
                    if runtime['planner'].replan_requested(xy=poses[i,:2],goal_xy=goals[i],cross_track_m=float(cross[i]),
                        follower_invalid=bool(invalid[i]),already_replanned=bool(changed[i]),contact=bool(failed[i]),fault=bool(fault[i]),ever_arrived=bool(arrived[i]),scheduled_time_s=p.get('protocol',{}).get('scheduled_replan_s'),task_time_s=int(ticks[i])*env.cc.dt,continuous_replanning=p['replanning'].get('continuous_replanning',False)):
                        if int(ticks[i])%4:raise ValueError('replanning boundary was not an upper boundary')
                        path,meta=runtime['planner'].dubins_goal_path(poses[i],goals[i],radius_m=p['replanning']['radius_m'],**(dict(goal_heading_rad=float(original_goal_headings[i])) if reconnection else {}),**(dict(length_m=p['protocol']['planner_support_m'],spacing_m=p['protocol']['planner_spacing_m']) if p.get('mode') in (CAPABILITY_MODE,FIVE_ONLINE_MODE,PREFIX_MODE,LABEL_MODE,SELECTION_MODE,'h75_channel_response','h75_channel_switching','h75_switching_continuation','h75_online_scalar','h75_online_confirmation100','h75_samestate_tolerance72','h75_samestate_bank144','tolerance_online120','tolerance_online40','tolerance_broaden120','user_preference120','user_prior_online72','task_aware_fallback72','frozen_confirmation200','speed_query_online60','guard_fallback_evaluation','frozen_transfer_evaluation','h75_late_state_tolerance48','h75_user_late_state192') else {}))
                        if reconnection:
                            remaining=max(0.,float(path_list[i].goal)-float(carry.state.projection.progress[i]))
                            if float(path.goal)>remaining+reconnection['maximum_added_length_m']:
                                rejected_replans.append(dict(case_id=cases[i]['case_id'],task_tick=int(ticks[i]),remaining_path_m=remaining,candidate_length_m=float(path.goal),reason='candidate exceeds declared added-length allowance',**meta))
                                continue
                        path_list[i]=path;changed[i]=True;mask[i]=True
                        path_file=f'replanned_path_{i:02d}_tick{int(ticks[i]):04d}.npz' if p['replanning'].get('continuous_replanning',False) else f'replanned_path_{i:02d}.npz'
                        np.savez_compressed(directory/path_file,**{k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')})
                        decisions.append(dict(case_id=cases[i]['case_id'],case_index=i,path_file=path_file,task_tick=int(ticks[i]),measured_pose=np.asarray(poses[i]).tolist(),cross_track_m=float(cross[i]),follower_invalid=bool(invalid[i]),contact=bool(failed[i]),fault=bool(fault[i]),ever_arrived=bool(arrived[i]),**meta))
                if mask.any():
                    paths=jax.tree.map(lambda *x:j.stack(x),*path_list)
                    carry=runtime['bind'](carry,paths,j.asarray(mask))
                save(directory/'replanning_decisions.json',decisions)
            if labels and block==0:
                # Export current information only, after the ordinary source-boundary planner.
                observations=jax.jit(jax.vmap(lambda st,path,v:env.observation(st,path,v,p['horizon_s'])[0]))(carry.state,paths,runtime['speeds'])
                np.savez_compressed(directory/'source_features.npz',native_observation=np.asarray(observations),source_user_alpha=np.asarray(carry.state.alpha),source_world_yaw=np.asarray(runtime['measurements'](carry))[:,2])
                with (directory/'source_post_boundary_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
                with (directory/'source_post_boundary_paths.pkl').open('wb') as f:pickle.dump(jax.device_get(paths),f)
                np.savez_compressed(directory/'source_post_boundary_host.npz',replanned=changed,ever_arrived=arrived,original_goals_xy=goals)
                leaves=jax.tree.leaves(jax.device_get((carry,paths)))
                for record in source_records:
                    ids=record['case_indices'];i=ids[0]
                    if len(ids)!=20 or any(any(not np.array_equal(x[i],x[k],equal_nan=True) for k in ids[1:]) for x in leaves):raise ValueError('post-boundary source siblings differ')
                    record.update(post_boundary_fingerprint=snapshot_fingerprint(jax.device_get(jax.tree.map(lambda x:x[i],carry))),post_boundary_path_fingerprint=snapshot_fingerprint(jax.device_get(jax.tree.map(lambda x:x[i],paths))),post_boundary_replanned=bool(changed[i]),all_twenty_siblings_exact=True)
                save(directory/'source_records.json',source_records)
            if labels:
                update(phase='inflight',completed_blocks=block,active_lower_ticks=actual,allocated_lower_slots=(block+1)*20*n,inflight_cost_unknown=True)
            carry,logs=runtime['chunk'](carry,paths,upper_steps=p.get('chunk_upper_steps',[50]*p['chunks'])[block]);jax.block_until_ready(carry)
            logs=flatten(jax.device_get(logs));all_logs.append(logs)
            if labels and block==0:
                np.savez_compressed(directory/'source_features.npz',native_observation=np.asarray(observations),source_user_alpha=np.asarray(carry.state.alpha),source_world_yaw=logs['pre_upper_yaw'][0],query_observation=logs['query_observation'][0],candidate_latent=logs['composed_latent'][0])
            np.savez_compressed(directory/f'chunk_{block+int(branch):02d}.npz',**logs)
            fatal=logs['contact_failure']|logs['nonfinite']|logs['lower_fault']|logs['policy_fault']|logs['continuation_geometry_fault']
            eligible=logs['active_tick']&(np.cumsum(fatal,axis=0)==0)
            arrived|=np.any(eligible&(np.linalg.norm(logs['actual_xy']-goals[None,:,:],axis=-1)<=1.),axis=0)
            # Persist returned state before any numerical analysis can fail.
            with (directory/f'complete_carry_{block+int(branch):02d}.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
            np.savez_compressed(directory/f'host_state_{block+int(branch):02d}.npz',replanned=changed,ever_arrived=arrived,original_goals_xy=goals)
            with (directory/f'executed_paths_{block+int(branch):02d}.pkl').open('wb') as f:pickle.dump(jax.device_get(paths),f)

            actual+=int(logs['active_tick'].sum())
            if actual>p['maximum_active_lower_ticks'] or sum(p.get('chunk_upper_steps',[50]*p['chunks'])[:block+1])*4*n>p['maximum_allocated_lower_slots']:
                raise RuntimeError('declared lower-tick budget exceeded')
            update(phase='running',**({'inflight_cost_unknown':False} if labels else {}),completed_blocks=block+1,active_lower_ticks=actual,allocated_lower_slots=sum(p.get('chunk_upper_steps',[50]*p['chunks'])[:block+1])*4*n,replanned_cases=int(changed.sum()))
            if p.get('mode')=='h75_switching_continuation':
                scheduled=audit_scheduled_pairs(logs,cases)
                save(directory/'scheduled_pair_receipt.json',dict(scheduled,completed_blocks=block+1))
            if (block==1 and p.get('mode')=='h75_channel_switching') or (block==0 and branch and not samestate):
                prefix={k:np.concatenate([b[k] for b in all_logs]) for k in all_logs[0]}
                checked=audit_stay_prefix(prefix,cases,json.loads((root/'cached_stay_records.json').read_text()),p['protocol'],own_switch=True)
                save(directory/'own_switch_prefix_receipt.json',checked)
            if np.any(logs['nonfinite']|logs['lower_fault']|logs['policy_fault']|logs['continuation_geometry_fault']) or bool(j.any(carry.state.fault)):
                raise RuntimeError('engineering fault: partial raw retained, no negative label or automatic retry')
            if bool(j.all(carry.state.physical.failed)) and p.get('mode')!=PREFIX_MODE:break
        data={k:np.concatenate([b[k] for b in all_logs]) for k in all_logs[0]};(directory/'cases').mkdir();index=[]
        for i,c in enumerate(cases):
            file=directory/'cases'/f'case_{i:02d}.npz';np.savez_compressed(file,**{k:v[:,i] for k,v in data.items()})
            index.append(dict(c,raw_file=str(file),raw_sha256=digest(file),original_goal_xy=goals[i].tolist(),replanned=bool(changed[i])))
        with (directory/'final_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        save(directory/'case_index.json',index);save(directory/'replanning_decisions.json',decisions)
        if reconnection:save(directory/'replanning_rejections.json',rejected_replans)
        update(phase='completed',completed_cases=n,**({'collection_s':p['collection_s'],'task_horizon_s':p['horizon_s'],'prefix_only':True} if p.get('mode')==PREFIX_MODE else {}))
    except BaseException as exc:
        (directory/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise


def run_late_prefix(root,*,shape_only=False):
    """Import four existing task6 snapshots, apply boundary planner, no physics."""
    import jax
    import jax.numpy as j
    import numpy as np
    root=Path(root).resolve();p=verify(root);protocol=p['protocol'];late_state_inventory(protocol);n_sources=len(protocol['source_groups'])
    if not shape_only and json.loads((root/'status.json').read_text())['phase']!='prepared':raise ValueError('fresh source import only')
    out=root/('prefix_shape' if shape_only else 'prefix');out.mkdir(exist_ok=False)
    started=time.monotonic();status=dict(phase='initializing',pid=os.getpid(),active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0)
    def update(**kw):
        status.update(kw,elapsed_s=time.monotonic()-started);save(out/'status.json',status)
        if not shape_only:save(root/'status.json',status)
    update()
    try:
        runtime=load(root,prefix_phase=True);env=runtime['env'];source=Path(p['samestate_source']['source'])
        file_module('_sttw_tolerance_coordination',source/'source/tolerance_coordination.py')
        original_cases=json.loads((source/'inventory.json').read_text())
        rows=late_source_rows(original_cases,protocol['source_groups'])
        with (source/'evaluation/complete_carry_05.pkl').open('rb') as f:saved=pickle.load(f).native
        with (source/'evaluation/executed_paths_05.pkl').open('rb') as f:saved_paths=pickle.load(f)
        carry=gather_source_carry(saved,rows);paths=gather_source_carry(saved_paths,rows)
        if p.get('mode')=='h75_user_late_state192':
            if not np.array_equal(np.asarray(carry.state.alpha),np.asarray([g['source_user_alpha'] for g in protocol['source_groups']])):raise ValueError('saved source user alpha differs')
        if any(not np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(carry) if np.issubdtype(np.asarray(x).dtype,np.number)):raise ValueError('nonfinite saved late source')
        with np.load(source/'evaluation/host_state_05.npz') as host:
            changed=host['replanned'][rows].copy();arrived=host['ever_arrived'][rows].copy();goals=host['original_goals_xy'][rows].copy()
        if not bool(j.all((carry.state.tick==1200)&(carry.state.physical.physical_tick==1900))) or bool(j.any(carry.state.fault|carry.state.physical.failed)):
            raise ValueError('saved task6 source clock/failure invalid; no replacement')
        if shape_only:
            shaped=jax.eval_shape(runtime['bind'],carry,paths,j.zeros(n_sources,bool))
            save(out/'shape_receipt.json',dict(source_lanes=n_sources,native_leaves=len(jax.tree.leaves(shaped)),physics_steps=0,import_rows=rows))
            update(phase='completed',physics_steps=0);return
        with (out/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        preplanner_fingerprints=[snapshot_fingerprint(jax.device_get(jax.tree.map(lambda x:x[i],carry))) for i in range(n_sources)]
        poses=jax.device_get(runtime['measurements'](carry))
        cross,invalid,failed,fault,ticks=jax.device_get((carry.state.projection.cross_track,carry.state.follower.invalid,carry.state.physical.failed,carry.state.fault,carry.state.tick))
        path_list=[jax.tree.map(lambda x:x[i],paths) for i in range(n_sources)];mask=np.zeros(n_sources,bool);decisions=[]
        for i in range(n_sources):
            requested=runtime['planner'].replan_requested(xy=poses[i,:2],goal_xy=goals[i],cross_track_m=float(cross[i]),follower_invalid=bool(invalid[i]),already_replanned=bool(changed[i]),contact=bool(failed[i]),fault=bool(fault[i]),ever_arrived=bool(arrived[i]))
            meta={}
            if requested:
                path,meta=runtime['planner'].dubins_goal_path(poses[i],goals[i],radius_m=p['replanning']['radius_m'],length_m=protocol['planner_support_m'],spacing_m=protocol['planner_spacing_m'])
                path_list[i]=path;changed[i]=True;mask[i]=True
            decisions.append(dict(source_index=i,source_row=rows[i],task_s=6.,task_tick=int(ticks[i]),requested=bool(requested),**meta))
        if mask.any():
            paths=jax.tree.map(lambda *x:j.stack(x),*path_list);carry=runtime['bind'](carry,paths,j.asarray(mask))
        with (out/'complete_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        with (out/'executed_paths.pkl').open('wb') as f:pickle.dump(jax.device_get(paths),f)
        np.savez_compressed(out/'host_state.npz',replanned=changed,ever_arrived=arrived,original_goals_xy=goals)
        save(out/'replanning_decisions.json',decisions)
        sources=root/'sources';sources.mkdir(exist_ok=False)
        for name in ('complete_carry.pkl','executed_paths.pkl','host_state.npz'):(sources/name).write_bytes((out/name).read_bytes())
        observations=jax.jit(jax.vmap(lambda st,path,v:env.observation(st,path,v,p['horizon_s'])[0]))(carry.state,paths,runtime['speeds'])
        feature_data=dict(source_world_yaw=np.asarray(runtime['measurements'](carry))[:,2])
        if p.get('mode')=='h75_user_late_state192':feature_data.update(native_observation=np.asarray(observations),source_user_alpha=np.asarray(carry.state.alpha))
        else:feature_data['native_observation_user1']=np.asarray(observations)
        np.savez_compressed(sources/'source_features.npz',**feature_data)
        records=[]
        for i,g in enumerate(protocol['source_groups']):
            st=jax.tree.map(lambda x:x[i],carry);path=jax.tree.map(lambda x:x[i],paths)
            np.savez_compressed(sources/f'executed_source_{i:02d}_path.npz',**{k:np.asarray(getattr(path,k)) for k in ('s','xy','heading','curvature','turn_end','goal')})
            records.append(dict(g,source_file=str(out/'complete_carry.pkl'),source_row=i,original_source_file=str(source/'evaluation/complete_carry_05.pkl'),original_source_row=rows[i],original_source_fingerprint=preplanner_fingerprints[i],source_fingerprint=snapshot_fingerprint(jax.device_get(st)),native_leaf_count=len(jax.tree.leaves(st)),new_prefix_lower_ticks=0,imported_preplanner_carry_file=str(out/'initial_carry.pkl'),source_task_tick=1200,source_physical_tick=1900,planner_boundary_applied_s=6.,prepared_bank_index=g['bank_index'],prepared_state_fingerprint=protocol['bank_state_fingerprints'][str(g['bank_index'])]))
        save(sources/'records.json',records)
        files={str(f):digest(f) for f in list(sources.iterdir())+list(out.iterdir()) if f.is_file() and f.name!='status.json'}
        save(sources/'seal.json',dict(plan_sha256=digest(root/'plan.json'),files=files,active_prefix_lower_ticks=0,allocated_prefix_lower_slots=0,selection='all declared nominal source lanes saved task6; no new integration',source_task_s=6.,planner_boundary_applied_s=6.))
        update(phase='prefix_completed',completed_prefixes=n_sources,source_groups=n_sources,physics_steps=0)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise


def run_prefix(root,*,shape_only=False):
    if json.loads((Path(root)/'plan.json').read_text()).get('mode') in ('h75_late_state_tolerance48','h75_user_late_state192'):return run_late_prefix(root,shape_only=shape_only)
    """Collect declared task0..1 prefixes and freeze complete source state."""
    root=Path(root).resolve();p=verify(root)
    fresh=new_confirmation_sources(p['protocol']);prefix_budget=p['protocol']['maximum_prefix_lower_ticks'];n_prefix=8 if fresh else 4
    if p.get('mode')!='h75_samestate_bank144':raise ValueError('bank144 prefix mode required')
    if not shape_only and json.loads((root/'status.json').read_text())['phase']!='prepared':raise ValueError('prefix prepared only, no retry')
    out=root/('prefix_shape' if shape_only else 'prefix');out.mkdir(exist_ok=False)
    started=time.monotonic();status=dict(phase='initializing',pid=os.getpid(),active_lower_ticks=0,allocated_lower_slots=0,optimizer_updates=0)
    def update(**kw):
        status.update(kw,elapsed_s=time.monotonic()-started);save(out/'status.json',status)
        if not shape_only:save(root/'status.json',status)
    update()
    try:
        import jax
        import jax.numpy as j
        import numpy as np
        runtime=load(root,prefix_phase=True);carry=runtime['initial'];paths=runtime['paths'];cases=runtime['cases'];env=runtime['env']
        shaped=jax.eval_shape(runtime['chunk'],carry,paths)
        save(out/'shape_receipt.json',dict(active_shape=list(shaped[1]['active_tick'].shape),physics_steps=0,consumed_native_sources=runtime['consumed'],initial_identity=runtime['initial_identity']))
        if shape_only:update(phase='completed',physics_steps=0);return
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('GPU required for prefix numerical execution')
        with (out/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        # Allocate before dispatch; preserve evidence even if post-return checks fail.
        update(phase='running',allocated_lower_slots=prefix_budget,active_lower_ticks=None)
        carry,raw=runtime['chunk'](carry,paths);jax.block_until_ready(carry)
        logs=flatten(jax.device_get(raw));np.savez_compressed(out/'chunk_00.npz',**logs)
        with (out/'complete_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
        with (out/'executed_paths.pkl').open('wb') as f:pickle.dump(jax.device_get(paths),f)
        actual=int(logs['active_tick'].sum());update(active_lower_ticks=actual)
        goals=np.array([[np.interp(float(path.goal),np.asarray(path.s),np.asarray(path.xy)[:,axis]) for axis in (0,1)] for path in [runtime['bank'][case_path_key(c)] for c in cases]])
        arrived=np.any(logs['active_tick']&(np.linalg.norm(logs['actual_xy']-goals[None,:,:],axis=-1)<=1.),axis=0)
        np.savez_compressed(out/'host_state.npz',replanned=np.zeros(n_prefix,bool),ever_arrived=arrived,original_goals_xy=goals)
        if actual!=prefix_budget or not bool(j.all((carry.state.tick==200)&(carry.state.physical.physical_tick==900))) or bool(j.any(carry.state.fault|carry.state.physical.failed)) or (fresh and np.any(arrived)):
            raise ValueError('prefix invalid or incomplete, retain raw and stop affected collection')
        for name in ('nonfinite','lower_fault','policy_fault','continuation_geometry_fault','physical_failure','contact_failure'):
            if np.any(logs[name]):raise ValueError('prefix invalid '+name)
        if np.any(logs['applied_force_impulse']) or not np.all(logs['channel_alpha_pair']==np.array([[c['speed_alpha'],c['steer_alpha']] for c in cases])[None,:,:]):
            raise ValueError('prefix force or hybrid query contract mismatch')
        source=Path(p['samestate_source']['source']);groups=p['protocol']['source_groups']
        # Explicit module alias is already bound by load(); no modern imports.
        if not fresh:
            with (source/'evaluation/complete_carry_00.pkl').open('rb') as f:fixed=pickle.load(f).native
            with (source/'evaluation/executed_paths_00.pkl').open('rb') as f:fixed_paths=pickle.load(f)
            fixed_host=np.load(source/'evaluation/host_state_00.npz')
        prefix_group={c['source_index']:i for i,c in enumerate(cases)}
        states=[];source_paths=[];source_host=[];records=[]
        for group in groups:
            row=group.get('confirmation_row');index=group['source_index']
            if group['origin']=='fixed1' and not fresh:
                state=jax.tree.map(lambda x:x[row],fixed);path=jax.tree.map(lambda x:x[row],fixed_paths)
                host=(bool(fixed_host['replanned'][row]),bool(fixed_host['ever_arrived'][row]),fixed_host['original_goals_xy'][row])
                if host[0] or host[1]:raise ValueError('fixed source host must be original-path and unarrived')
                origin=dict(source_file=str(source/'evaluation/complete_carry_00.pkl'),source_row=row,new_prefix_lower_ticks=0)
            else:
                lane=prefix_group[index];state=jax.tree.map(lambda x:x[lane],carry);path=jax.tree.map(lambda x:x[lane],paths)
                host=(False,bool(arrived[lane]),goals[lane]);origin=dict(source_file=str(out/'complete_carry.pkl'),source_row=lane,new_prefix_lower_ticks=200)
                if fresh:origin.update(prefix_initial_carry_file=str(out/'initial_carry.pkl'),prefix_initial_source_row=lane,prepared_bank_index=group['bank_index'],prepared_state_fingerprint=p['protocol']['bank_state_fingerprints'][str(group['bank_index'])])
            if int(state.state.tick)!=200 or bool(state.state.fault|state.state.physical.failed):raise ValueError('invalid assembled source')
            expected_path=runtime['bank'][group['path_key']]
            if any(not np.array_equal(a,b,equal_nan=True) for a,b in zip(jax.tree.leaves(jax.device_get(path)),jax.tree.leaves(jax.device_get(expected_path)))):raise ValueError('source execution path differs from original bank path')
            states.append(state);source_paths.append(path);source_host.append(host)
            records.append(dict(group,source_fingerprint=snapshot_fingerprint(jax.device_get(state)),native_leaf_count=len(jax.tree.leaves(state)),**origin))
        stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
        sources=root/'sources';sources.mkdir(exist_ok=False)
        with (sources/'complete_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(stack(states)),f)
        with (sources/'executed_paths.pkl').open('wb') as f:pickle.dump(jax.device_get(stack(source_paths)),f)
        np.savez_compressed(sources/'host_state.npz',replanned=np.array([x[0] for x in source_host]),ever_arrived=np.array([x[1] for x in source_host]),original_goals_xy=np.array([x[2] for x in source_host]))
        save(sources/'records.json',records)
        files={str(f):digest(f) for f in (sources/'complete_carry.pkl',sources/'executed_paths.pkl',sources/'host_state.npz',sources/'records.json',out/'chunk_00.npz',out/'complete_carry.pkl',out/'host_state.npz',out/'executed_paths.pkl')}
        if fresh:files[str(out/'initial_carry.pkl')]=digest(out/'initial_carry.pkl')
        save(sources/'seal.json',dict(plan_sha256=digest(root/'plan.json'),files=files,active_prefix_lower_ticks=actual,allocated_prefix_lower_slots=prefix_budget,selection='fixed metadata, no labels or outcome search'))
        update(phase='prefix_completed',completed_prefixes=n_prefix,source_groups=8)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise
