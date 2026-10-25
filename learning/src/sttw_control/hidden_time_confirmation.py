"""Fixed DEV timing screen for authenticated existing Actors, no training."""
import json,pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .preference_best import atomic_json,digest


def timing_cases(route,*,speed,world_direction):
    if route not in ('left90_R1p1','right90_R1p1'):raise ValueError('declared original route required')
    if speed!=({'left90_R1p1':1.8,'right90_R1p1':1.5}[route]):raise ValueError('original request required')
    d=np.asarray(world_direction,float)
    if d.shape!=(3,) or not np.isfinite(d).all() or not np.isclose(np.linalg.norm(d),1.):raise ValueError('unit original world direction required')
    roles=[('parent','parent',0,1.),('task_only','task_only',0,1.),('physical_credit','physical_credit',0,1.),
           ('floor','parent',0,1.5),('causal_eso_allalpha','parent',5,1.),('zero_middle','parent',0,1.)]
    events=[('zero',0.,380)]+[(f't{t}_F{f:+g}',f,t) for t in [380,580] for f in [-32.,32.]]
    rows=[]
    for role,actor,profile,floor in roles:
        for alpha in ([None] if role=='zero_middle' else [.5,.75]):
            for event,force,onset in events:
                state_alpha=0. if alpha is None else alpha
                rows.append(dict(case_id=f'{role}/alpha{alpha}/{event}',role=role,actor_role=actor,
                    alpha=alpha,state_alpha=state_alpha,source_batch_slot=[0.,.5,.75].index(state_alpha),
                    profile=profile,reference_speed_floor_m_s=floor,zero_mode=role=='zero_middle',
                    event_id=event,force_N=force,pulse_onset_task_tick=onset,pulse_duration_s=.1,
                    world_direction=d.tolist(),route=route,requested_speed_m_s=speed,horizon_s=10.,stage='timing'))
    return rows


def historical_source_receipt(expected,archives):
    """Verify exact old source bytes, distinguishing current comparator code."""
    archived=[p for directory in archives for p in Path(directory).rglob('*') if p.is_file()]
    receipt=[]
    for path,sha in expected.items():
        live=Path(path);matches=live.is_file() and digest(live)==sha
        found=live if matches else next((p for p in archived if p.name.endswith(live.name) and digest(p)==sha),None)
        if found is None:raise ValueError('missing exact historical source: '+path)
        receipt.append(dict(original_path=path,sha256=sha,verified_path=str(found),live_matches=matches))
    return receipt


def authenticate_saved_endpoint(source,training):
    """Recheck original receipt and actual native/Actor contents; never alter old plans."""
    import torch
    from .survival_endpoint_models import verify_native_export
    from .survival_endpoint_protocol import validate_endpoint_metadata
    from .preference_best import config_hash
    auth=json.loads((source/'authenticated_models.json').read_text());plan=json.loads((training/'plan.json').read_text())
    if digest(training/'plan.json')!=auth['training_plan_sha256'] or digest(auth['endpoint_pair'])!=auth['endpoint_pair_sha256']:
        raise ValueError('historical training plan/pair drift')
    status=json.loads((training/'status.json').read_text())
    if status['phase']!='completed' or status['completed_updates']!=64:raise ValueError('fixed64 completion required')
    if digest(training/'effective_config.json')!=plan['effective_config_sha256']:raise ValueError('training config drift')
    config=json.loads((training/'effective_config.json').read_text())
    if config!=auth['config']:raise ValueError('authenticated config drift')
    sources=historical_source_receipt(plan['source_sha256'],[training/'source_at_launch',source/'source_snapshot'])
    for key in ['source_plan','source_declaration']:
        if digest(plan[key])!=plan[key+'_sha256']:raise ValueError('training provenance drift')
    pair=json.loads(Path(auth['endpoint_pair']).read_text());native={}
    for model in pair['models']:
        for kind in ['actor','checkpoint']:
            if digest(model[kind])!=model[kind+'_sha256'] or auth['models'][model['arm']][kind+'_sha256']!=model[kind+'_sha256']:
                raise ValueError('fixed model identity drift')
        native[model['arm']]=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
        with Path(model['actor']).open('rb') as f:exported=pickle.load(f)
        verify_native_export(config,native[model['arm']],exported,model)
    validate_endpoint_metadata(plan,pair,native,plan_sha256=digest(training/'plan.json'),config_sha256=config_hash(config))
    for k,v in pair['cost_counters'].items():
        if status[k]!=v:raise ValueError('training cost drift')
    parent=auth['models']['parent']
    for kind in ['actor','checkpoint']:
        if digest(parent[kind])!=parent[kind+'_sha256']:raise ValueError('parent model drift')
    return auth,dict(historical_sources=sources,old_authenticated_receipt_sha256=digest(source/'authenticated_models.json'),
        native_export_rechecked=['task_only','physical_credit'],recovery_history='Inherited authenticated receipt; not replayed or retrained',physics_steps=0,optimizer_updates=0)


def prepare_timing_confirmation(source,output):
    """Extract real task0 complete states only; no reset, integration or selection."""
    import jax
    source=Path(source).resolve();output=Path(output).resolve()
    prior=json.loads((source/'status.json').read_text())
    if prior['phase']!='completed' or prior['completed_cases']!=488:raise ValueError('completed original488 required')
    oldplan=json.loads((source/'plan.json').read_text());training=Path(oldplan['protocol_plan']['training_run'])
    auth,identity_receipt=authenticate_saved_endpoint(source,training);cfg=json.loads((source/'common_environment_config.json').read_text())
    if cfg!=auth['config']:raise ValueError('original endpoint config drift')
    train=json.loads((training/'plan.json').read_text());declaration=Path(train['source_declaration'])
    cases=json.loads((source/'cases.json').read_text());groups=json.loads((source/'groups.json').read_text())
    index={x['case_id']:x for x in json.loads((source/'case_index.json').read_text())}
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'historical_identity_receipt.json',identity_receipt)
    sources=list(Path(__file__).parent.glob('*.py'))+[Path('learning/cli/evaluate_anticipatory_reference.py').resolve()]
    total=[]
    for short,route,speed in [('left','left90_R1p1',1.8),('right','right90_R1p1',1.5)]:
        root=output/short;root.mkdir();states=[];origin=[]
        for alpha in [0.,.5,.75]:
            old=next(c for c in cases if c['panel']=='physical_deterministic' and c['route']==route and c['role']=='parent' and c['alpha']==alpha and c['force_N']==0.)
            pos=cases.index(old);g=next(g for g in groups if pos in g['case_indices']);lane=g['case_indices'].index(pos)
            initial=source/g['group_id']/'initial.pkl'
            with initial.open('rb') as f:carry=pickle.load(f)
            s=jax.tree.map(lambda x:np.asarray(x[lane]),carry.state)
            if int(s.tick)!=0 or bool(s.fault|s.domain_exit|s.physical.failed) or abs(float(s.alpha)-alpha)>1e-7:raise ValueError('valid actual task0 source required')
            states.append(s);origin.append(dict(source_case_id=old['case_id'],initial_file=str(initial),initial_sha256=digest(initial),lane=lane,
                task_tick=int(s.tick),physical_tick=int(s.physical.physical_tick),alpha=alpha))
        bundle=root/'origin_states.pkl'
        with bundle.open('wb') as f:pickle.dump(SimpleNamespace(state=jax.tree.map(lambda *x:np.stack(x),*states)),f)
        data=timing_cases(route,speed=speed,world_direction=old['world_direction'])
        pathfile=source/index[old['case_id']]['reference']
        (root/'common_environment_config.json').write_bytes((source/'common_environment_config.json').read_bytes())
        (root/'source_declaration.json').write_bytes(declaration.read_bytes())
        atomic_json(root/'authenticated_models.json',auth)
        atomic_json(root/'inventory.json',dict(project='STTW_CONTROL',source_run=str(source),origin=origin,cases=data))
        atomic_json(root/'origins.json',origin)
        archive=root/'source_snapshot';archive.mkdir()
        for i,p in enumerate(sorted(sources)):(archive/f'{i:03d}_{p.name}').write_bytes(p.read_bytes())
        plan=dict(project='STTW_CONTROL',data_role='DEV',purpose='Confirm existing joint-task signal at unseen event times versus strong matched rules; not a new trained method.',
            maximum_active_lower_ticks=110000,optimizer_updates=0,batch_size=16,steps_per_chunk=50,max_chunks=10,horizon_s=10.,lower_dt_s=.005,
            stage_counts=dict(timing=55),logical_cases=55,inherited_cases=0,
            sealed_files={str((root/name).resolve()):digest(root/name) for name in ['common_environment_config.json','source_declaration.json','authenticated_models.json','inventory.json','origins.json']},
            source_sha256={str(p.resolve()):digest(p) for p in sources},source_initial_state=str(bundle),source_initial_state_sha256=digest(bundle),
            path_file=str(pathfile),path_sha256=digest(pathfile),
            origin_source_sha256={x['initial_file']:x['initial_sha256'] for x in origin},
            information='No true pulse/time/direction passed to Actor or causal rule; only integrator receives pulse. Same original useralpha in state and cost.',
            contract='Original common50Hz/4x5ms live reference kernel, domain exit zero next upper boundary for all roles,10s contact-only horizon. Historical488held results not directly substituted.',
            stop_rule='55 perroute only, no training/tuning/retries/automatic expansion. Engineering fault stops stage and preserves logs.',
            inference='Previously chosen twoalpha, two related routes and one common parent origin. DEV finite pulse screen, not independent confirmation, force threshold, safety certificate or isolated causal mechanism.')
        atomic_json(root/'plan.json',plan)
        atomic_json(root/'status.json',dict(phase='prepared',declared_cases=55,completed_cases=0,active_lower_ticks=0,optimizer_updates=0))
        total.append(dict(route=route,run=str(root),plan_sha256=digest(root/'plan.json'),cases=55))
    atomic_json(output/'plan.json',dict(project='STTW_CONTROL',routes=total,maximum_new_lower_ticks=220000,new_cases=110,optimizer_updates=0,
        source_run=str(source),source_results_sha256=digest(source/'results.json'),source_training_run=str(training),
        causal_rule='profile5 explicit all-preferences counterpart of originalprofile4; original alpha0 gate untouched by default',
        disposition='Stop expansion if floor/ESO cover gains; repeatable joint-task increment precedes prefix/post mechanism and independent testing.'))
    atomic_json(output/'status.json',dict(phase='prepared',declared_cases=110,completed_cases=0,active_lower_ticks=0,optimizer_updates=0))
