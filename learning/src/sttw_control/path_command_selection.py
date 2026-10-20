"""Fixed geometric validation and immutable, task-specific best generations."""
from pathlib import Path
import json,shutil
import numpy as np
from .preference_best import atomic_json
from .path_command_policy import SCHEMA

def cases(config):
    if config.get('preference_selection_schema')=='path_conflict_preference_v1':
        from .path_command_scenarios import explicit_case_declarations
        return [(d['name'],d['route_id'],float(d['speed_values'][0])) for d in explicit_case_declarations(config)]
    e=config['stage_C_evaluation']
    return [(f'{route}_{speed:.1f}',route,speed) for route in e['route_ids'] for speed in [e['low_speed_m_s'],e['conflict_speed_m_s']]]

TASK_SELECTION_SCHEMA='path_task_best_v2'
SAFETY_SELECTION_SCHEMA='path_safety_task_v1'

def task_pointer_name(config):
    version=config.get('task_selection_schema',TASK_SELECTION_SCHEMA)
    if version not in (TASK_SELECTION_SCHEMA,SAFETY_SELECTION_SCHEMA):raise ValueError('unknown task selection schema')
    return 'best_safety_task_model.json' if version==SAFETY_SELECTION_SCHEMA else 'best_task_model.json'

def summarize(traces,alpha,config,*,task=False):
    if set(traces)!={x[0] for x in cases(config)}:raise ValueError('complete fixed path panel required')
    horizon=config['training_future_phase_C']['episode_s'];dt=config['timing']['lower_dt_s'];b=config['stage_B_baseline'];metrics={}
    for name,route,_ in cases(config):
        d=traces[name];t=np.asarray(d['time'],float)+dt;n=len(t)
        if not n:raise ValueError('empty validation')
        if any(np.any(d[x]) for x in ['policy_fault','lower_fault']):raise ValueError('engineering fault cannot be ranked as physical performance')
        ev=np.asarray(d['actual_forward_speed'],float)-d['v_user'];ey=np.asarray(d['path_cross_track'],float);eh=np.asarray(d['path_heading_error'],float);roll=np.asarray(d['peak_roll'],float)
        if not all(np.isfinite(x).all() for x in [t,ev,ey,eh,roll,d['chi'],d['goal_section_signed_distance'],d['path_progress'],d['goal_progress']]):raise ValueError('nonfinite validation')
        if not np.allclose(t, np.arange(1,n+1)*dt, rtol=0, atol=dt*.05):raise ValueError('missing or misordered validation frames')
        physical=bool(np.any(d['physical_failure']));domain=bool(np.any(d['domain_exit']))
        failure=physical or domain
        terminal=np.asarray(d['physical_failure'],bool)|np.asarray(d['domain_exit'],bool)
        if terminal[:-1].any() or n>round(horizon/dt):raise ValueError('validation continues beyond terminal')
        if not failure and n!=round(horizon/dt):raise ValueError('partial validation cannot select best')
        tail=t>horizon-b['hold_s']+1e-5
        hold=bool(tail.sum()==round(b['hold_s']/dt) and np.all(abs(ey[tail])<=b['final_y_error_m']) and np.all(abs(eh[tail])<=b['final_path_heading_error_rad']) and np.all(abs(ev[tail])<=b['final_speed_error_m_s']) and not failure)
        goal=True if route=='straight' else bool(np.any((d['goal_section_signed_distance']>=0)&(d['path_progress']>=d['goal_progress'])))
        conflict=np.asarray(d['chi'])>0
        # Continuous ordering score, NOT an invented primary pass threshold.
        selected=conflict if conflict.any() else np.ones(n,bool)
        primary=ey/.10 if alpha==0 else ev/.05
        rms=lambda x:float(np.sqrt(np.mean(np.asarray(x)**2)))
        primary_ratio=(max(rms(ev/.10),rms(ey/.10)) if task and not conflict.any() else rms(primary[selected]))
        metrics[name]=dict(physical_failure=physical,tracking_domain_failure=domain,task_failure=failure,primary_band_failure=primary_ratio>1.,working_failure=bool(np.any(roll>b['working_roll_rad'])),goal=goal,final_hold=hold,primary_normalized_rmse=primary_ratio,primary_window='chi>0' if conflict.any() else 'full ordinary',worst_normalized_rmse=max(rms(ev/.10),rms(ey/.10),rms(eh/.05)),speed_rmse=rms(ev),path_rmse=rms(ey),heading_rmse=rms(eh),observed_steps=n,peak_roll=float(roll.max()))
    summary=dict(schema=SCHEMA,all_declared_traces_complete=True,cases=metrics,physical_failures=sum(x['physical_failure'] for x in metrics.values()),tracking_domain_failures=sum(x['tracking_domain_failure'] for x in metrics.values()),task_failures=sum(x['task_failure'] for x in metrics.values()),working_failures=sum(x['working_failure'] for x in metrics.values()),primary_score=max(x['primary_normalized_rmse'] for x in metrics.values()),goal_hold_failures=sum(not(x['goal'] and x['final_hold']) for x in metrics.values()),worst_error=max(x['worst_normalized_rmse'] for x in metrics.values()))
    summary['qualified']=summary['task_failures']==summary['working_failures']==summary['goal_hold_failures']==0
    summary['qualified_scope']='declared physical/work/goal/final hold only; not alpha preference or primary-task certification'
    if task:
        summary.update(selection_schema=TASK_SELECTION_SCHEMA,primary_band_failures=sum(x['primary_band_failure'] for x in metrics.values()))
        summary['qualified_task']=summary['task_failures']==summary['working_failures']==summary['primary_band_failures']==summary['goal_hold_failures']==0
        summary['qualified_preference']=None
        summary['qualified_scope']='task tolerance and safety on evaluated DEV candidates; preference requires paired evidence'
        summary['qualified']=summary['qualified_task']
    summary['score_tuple']=task_score_key(summary) if task else score_key(summary)
    return summary

def summarize_task(traces,alpha,config):
    task_pointer_name(config)
    s=summarize(traces,alpha,config,task=True)
    if config.get('task_selection_schema',TASK_SELECTION_SCHEMA)==TASK_SELECTION_SCHEMA:return s
    dt=config['timing']['lower_dt_s'];limit=config['stage_B_baseline']['working_roll_rad']
    for name,d in traces.items():
        m=s['cases'][name];excess=np.maximum(np.asarray(d['peak_roll'])-limit,0.)
        corridor=np.maximum(np.abs(np.asarray(d['path_cross_track']))-.40,0.)
        m.update(max_roll_excess_rad=float(excess.max()),roll_exceedance_seconds=float(np.sum(excess>0))*dt,
                 corridor_failure=bool(np.any(corridor>0)),max_corridor_excess_m=float(corridor.max()))
    s.update(selection_schema=SAFETY_SELECTION_SCHEMA,
             max_roll_excess_rad=max(m['max_roll_excess_rad'] for m in s['cases'].values()),
             roll_exceedance_seconds=sum(m['roll_exceedance_seconds'] for m in s['cases'].values()),
             corridor_failures=sum(m['corridor_failure'] for m in s['cases'].values()),
             max_corridor_excess_m=max(m['max_corridor_excess_m'] for m in s['cases'].values()))
    s['qualified_task']=s['qualified_task'] and s['corridor_failures']==0
    s['qualified']=s['qualified_task'];s['score_tuple']=task_score_key(s)
    return s

def task_score_key(s):
    version=s.get('selection_schema')
    if version not in (TASK_SELECTION_SCHEMA,SAFETY_SELECTION_SCHEMA):raise ValueError('invalid task selection schema')
    score_key(s)
    if version==SAFETY_SELECTION_SCHEMA:
        return (s['physical_failures'],s['tracking_domain_failures'],s['working_failures'],
                s['max_roll_excess_rad'],s['roll_exceedance_seconds'],s['corridor_failures'],
                s['max_corridor_excess_m'],s['goal_hold_failures'],s['primary_band_failures'],s['primary_score'],s['worst_error'])
    return (s['task_failures'],s['working_failures'],s['primary_band_failures'],s['goal_hold_failures'],s['primary_score'],s['worst_error'])

def score_key(s):
    if s.get('schema')!=SCHEMA or not s.get('all_declared_traces_complete'):raise ValueError('invalid geometric panel identity')
    return (s['physical_failures'],s['working_failures'],s['primary_score'],s['goal_hold_failures'],s['worst_error'])

def persist_best(root,checkpoint,actor,summary,identity):
    root=Path(root);pointer=root/'best_model.json'
    incumbent=json.loads(pointer.read_text()) if pointer.exists() else None
    if incumbent and incumbent['schema']!=SCHEMA:raise ValueError('legacy best pointer cannot be overwritten')
    if incumbent and score_key(summary)>=tuple(incumbent['score_tuple']):return incumbent
    generation=root/'best_generations'/f"update_{identity.get('source_update',identity.get('update',0)):04d}";generation.mkdir(parents=True,exist_ok=True)
    for src,name in [(checkpoint,'best_model.pt'),(actor,'best_actor.pkl')]:
        dest=generation/name
        if not dest.exists():shutil.copyfile(src,dest)
    record=dict(identity,schema=SCHEMA,checkpoint=str((generation/'best_model.pt').resolve()),checkpoint_sha256=None,actor=str((generation/'best_actor.pkl').resolve()),actor_sha256=None,score_tuple=score_key(summary),qualified=summary['qualified'],metrics=summary)
    atomic_json(generation/'selection.json',record);atomic_json(pointer,record);return record


def persist_task_best(root,checkpoint,actor,summary,identity):
    """Separate pointer and generations; legacy best remains byte-for-byte untouched."""
    root=Path(root);safety=summary.get('selection_schema')==SAFETY_SELECTION_SCHEMA
    pointer=root/('best_safety_task_model.json' if safety else 'best_task_model.json')
    key=task_score_key(summary)
    incumbent=json.loads(pointer.read_text()) if pointer.exists() else None
    if incumbent:
        if incumbent.get('schema')!=SCHEMA or incumbent.get('selection_schema')!=summary['selection_schema']:
            raise ValueError('task best pointer identity mismatch')
        if any(incumbent.get(k)!=v for k,v in identity.items() if k not in ('update','source_update','validation_update')):
            raise ValueError('task best learner/protocol identity mismatch')
        if key>=tuple(incumbent['score_tuple']):return incumbent
    generation=root/('safety_task_best_generations' if safety else 'task_best_generations')/f"update_{identity.get('source_update',identity.get('update',0)):04d}";generation.mkdir(parents=True,exist_ok=True)
    for src,name in [(checkpoint,'best_model.pt'),(actor,'best_actor.pkl')]:
        dest=generation/name
        if not dest.exists():shutil.copyfile(src,dest)
    record=dict(identity,schema=SCHEMA,selection_schema=summary['selection_schema'],
        checkpoint=str((generation/'best_model.pt').resolve()),checkpoint_sha256=None,
        actor=str((generation/'best_actor.pkl').resolve()),actor_sha256=None,
        score_tuple=key,qualified_task=summary['qualified_task'],qualified_preference=None,
        qualified=summary['qualified_task'],metrics=summary)
    # Selection may be recomputed, but weights are immutable and every pointer is atomic.
    atomic_json(generation/'selection.json',record);atomic_json(pointer,record)
    return record


def qualify_preference(traces0,traces1,paths,config):
    """Predeclared full-turn, equal-bin MSE preference gates after safety."""
    summaries=[summarize_task(traces0,0,config),summarize_task(traces1,1,config)]
    if config.get('task_selection_schema')!=SAFETY_SELECTION_SCHEMA:raise ValueError('preference qualification requires safety task schema')
    if not all(s['qualified_task'] for s in summaries):
        return dict(qualified=False,status='common_task_failed',cases={})
    result={}
    speed=config['stage_C_evaluation']['conflict_speed_m_s']
    for name,route,v in cases(config):
        if route=='straight' or v!=speed:continue
        p=paths[name];curved=np.flatnonzero(np.abs(p['curvature'])>1e-8)
        if not len(curved):raise ValueError('missing fixed turn geometry')
        start=float(config['path']['straight_prefix_m']);end=float(p['turn_end']);edges=np.append(np.arange(start,end,.5),end)
        errors=[];counts=[]
        for d in (traces0[name],traces1[name]):
            progress=np.asarray(d['path_progress']);ev=np.asarray(d['actual_forward_speed'])-d['v_user'];ey=np.asarray(d['path_cross_track'])
            bins=[];ns=[]
            for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                mask=(progress>=lo)&((progress<=hi) if i==len(edges)-2 else (progress<hi));ns.append(int(mask.sum()))
                if mask.any():bins.append([float(np.mean(ev[mask]**2)),float(np.mean(ey[mask]**2))])
            counts.append(ns)
            errors.append(None if not ns or not all(ns) else np.sqrt(np.mean(bins,axis=0)))
        if any(x is None for x in errors):result[name]=dict(qualified=False,status='incomplete_turn_bins',bin_counts=counts);continue
        dv=float(errors[0][0]-errors[1][0]);dy=float(errors[1][1]-errors[0][1])
        result[name]=dict(qualified=dv>=.03 and dy>=.02,status='evaluated',D_v_m_s=dv,D_y_m=dy,bin_counts=counts,edges_m=edges.tolist(),equal_bin_rmse=[x.tolist() for x in errors])
    return dict(qualified=len(result)==2 and all(x['qualified'] for x in result.values()),status='evaluated',cases=result)


CONDITIONAL_SELECTION_SCHEMA='path_conditional_task_v1'
CONDITIONAL_ALPHAS=(0.,.25,.5,.75,1.)


def alpha_key(value):
    """Canonical, lossless folder key; never infer alpha from a final character."""
    from decimal import Decimal
    number=float(value)
    if not np.isfinite(number) or not 0<=number<=1:raise ValueError('alpha must be finite in [0,1]')
    text=format(Decimal(str(number)).normalize(),'f')
    if number==0:text='0'
    return 'alpha_'+text.replace('.','p')


def alpha_value(key):
    if not isinstance(key,str) or not key.startswith('alpha_'):raise ValueError('invalid alpha key')
    try:value=float(key[6:].replace('p','.'))
    except ValueError as exc:raise ValueError('invalid alpha key') from exc
    if alpha_key(value)!=key:raise ValueError('noncanonical alpha key')
    return value


def conditional_grid(config):
    values=config.get('alpha_conditioning',{}).get('evaluation_alphas',CONDITIONAL_ALPHAS)
    values=tuple(float(a) for a in values)
    if values!=CONDITIONAL_ALPHAS:raise ValueError('conditional alpha grid must be 0/.25/.5/.75/1')
    return values


def summarize_conditional(panels,config):
    """Rank one shared model on all 30 cases; legacy thresholds are diagnostics.

    Interior preferences have no invented endpoint-primary acceptance threshold.
    They retain the same physical, corridor, goal and final-hold diagnostics.
    """
    if config.get('task_selection_schema')!=CONDITIONAL_SELECTION_SCHEMA:
        raise ValueError('conditional task selection schema required')
    alphas=conditional_grid(config);expected={alpha_key(a) for a in alphas}
    if set(panels)!=expected:raise ValueError('complete conditional alpha panel required')
    safety_config=dict(config,task_selection_schema=SAFETY_SELECTION_SCHEMA)
    per_alpha={};weighted=[];endpoint_primary=0
    for alpha in alphas:
        key=alpha_key(alpha);traces=panels[key]
        for d in traces.values():
            logged=np.asarray(d.get('upper_alpha',[]),float)
            if logged.shape!=np.asarray(d['time']).shape or not np.allclose(logged,alpha,rtol=0,atol=1e-7):
                raise ValueError('conditional trace alpha identity mismatch')
        s=summarize_task(traces,0 if alpha==0 else 1,safety_config)
        applicable=alpha in (0.,1.)
        for name,d in traces.items():
            m=s['cases'][name];mask=np.asarray(d['chi'])>0
            if not mask.any():mask=np.ones(len(d['time']),bool)
            ev=np.asarray(d['actual_forward_speed'],float)-np.asarray(d['v_user'],float)
            ey=np.asarray(d['path_cross_track'],float)
            rms=lambda x:float(np.sqrt(np.mean(np.square(x[mask]))))
            score=(1-alpha)*rms(ey)/.1+alpha*rms(ev)/.05
            m.update(alpha=alpha,primary_applicable=applicable,weighted_normalized_error=score,
                     demand_speed_rmse=rms(ev),demand_path_rmse=rms(ey))
            weighted.append(score)
            if not applicable:
                m['primary_band_failure']=None;m['primary_normalized_rmse']=None
        if applicable:endpoint_primary+=s['primary_band_failures']
        else:
            s['primary_band_failures']=None;s['primary_score']=None
            s['qualified_task']=not any(s[k] for k in ('physical_failures','tracking_domain_failures','working_failures','corridor_failures','goal_hold_failures'))
            s['qualified']=s['qualified_task']
        s.update(alpha=alpha,primary_applicable=applicable)
        s.pop('score_tuple',None)
        per_alpha[key]=s
    summaries=list(per_alpha.values())
    result=dict(schema=SCHEMA,selection_schema=CONDITIONAL_SELECTION_SCHEMA,
        mode='shared_conditional_v1',all_declared_traces_complete=True,
        alphas={alpha_key(a):a for a in alphas},declared_episodes=sum(len(s['cases']) for s in summaries),
        per_alpha=per_alpha,endpoint_primary_failures=endpoint_primary,primary_band_failures=endpoint_primary,
        mean_weighted_normalized_error=float(np.mean(weighted)),primary_score=float(np.mean(weighted)),
        max_roll_excess_rad=max(s['max_roll_excess_rad'] for s in summaries),
        max_corridor_excess_m=max(s['max_corridor_excess_m'] for s in summaries),
        worst_error=max(s['worst_error'] for s in summaries),qualified_preference=None,
        qualification_role='legacy diagnostic only; never a training budget stop rule')
    for k in ('physical_failures','tracking_domain_failures','task_failures','working_failures',
              'roll_exceedance_seconds','corridor_failures','goal_hold_failures'):
        result[k]=sum(s[k] for s in summaries)
    if 'reward_version' not in config['path_reward']:
        # Original Coverage has no adopted corridor constraint. Keep measurements
        # descriptive; neither rank nor qualify by an extra corridor threshold.
        result['corridor_role']='diagnostic'
        for summary in per_alpha.values():
            fields=('physical_failures','tracking_domain_failures','working_failures','goal_hold_failures')
            summary['qualified_task']=not any(summary[k] for k in fields) and (not summary['primary_band_failures'] if summary['primary_applicable'] else True)
            summary['qualified']=summary['qualified_task']
    fields=('physical_failures','tracking_domain_failures','working_failures','goal_hold_failures','endpoint_primary_failures')
    if result.get('corridor_role')!='diagnostic':fields=fields+('corridor_failures',)
    result['qualified_task']=not any(result[k] for k in fields)
    result['qualified']=result['qualified_task']
    result['score_tuple']=conditional_score_key(result)
    return result


def conditional_score_key(summary):
    if (summary.get('schema')!=SCHEMA or summary.get('selection_schema')!=CONDITIONAL_SELECTION_SCHEMA
            or not summary.get('all_declared_traces_complete')):
        raise ValueError('invalid conditional selection identity')
    fields=('physical_failures','tracking_domain_failures','working_failures','max_roll_excess_rad',
            'roll_exceedance_seconds','corridor_failures','max_corridor_excess_m','goal_hold_failures',
            'endpoint_primary_failures','mean_weighted_normalized_error','worst_error')
    if summary.get('corridor_role')=='diagnostic':fields=tuple(k for k in fields if k not in ('corridor_failures','max_corridor_excess_m'))
    summary['ranking_fields']=list(fields)
    key=tuple(summary[k] for k in fields)
    if not all(np.isfinite(v) for v in key):raise ValueError('nonfinite conditional selection')
    return key


def persist_conditional_best(root,checkpoint,actor,summary,identity,*,
        _selection_schema=CONDITIONAL_SELECTION_SCHEMA,_pointer='best_conditional_model.json',
        _score=conditional_score_key,_generation='conditional_best_generations'):
    """One atomic pointer for the same shared checkpoint across the entire grid."""
    root=Path(root);key=_score(summary)
    update=identity.get('source_update',identity.get('update'))
    if update is None:raise ValueError('conditional best requires source update identity')
    validation=Path(identity.get('validation_dir',root/'validation'/f'update_{int(update):04d}')).resolve()
    source=json.loads((validation/'source.json').read_text())
    hashes={'checkpoint_sha256':None,'actor_sha256':None}
    if source.get('source_update',source.get('update'))!=update:
        raise ValueError('conditional validation source identity mismatch')
    logical_identity={k:v for k,v in identity.items() if k not in ('update','source_update','validation_update','validation_dir')}
    if any(source.get(k)!=v for k,v in logical_identity.items()):
        raise ValueError('conditional validation source learner identity mismatch')
    meta=json.loads((validation/'panel_identity.json').read_text())
    if not meta.get('completed') or meta.get('alphas')!=summary['alphas']:
        raise ValueError('incomplete conditional validation identity')
    pointer=root/_pointer;incumbent=json.loads(pointer.read_text()) if pointer.exists() else None
    if incumbent:
        if incumbent.get('selection_schema')!=_selection_schema:raise ValueError('conditional best identity mismatch')
        if any(incumbent['panel_identity'].get(k)!=meta.get(k) for k in ('schema','mode','alphas','config_key')):
            raise ValueError('conditional best protocol identity mismatch')
        for k,v in identity.items():
            if k not in ('update','source_update','validation_update','validation_dir') and incumbent.get(k)!=v:
                raise ValueError('conditional learner identity mismatch')
        if key>=tuple(incumbent['score_tuple']):return incumbent
    generation=root/_generation/f'update_{int(update):04d}';generation.mkdir(parents=True,exist_ok=True)
    for original,name in ((checkpoint,'best_model.pt'),(actor,'best_actor.pkl')):
        dest=generation/name
        if not dest.exists():shutil.copyfile(original,dest)
    record=dict(identity,schema=SCHEMA,selection_schema=_selection_schema,source_update=int(update),
        checkpoint=str((generation/'best_model.pt').resolve()),actor=str((generation/'best_actor.pkl').resolve()),
        **hashes,validation_dir=str(validation),panel_identity=meta,logical_identity=logical_identity,alphas=summary['alphas'],
        score_tuple=key,qualified_task=summary['qualified_task'],qualified=summary.get('qualified',summary['qualified_task']),
        qualified_preference=summary.get('qualified_preference'),metrics=summary)
    if _selection_schema in (RESEARCH_PREFERENCE_SELECTION_SCHEMA,CONFLICT_PREFERENCE_SELECTION_SCHEMA):
        record.update(acceptance=summary['acceptance'],acceptance_key=summary['acceptance_key'],
            qualified_research_preference=summary['qualified_research_preference'],
            legacy_strict_preference_qualified=summary['legacy_strict_preference_qualified'])
    atomic_json(generation/'selection.json',record);atomic_json(pointer,record)
    return record


PREFERENCE_SELECTION_SCHEMA='path_strong_preference_v1'
RESEARCH_PREFERENCE_SELECTION_SCHEMA='path_preference_research_v1'
CONFLICT_PREFERENCE_SELECTION_SCHEMA='path_conflict_preference_v1'


def resolve_preference_acceptance(config):
    """Resolve only acceptance metadata; never mutate rewards or baseline gates."""
    schema=config.get('preference_selection_schema')
    legacy=dict(spatial_bin_width_m=.5,endpoint_path_rmse_m=.10,
        endpoint_speed_rmse_m_s=.05,min_speed_gap_m_s=.03,min_path_gap_m=.02,
        final_hold_s=1.,final_speed_error_m_s=.10,final_path_error_m=.10,
        final_heading_error_rad=.05,adjacent_monotonicity_role='hard',
        adjacent_speed_reverse_m_s=.01,adjacent_path_reverse_m=.01,
        roll_limit_role='diagnostic',corridor_limit_role='diagnostic')
    if schema==PREFERENCE_SELECTION_SCHEMA:return legacy
    if schema not in (RESEARCH_PREFERENCE_SELECTION_SCHEMA,CONFLICT_PREFERENCE_SELECTION_SCHEMA):raise ValueError('unknown preference selection schema')
    if schema==CONFLICT_PREFERENCE_SELECTION_SCHEMA:
        legacy.update(pressure_endpoint_path_rmse_m=.20,pressure_endpoint_speed_rmse_m_s=.15,
            pressure_min_speed_gap_m_s=.05,pressure_min_path_gap_m=.02,
            min_actual_speed_reduction_m_s=.10,max_actual_speed_offset_m_s=-.05)
    supplied=config.get('preference_acceptance')
    if not isinstance(supplied,dict) or set(supplied)!=set(legacy):
        raise ValueError('unknown or missing preference acceptance fields')
    acceptance=dict(supplied)
    for key,value in acceptance.items():
        if key.endswith('_role'):
            if value!='diagnostic':raise ValueError('research acceptance roles must be diagnostic')
        elif key=='max_actual_speed_offset_m_s':
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value>=0:
                raise ValueError('actual speed offset must be finite and negative')
        else:
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
                raise ValueError('preference acceptance numbers must be finite and positive')
    if acceptance['spatial_bin_width_m']!=.5:raise ValueError('full original turn requires 0.5m bins')
    dt=float(config['timing']['lower_dt_s']);horizon=float(config['training_future_phase_C']['episode_s'])
    if not np.isfinite(dt) or dt<=0 or not np.isfinite(horizon) or horizon<=0:
        raise ValueError('invalid preference hold timing')
    ticks=acceptance['final_hold_s']/dt
    if not np.isclose(ticks,round(ticks),rtol=0,atol=1e-8) or acceptance['final_hold_s']>horizon:
        raise ValueError('preference hold must contain integer lower ticks within horizon')
    return acceptance


def preference_hold_metrics(trace,config,acceptance):
    """Check actual post-step lower ticks at the original episode endpoint."""
    dt=float(config['timing']['lower_dt_s']);horizon=float(config['training_future_phase_C']['episode_s'])
    end_tick=round(horizon/dt);hold_ticks=round(acceptance['final_hold_s']/dt)
    times=np.asarray(trace['time'],float)
    finite_time=np.isfinite(times).all()
    tick_end=np.rint(np.where(np.isfinite(times),times,0.)/dt).astype(np.int64)+1
    mask=(tick_end>end_tick-hold_ticks)&(tick_end<=end_tick)
    complete=finite_time and np.allclose(times,(tick_end-1)*dt,rtol=0,atol=dt*.05) and np.array_equal(
        tick_end[mask],np.arange(end_tick-hold_ticks+1,end_tick+1))
    failed=any(np.any(trace[k]) for k in ('physical_failure','domain_exit'))
    fields=(('final_hold_max_speed_error_m_s',np.asarray(trace['actual_forward_speed'],float)-np.asarray(trace['v_user'],float),'final_speed_error_m_s'),
            ('final_hold_max_path_error_m',np.asarray(trace['path_cross_track'],float),'final_path_error_m'),
            ('final_hold_max_heading_error_rad',np.asarray(trace['path_heading_error'],float),'final_heading_error_rad'))
    result=dict(final_hold=False,hold_lower_ticks=int(mask.sum()),hold_window_s=float(acceptance['final_hold_s']))
    within=True
    for name,values,limit in fields:
        valid=np.isfinite(values).all()
        maximum=float(np.max(np.abs(values[mask]))) if mask.any() and valid else None
        result[name]=maximum
        within=within and maximum is not None and maximum<=acceptance[limit]
    result['final_hold']=bool(complete and not failed and within)
    return result


def _full_turn_metrics(trace,config):
    """Equal-MSE spatial bins of the immutable route, never common fragments."""
    c=config['path'];start=float(c['straight_prefix_m'])
    end=start+float(c['curvature_transition_length_m'])+abs(float(c['turn_angle_rad']))*float(c['turn_radius_m'])
    edges=np.append(np.arange(start,end,.5),end)
    progress=np.asarray(trace['path_progress'],float)
    ev=np.asarray(trace['actual_forward_speed'],float)-np.asarray(trace['v_user'],float)
    ey=np.asarray(trace['path_cross_track'],float)
    counts=[];squares=[]
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        mask=(progress>=lo)&((progress<=hi) if i==len(edges)-2 else (progress<hi))
        counts.append(int(mask.sum()))
        if mask.any():squares.append([float(np.mean(ev[mask]**2)),float(np.mean(ey[mask]**2))])
    complete=bool(counts) and all(counts)
    errors=np.sqrt(np.mean(squares,axis=0)) if complete else (None,None)
    return dict(complete=complete,bin_counts=counts,edges_m=edges.tolist(),
        speed_rmse_m_s=None if errors[0] is None else float(errors[0]),
        path_rmse_m=None if errors[1] is None else float(errors[1]))


def _summarize_strong_preference(panels,config):
    """Strong preference selection; roll/corridor remain separate diagnostics."""
    if config.get('preference_selection_schema')!=PREFERENCE_SELECTION_SCHEMA:
        raise ValueError('strong preference selection schema mismatch')
    result=summarize_conditional(panels,config)
    result.update(selection_schema=PREFERENCE_SELECTION_SCHEMA,preference_cases={},
                  highspeed_endpoint_failures=0,preference_continuity_failures=0)
    worst=0.;deficit=0.;missing=False
    high=config['stage_C_evaluation']['conflict_speed_m_s']
    names=[name for name,route,v in cases(config) if route!='straight' and v==high]
    if len(names)!=2:raise ValueError('strong preference requires high left/right turns')
    for name in names:
        metrics={key:_full_turn_metrics(panels[key][name],config) for key in result['alphas']}
        complete=all(m['complete'] for m in metrics.values())
        zero=metrics['alpha_0'];one=metrics['alpha_1']
        path=zero['path_rmse_m'];speed=one['speed_rmse_m_s']
        endpoints=[path,speed];limits=[.10,.05]
        endpoint_fail=[value is None or value>limit for value,limit in zip(endpoints,limits)]
        result['highspeed_endpoint_failures']+=sum(endpoint_fail)
        for value,limit in zip(endpoints,limits):
            if value is None:missing=True
            else:worst=max(worst,value/limit)
        dv=zero['speed_rmse_m_s']-one['speed_rmse_m_s'] if zero['complete'] and one['complete'] else None
        dy=one['path_rmse_m']-zero['path_rmse_m'] if zero['complete'] and one['complete'] else None
        gap_fail=[dv is None or dv<.03,dy is None or dy<.02]
        result['preference_continuity_failures']+=sum(gap_fail)
        deficit+=sum(1. if v is None else max(0.,(limit-v)/limit) for v,limit in [(dv,.03),(dy,.02)])
        reverse_speed=[];reverse_path=[]
        keys=list(metrics)
        for left,right in zip(keys[:-1],keys[1:]):
            a=metrics[left];b=metrics[right]
            if not (a['complete'] and b['complete']):
                reverse_speed.append(None);reverse_path.append(None)
                result['preference_continuity_failures']+=2;deficit+=2.;missing=True
                continue
            rv=b['speed_rmse_m_s']-a['speed_rmse_m_s']
            ry=a['path_rmse_m']-b['path_rmse_m']
            reverse_speed.append(rv);reverse_path.append(ry)
            result['preference_continuity_failures']+=int(rv>.01)+int(ry>.01)
            deficit+=max(0.,(rv-.01)/.01)+max(0.,(ry-.01)/.01)
        result['preference_cases'][name]=dict(per_alpha=metrics,all_alpha_complete_turn=complete,
            endpoint_main_failures=endpoint_fail,D_v_m_s=dv,D_y_m=dy,
            gap_failures=gap_fail,adjacent_speed_reverse_changes_m_s=reverse_speed,
            adjacent_path_reverse_changes_m=reverse_path)
    result.update(worst_endpoint_normalized_error=float(np.finfo(np.float32).max) if missing else worst,
        normalized_preference_deficit=deficit,missing_turn_bins=missing,
        missing_error_semantics='errors remain null; missing coverage fails criteria and receives finite maximum rank sentinel',
        qualification_role='physical/domain, goal/hold, highspeed endpoint targets and preference continuity; roll/corridor diagnostic only')
    result['qualified_preference']=not any(result[k] for k in ('physical_failures','tracking_domain_failures',
        'goal_hold_failures','highspeed_endpoint_failures','preference_continuity_failures')) and not missing
    result['qualified']=result['qualified_preference']
    result['score_tuple']=preference_score_key(result)
    return result


def summarize_preference(panels,config):
    acceptance=resolve_preference_acceptance(config)
    if config['preference_selection_schema']==CONFLICT_PREFERENCE_SELECTION_SCHEMA:
        return _summarize_conflict_preference(panels,config,acceptance)
    if config['preference_selection_schema']==PREFERENCE_SELECTION_SCHEMA:
        return _summarize_strong_preference(panels,config)
    from .path_command_policy import config_key
    legacy=_summarize_strong_preference(panels,dict(config,preference_selection_schema=PREFERENCE_SELECTION_SCHEMA))
    result=summarize_conditional(panels,config)
    result.update(selection_schema=RESEARCH_PREFERENCE_SELECTION_SCHEMA,
        qualification_role='fixed_alpha_research_behavior',acceptance=acceptance,
        acceptance_key=config_key(acceptance),preference_cases={},research_goal_hold_failures=0,
        highspeed_endpoint_failures=0,preference_direction_failures=0,incomplete_turn_conditions=0,
        legacy_strict_preference_qualified=legacy['qualified_preference'],
        legacy_strict_preference_metrics=legacy)
    # All 30 ordinary and conflict episodes retain goal and true failure requirements.
    for key,summary in result['per_alpha'].items():
        for name,m in summary['cases'].items():
            research=preference_hold_metrics(panels[key][name],config,acceptance)
            m.update(research_hold=research,legacy_final_hold=m['final_hold'],acceptance=acceptance)
            result['research_goal_hold_failures']+=int(not(m['goal'] and research['final_hold']))
    worst=0.;deficit=0.;missing=False;endpoint_missing=False
    for name,old_case in legacy['preference_cases'].items():
        metrics=old_case['per_alpha']
        incomplete=sum(not m['complete'] for m in metrics.values())
        result['incomplete_turn_conditions']+=incomplete
        missing=missing or bool(incomplete)
        zero=metrics['alpha_0'];one=metrics['alpha_1']
        endpoints=(zero['path_rmse_m'],one['speed_rmse_m_s'])
        limits=(acceptance['endpoint_path_rmse_m'],acceptance['endpoint_speed_rmse_m_s'])
        endpoint_fail=[v is None or v>limit for v,limit in zip(endpoints,limits)]
        result['highspeed_endpoint_failures']+=sum(endpoint_fail)
        for value,limit in zip(endpoints,limits):
            if value is None:endpoint_missing=True
            else:worst=max(worst,value/limit)
        dv=old_case['D_v_m_s'];dy=old_case['D_y_m']
        gaps=(acceptance['min_speed_gap_m_s'],acceptance['min_path_gap_m'])
        gap_fail=[v is None or v<limit for v,limit in zip((dv,dy),gaps)]
        result['preference_direction_failures']+=sum(gap_fail)
        deficit+=sum(1. if v is None else max(0.,(limit-v)/limit) for v,limit in zip((dv,dy),gaps))
        result['preference_cases'][name]=dict(old_case,endpoint_main_failures=endpoint_fail,
            gap_failures=gap_fail,acceptance=acceptance,
            research_hold={key:result['per_alpha'][key]['cases'][name]['research_hold'] for key in metrics},
            legacy_final_hold={key:result['per_alpha'][key]['cases'][name]['final_hold'] for key in metrics},
            legacy_strong_gap_qualified=dv is not None and dy is not None and dv>=.03 and dy>=.02)
    result.update(worst_endpoint_normalized_error=float(np.finfo(np.float32).max) if endpoint_missing else worst,
        normalized_preference_deficit=deficit,missing_turn_bins=missing,
        missing_error_semantics=legacy['missing_error_semantics'])
    result['qualified_research_preference']=not any(result[k] for k in (
        'physical_failures','tracking_domain_failures','incomplete_turn_conditions',
        'research_goal_hold_failures','highspeed_endpoint_failures','preference_direction_failures'))
    result['qualified_preference']=result['qualified_research_preference']
    result['qualified']=result['qualified_research_preference']
    result['score_tuple']=preference_score_key(result)
    return result


def preference_score_key(summary,update=0):
    if (summary.get('schema')!=SCHEMA or summary.get('selection_schema') not in (PREFERENCE_SELECTION_SCHEMA,RESEARCH_PREFERENCE_SELECTION_SCHEMA,CONFLICT_PREFERENCE_SELECTION_SCHEMA)
            or not summary.get('all_declared_traces_complete')):raise ValueError('invalid strong preference identity')
    if summary['selection_schema']==CONFLICT_PREFERENCE_SELECTION_SCHEMA:
        key=(summary['physical_failures']+summary['tracking_domain_failures'],summary['research_goal_hold_failures'],
             summary['highspeed_endpoint_failures'],summary['preference_direction_failures'],summary['old_regression_failures'],
             summary['normalized_preference_deficit'],summary['worst_endpoint_normalized_error'],int(update))
        if not all(np.isfinite(v) for v in key):raise ValueError('nonfinite conflict preference selection')
        return key
    if summary['selection_schema']==RESEARCH_PREFERENCE_SELECTION_SCHEMA:
        key=(summary['physical_failures']+summary['tracking_domain_failures'],summary['incomplete_turn_conditions'],
            summary['research_goal_hold_failures'],summary['highspeed_endpoint_failures'],summary['preference_direction_failures'],
            summary['worst_endpoint_normalized_error'],summary['normalized_preference_deficit'],int(update))
        if not all(np.isfinite(v) for v in key):raise ValueError('nonfinite research preference selection')
        return key
    key=(summary['physical_failures']+summary['tracking_domain_failures'],summary['goal_hold_failures'],
        summary['highspeed_endpoint_failures'],summary['preference_continuity_failures'],
        summary['worst_endpoint_normalized_error'],summary['normalized_preference_deficit'],int(update))
    if not all(np.isfinite(v) for v in key):raise ValueError('nonfinite strong preference selection')
    return key


def persist_preference_best(root,checkpoint,actor,summary,identity):
    update=identity.get('source_update',identity.get('update'))
    if update is None:raise ValueError('strong preference requires source update')
    from .path_command_policy import MODULATED_ARCHITECTURE,validate_actor_parameters
    from .path_command_policy import config_key
    import pickle
    if identity.get('architecture')!=MODULATED_ARCHITECTURE or identity.get('reward_version')!='path_safety_preference_v2':
        raise ValueError('strong preference architecture/reward identity mismatch')
    folder=Path(identity.get('validation_dir',Path(root)/'validation'/f'update_{int(update):04d}'))
    meta=json.loads((folder/'panel_identity.json').read_text())
    for key in ('architecture','reward_version','config_key'):
        if meta.get(key)!=identity.get(key):raise ValueError('strong preference panel identity mismatch '+key)
    if summary.get('selection_schema')==CONFLICT_PREFERENCE_SELECTION_SCHEMA:
        from .path_command_scenarios import explicit_case_declarations
        sha=identity.get('case_declarations_key')
        if not sha or meta.get('case_declarations_key')!=sha:raise ValueError('conflict preference declaration identity mismatch')
        record=json.loads((folder/'source.json').read_text())
        if record.get('case_declarations_key')!=sha:raise ValueError('conflict preference source declaration identity mismatch')
    research=summary.get('selection_schema') in (RESEARCH_PREFERENCE_SELECTION_SCHEMA,CONFLICT_PREFERENCE_SELECTION_SCHEMA)
    if research:
        acceptance=summary.get('acceptance')
        sha=config_key(acceptance)
        if summary.get('acceptance_key')!=sha or identity.get('acceptance_key')!=sha or meta.get('acceptance_key')!=sha:
            raise ValueError('research preference acceptance identity mismatch')
        # Source identity pins the SHA; the full acceptance is retained in the pointer.
        source=json.loads((folder/'source.json').read_text())
        if source.get('acceptance_key')!=sha:raise ValueError('research preference source acceptance identity mismatch')
    with Path(actor).open('rb') as stream:params=pickle.load(stream)
    validate_actor_parameters(params,{'network':{'architecture':identity['architecture']}})
    summary=dict(summary,score_tuple=preference_score_key(summary,update))
    return persist_conditional_best(root,checkpoint,actor,summary,identity,
        _selection_schema=summary['selection_schema'],_pointer='best_preference_model.json',
        _score=lambda m:preference_score_key(m,update),_generation='preference_best_generations')


def _conflict_turn_metrics(trace,config):
    """Equal weight per 0.5 m bin of each actual turn; connectors are excluded."""
    starts=np.asarray(trace['turn_starts'],float);ends=np.asarray(trace['turn_ends'],float)
    valid=np.asarray(trace['turn_mask'],bool);directions=np.asarray(trace['turn_directions'],float)
    if any(a.shape!=(4,) for a in (starts,ends,valid,directions)):
        raise ValueError('turn metadata must have four slots')
    if not all(np.isfinite(a).all() for a in (starts,ends,directions)) or not valid.any():
        raise ValueError('invalid turn metadata')
    if np.any(ends[valid]<=starts[valid]) or np.any(np.abs(directions[valid])!=1):raise ValueError('invalid turn interval')
    if np.any(starts[valid][1:]<ends[valid][:-1]):raise ValueError('overlapping turn intervals')
    progress=np.asarray(trace['path_progress'],float);speed=np.asarray(trace['actual_forward_speed'],float)
    request=np.broadcast_to(np.asarray(trace['v_user'],float),speed.shape)
    path=np.asarray(trace['path_cross_track'],float);offset=np.asarray(trace['offsets'],float)[:,0]
    arrays=(progress,speed,request,path,offset)
    if any(a.shape!=speed.shape for a in arrays) or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('nonfinite or malformed turn trace')
    for key in ('physical_failure','domain_exit'):
        terminal=np.asarray(trace.get(key,np.zeros(len(speed),bool)),bool)
        if terminal[:-1].any():raise ValueError('turn trace continues after failure')
    counts=[];edges=[];rows=[]
    for lo,hi in zip(starts[valid],ends[valid]):
        edge=np.append(np.arange(lo,hi,.5),hi);ns=[]
        for i,(a,b) in enumerate(zip(edge[:-1],edge[1:])):
            mask=(progress>=a)&((progress<=b) if i==len(edge)-2 else progress<b)
            ns.append(int(mask.sum()))
            if mask.any():rows.append([np.mean((speed[mask]-request[mask])**2),np.mean(path[mask]**2),
                                      np.mean(speed[mask]),np.mean(request[mask]),np.mean(offset[mask])])
        counts.append(ns);edges.append(edge.tolist())
    complete=all(all(n) for n in counts)
    means=np.mean(rows,axis=0) if complete else [None]*5
    return dict(complete=complete,turn_bin_counts=counts,bin_counts=[n for turn in counts for n in turn],
        turn_edges_m=edges,speed_rmse_m_s=float(np.sqrt(means[0])) if complete else None,
        path_rmse_m=float(np.sqrt(means[1])) if complete else None,
        mean_actual_speed_m_s=float(means[2]) if complete else None,
        mean_original_speed_m_s=float(means[3]) if complete else None,
        mean_actual_speed_offset_m_s=float(means[4]) if complete else None)


def _summarize_conflict_preference(panels,config,acceptance):
    from .path_command_policy import config_key
    names=[c[0] for c in cases(config)];old_names=names[:6]
    if len(names)!=12 or set(panels)!={alpha_key(a) for a in conditional_grid(config)}:
        raise ValueError('complete 12-case five-alpha conflict panel required')
    if any(set(panel)!=set(names) for panel in panels.values()):raise ValueError('complete conflict cases required')
    old_cfg=dict(config,preference_selection_schema=RESEARCH_PREFERENCE_SELECTION_SCHEMA)
    old_cfg['preference_acceptance']={k:v for k,v in acceptance.items() if not k.startswith('pressure_') and k not in ('min_actual_speed_reduction_m_s','max_actual_speed_offset_m_s')}
    old_panels={key:{n:panel[n] for n in old_names} for key,panel in panels.items()}
    old=summarize_preference(old_panels,old_cfg)
    result=summarize_conditional(panels,config)
    result.update(selection_schema=CONFLICT_PREFERENCE_SELECTION_SCHEMA,acceptance=acceptance,
        acceptance_key=config_key(acceptance),preference_cases={},research_goal_hold_failures=0,
        highspeed_endpoint_failures=0,preference_direction_failures=0,incomplete_turn_conditions=0,
        old_regression_failures=old['highspeed_endpoint_failures']+old['incomplete_turn_conditions'],
        old_research_metrics=old,legacy_strict_preference_qualified=old['legacy_strict_preference_qualified'],
        legacy_strict_preference_metrics=old['legacy_strict_preference_metrics'],
        qualification_role='diagnostic_continue_declared_budget',scientific_failure_action='continue_declared_200_updates')
    for key,summary in result['per_alpha'].items():
        for name,m in summary['cases'].items():
            hold=preference_hold_metrics(panels[key][name],config,acceptance)
            m.update(research_hold=hold,legacy_final_hold=m['final_hold'])
            result['research_goal_hold_failures']+=int(not(m['goal'] and hold['final_hold']))
    deficit=0.;worst=0.;missing=False
    for name in ('left180_R1p8_2.8','right180_R1p8_2.8','compact_LR_2.8','compact_RL_2.8'):
        metrics={key:_conflict_turn_metrics(panel[name],config) for key,panel in panels.items()}
        incomplete=sum(not m['complete'] for m in metrics.values());result['incomplete_turn_conditions']+=incomplete
        missing|=bool(incomplete);zero=metrics['alpha_0'];one=metrics['alpha_1']
        ends=(zero['path_rmse_m'],one['speed_rmse_m_s'])
        limits=(acceptance['pressure_endpoint_path_rmse_m'],acceptance['pressure_endpoint_speed_rmse_m_s'])
        ef=[v is None or v>lim for v,lim in zip(ends,limits)];result['highspeed_endpoint_failures']+=sum(ef)
        worst=max(worst,*[float(np.finfo(np.float32).max) if v is None else v/lim for v,lim in zip(ends,limits)])
        dv=zero['speed_rmse_m_s']-one['speed_rmse_m_s'] if zero['complete'] and one['complete'] else None
        dy=one['path_rmse_m']-zero['path_rmse_m'] if zero['complete'] and one['complete'] else None
        reduction=zero['mean_original_speed_m_s']-zero['mean_actual_speed_m_s'] if zero['complete'] else None
        values=(dv,dy,reduction,-zero['mean_actual_speed_offset_m_s'] if zero['complete'] else None)
        mins=(acceptance['pressure_min_speed_gap_m_s'],acceptance['pressure_min_path_gap_m'],
              acceptance['min_actual_speed_reduction_m_s'],-acceptance['max_actual_speed_offset_m_s'])
        failures=[v is None or v<lim for v,lim in zip(values,mins)]
        result['preference_direction_failures']+=sum(failures)+incomplete
        deficit+=sum(1. if v is None else max(0.,(lim-v)/lim) for v,lim in zip(values,mins))
        adjacent=list(zip(metrics.values(),list(metrics.values())[1:]))
        reverse_speed=[b['speed_rmse_m_s']-a['speed_rmse_m_s'] if a['complete'] and b['complete'] else None for a,b in adjacent]
        reverse_path=[a['path_rmse_m']-b['path_rmse_m'] if a['complete'] and b['complete'] else None for a,b in adjacent]
        result['preference_cases'][name]=dict(per_alpha=metrics,all_alpha_complete_turn=not incomplete,
            endpoint_main_failures=ef,D_v_m_s=dv,D_y_m=dy,gap_failures=failures[:2],
            behavior_failures=failures[2:],alpha0_actual_speed_reduction_m_s=reduction,acceptance=acceptance,
            adjacent_speed_reverse_changes_m_s=reverse_speed,adjacent_path_reverse_changes_m=reverse_path)
    result.update(worst_endpoint_normalized_error=worst,normalized_preference_deficit=deficit,missing_turn_bins=missing,
        missing_error_semantics='missing bins remain null and fail qualification; finite maximum rank sentinel')
    result['qualified_research_preference']=not any(result[k] for k in ('physical_failures','tracking_domain_failures',
        'research_goal_hold_failures','highspeed_endpoint_failures','preference_direction_failures','old_regression_failures'))
    result['qualified_preference']=result['qualified_research_preference'];result['qualified']=result['qualified_preference']
    result['score_tuple']=preference_score_key(result)
    return result
