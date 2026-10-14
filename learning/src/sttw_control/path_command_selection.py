"""Fixed geometric validation and immutable, task-specific best generations."""
from pathlib import Path
import json,shutil
import numpy as np
from .preference_best import atomic_json,digest
from .path_command_policy import SCHEMA

def cases(config):
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
    sha=digest(checkpoint);generation=root/'best_generations'/sha;generation.mkdir(parents=True,exist_ok=True)
    for src,name in [(checkpoint,'best_model.pt'),(actor,'best_actor.pkl')]:
        dest=generation/name
        if dest.exists() and digest(src)!=digest(dest):raise ValueError('immutable generation differs')
        if not dest.exists():shutil.copyfile(src,dest)
    record=dict(identity,schema=SCHEMA,checkpoint=str((generation/'best_model.pt').resolve()),checkpoint_sha256=sha,actor=str((generation/'best_actor.pkl').resolve()),actor_sha256=digest(generation/'best_actor.pkl'),score_tuple=score_key(summary),qualified=summary['qualified'],metrics=summary)
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
        for field in ('checkpoint','actor'):
            if digest(incumbent[field])!=incumbent[field+'_sha256']:raise ValueError('task best file SHA mismatch')
        if key>=tuple(incumbent['score_tuple']):return incumbent
    sha=digest(checkpoint);actor_sha=digest(actor)
    generation=root/('safety_task_best_generations' if safety else 'task_best_generations')/sha;generation.mkdir(parents=True,exist_ok=True)
    for src,name in [(checkpoint,'best_model.pt'),(actor,'best_actor.pkl')]:
        dest=generation/name
        if dest.exists() and digest(src)!=digest(dest):raise ValueError('immutable task generation differs')
        if not dest.exists():shutil.copyfile(src,dest)
        if digest(src)!=digest(dest):raise ValueError('task generation copy SHA mismatch')
    record=dict(identity,schema=SCHEMA,selection_schema=summary['selection_schema'],
        checkpoint=str((generation/'best_model.pt').resolve()),checkpoint_sha256=sha,
        actor=str((generation/'best_actor.pkl').resolve()),actor_sha256=actor_sha,
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
    result['qualified_task']=not any(result[k] for k in ('physical_failures','tracking_domain_failures',
        'working_failures','corridor_failures','goal_hold_failures','endpoint_primary_failures'))
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
    hashes={'checkpoint_sha256':digest(checkpoint),'actor_sha256':digest(actor)}
    if any(source.get(k)!=v for k,v in hashes.items()) or source.get('source_update',source.get('update'))!=update:
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
        if any(incumbent['panel_identity'].get(k)!=meta.get(k) for k in ('schema','mode','alphas','config_sha256')):
            raise ValueError('conditional best protocol identity mismatch')
        for k,v in identity.items():
            if k not in ('update','source_update','validation_update','validation_dir') and incumbent.get(k)!=v:
                raise ValueError('conditional learner identity mismatch')
        for kind in ('actor','checkpoint'):
            if digest(incumbent[kind])!=incumbent[kind+'_sha256']:raise ValueError('conditional best SHA mismatch')
        if key>=tuple(incumbent['score_tuple']):return incumbent
    generation=root/_generation/hashes['checkpoint_sha256'];generation.mkdir(parents=True,exist_ok=True)
    for original,name in ((checkpoint,'best_model.pt'),(actor,'best_actor.pkl')):
        dest=generation/name
        if dest.exists() and digest(original)!=digest(dest):raise ValueError('immutable conditional generation differs')
        if not dest.exists():shutil.copyfile(original,dest)
        if digest(original)!=digest(dest):raise ValueError('conditional generation copy SHA mismatch')
    record=dict(identity,schema=SCHEMA,selection_schema=_selection_schema,source_update=int(update),
        checkpoint=str((generation/'best_model.pt').resolve()),actor=str((generation/'best_actor.pkl').resolve()),
        **hashes,validation_dir=str(validation),panel_identity=meta,logical_identity=logical_identity,alphas=summary['alphas'],
        score_tuple=key,qualified_task=summary['qualified_task'],qualified=summary.get('qualified',summary['qualified_task']),
        qualified_preference=summary.get('qualified_preference'),metrics=summary)
    atomic_json(generation/'selection.json',record);atomic_json(pointer,record)
    return record


PREFERENCE_SELECTION_SCHEMA='path_strong_preference_v1'


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


def summarize_preference(panels,config):
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


def preference_score_key(summary,update=0):
    if (summary.get('schema')!=SCHEMA or summary.get('selection_schema')!=PREFERENCE_SELECTION_SCHEMA
            or not summary.get('all_declared_traces_complete')):raise ValueError('invalid strong preference identity')
    key=(summary['physical_failures']+summary['tracking_domain_failures'],summary['goal_hold_failures'],
        summary['highspeed_endpoint_failures'],summary['preference_continuity_failures'],
        summary['worst_endpoint_normalized_error'],summary['normalized_preference_deficit'],int(update))
    if not all(np.isfinite(v) for v in key):raise ValueError('nonfinite strong preference selection')
    return key


def persist_preference_best(root,checkpoint,actor,summary,identity):
    update=identity.get('source_update',identity.get('update'))
    if update is None:raise ValueError('strong preference requires source update')
    from .path_command_policy import MODULATED_ARCHITECTURE,validate_actor_parameters
    from .preference_best import config_hash
    from .path_command_evaluation import _conditional_parameter_digest
    import pickle
    if identity.get('architecture')!=MODULATED_ARCHITECTURE or identity.get('reward_version')!='path_safety_preference_v2':
        raise ValueError('strong preference architecture/reward identity mismatch')
    folder=Path(identity.get('validation_dir',Path(root)/'validation'/f'update_{int(update):04d}'))
    meta=json.loads((folder/'panel_identity.json').read_text())
    for key in ('architecture','reward_version','config_sha256'):
        if meta.get(key)!=identity.get(key):raise ValueError('strong preference panel identity mismatch '+key)
    with Path(actor).open('rb') as stream:params=pickle.load(stream)
    validate_actor_parameters(params,{'network':{'architecture':identity['architecture']}})
    if _conditional_parameter_digest(params)!=meta.get('actor_parameters_sha256'):
        raise ValueError('strong preference full Actor parameters identity mismatch')
    summary=dict(summary,score_tuple=preference_score_key(summary,update))
    return persist_conditional_best(root,checkpoint,actor,summary,identity,
        _selection_schema=PREFERENCE_SELECTION_SCHEMA,_pointer='best_preference_model.json',
        _score=lambda m:preference_score_key(m,update),_generation='preference_best_generations')
