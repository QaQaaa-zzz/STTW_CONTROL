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
