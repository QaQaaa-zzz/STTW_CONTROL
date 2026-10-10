"""Fixed geometric validation and immutable, task-specific best generations."""
from pathlib import Path
import json,shutil
import numpy as np
from .preference_best import atomic_json,digest
from .path_command_policy import SCHEMA

def cases(config):
    e=config['stage_C_evaluation']
    return [(f'{route}_{speed:.1f}',route,speed) for route in e['route_ids'] for speed in [e['low_speed_m_s'],e['conflict_speed_m_s']]]

def summarize(traces,alpha,config):
    if set(traces)!={x[0] for x in cases(config)}:raise ValueError('complete fixed path panel required')
    horizon=config['training_future_phase_C']['episode_s'];dt=config['timing']['lower_dt_s'];b=config['stage_B_baseline'];metrics={}
    for name,route,_ in cases(config):
        d=traces[name];t=np.asarray(d['time'],float)+dt;n=len(t)
        if not n:raise ValueError('empty validation')
        if any(np.any(d[x]) for x in ['policy_fault','lower_fault','domain_exit']):raise ValueError('engineering fault cannot be ranked as physical performance')
        ev=np.asarray(d['actual_forward_speed'],float)-d['v_user'];ey=np.asarray(d['path_cross_track'],float);eh=np.asarray(d['path_heading_error'],float);roll=np.asarray(d['peak_roll'],float)
        if not all(np.isfinite(x).all() for x in [ev,ey,eh,roll]):raise ValueError('nonfinite validation')
        failure=bool(np.any(d['physical_failure']))
        if not failure and n!=round(horizon/dt):raise ValueError('partial validation cannot select best')
        tail=t>horizon-b['hold_s']+1e-5
        hold=bool(tail.sum()==round(b['hold_s']/dt) and np.all(abs(ey[tail])<=b['final_y_error_m']) and np.all(abs(eh[tail])<=b['final_path_heading_error_rad']) and np.all(abs(ev[tail])<=b['final_speed_error_m_s']) and not failure)
        goal=True if route=='straight' else bool(np.any((d['goal_section_signed_distance']>=0)&(d['path_progress']>=d['goal_progress'])))
        conflict=np.asarray(d['chi'])>0
        # Continuous ordering score, NOT an invented primary pass threshold.
        selected=conflict if conflict.any() else np.ones(n,bool)
        primary=ey/.10 if alpha==0 else ev/.05
        rms=lambda x:float(np.sqrt(np.mean(np.asarray(x)**2)))
        metrics[name]=dict(physical_failure=failure,working_failure=bool(np.any(roll>b['working_roll_rad'])),goal=goal,final_hold=hold,primary_normalized_rmse=rms(primary[selected]),primary_window='chi>0' if conflict.any() else 'full ordinary',worst_normalized_rmse=max(rms(ev/.10),rms(ey/.10),rms(eh/.05)),speed_rmse=rms(ev),path_rmse=rms(ey),heading_rmse=rms(eh),observed_steps=n,peak_roll=float(roll.max()))
    summary=dict(schema=SCHEMA,all_declared_traces_complete=True,cases=metrics,physical_failures=sum(x['physical_failure'] for x in metrics.values()),working_failures=sum(x['working_failure'] for x in metrics.values()),primary_score=max(x['primary_normalized_rmse'] for x in metrics.values()),goal_hold_failures=sum(not(x['goal'] and x['final_hold']) for x in metrics.values()),worst_error=max(x['worst_normalized_rmse'] for x in metrics.values()))
    summary['qualified']=summary['physical_failures']==summary['working_failures']==summary['goal_hold_failures']==0
    summary['qualified_scope']='declared physical/work/goal/final hold only; not alpha preference or primary-task certification'
    summary['score_tuple']=score_key(summary)
    return summary

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
