"""Check new offline task metrics against original labels on one saved real batch."""
import argparse,json,os,time
from pathlib import Path
from sttw_control.preference_best import atomic_json,digest


def audit(training,output,collection):
    output.mkdir(parents=True,exist_ok=False)
    status=dict(phase='running',stage='read_saved_real_batch',pid=os.getpid(),physics_steps=0,optimizer_steps=0)
    def update(**kw):
        status.update(kw,checked_unix=time.time());atomic_json(output/'status.json',status)
    update()
    from prepare_short_recovery_campaign import register
    register(output/'ACTIVE_RUN.json','STTW_CONTROL 终点评测原任务标签真实旧轨迹核对',output/'status.json')
    try:
        import numpy as np
        from sttw_control.survival_endpoint_task_metrics import task_metrics
        from sttw_control.recovery_margin_scan import outcome_labels
        plan=json.loads((training/'plan.json').read_text());cfg=json.loads((training/'effective_config.json').read_text())
        if json.loads((training/'status.json').read_text())['completed_updates']<collection:raise ValueError('completed collection required')
        b=cfg['stage_B_baseline']
        if [b[k] for k in ['final_y_error_m','final_path_heading_error_rad','final_speed_error_m_s']]!=[.1,.05,.1]:
            raise ValueError('original legacy constants differ from current configured task bands')
        directory=training/'collections'/f'update_{collection:04d}';complete=json.loads((directory/'collection_complete.json').read_text())
        atomic_json(output/'plan.json',dict(project='STTW_CONTROL',purpose=__doc__,training=str(training),collection=collection,
            code_sha256={str(p.resolve()):digest(p) for p in [Path(__file__),Path('learning/src/sttw_control/survival_endpoint_task_metrics.py')]},
            current_bands=b,scope='Existing real training data only; no policy endpoint or new evaluation.'))
        fields=['active_tick','task_reward_valid','physical_tick','nonfinite','lower_fault','policy_fault','domain_exit','contact_failure',
                'fallback_active','time','actual_forward_speed','path_cross_track','path_heading_error','path_progress','goal_section_signed_distance','peak_roll']
        chunks=[];sources={}
        for name,sha in sorted(complete['raw_chunks'].items()):
            p=directory/name
            if digest(p)!=sha:raise ValueError('saved raw changed: '+str(p))
            sources[str(p)]=sha
            with np.load(p) as z:chunks.append({k:z['lower_'+k] for k in fields})
        data={k:np.concatenate([c[k] for c in chunks]) for k in fields};goals={};rows=[]
        for i,case in enumerate(plan['cases']):
            route=case['route']
            if route not in goals:
                p=training/(route+'_path.npz');sources[str(p)]=digest(p)
                with np.load(p) as z:goals[route]=float(z['goal'])
            physical={k:v[:,i] for k,v in data.items()};metrics=task_metrics(physical,speed=case['speed'],goal_s=goals[route],config=cfg)
            mask=physical['active_tick'] & physical['task_reward_valid']
            olddata={k:v[mask] for k,v in physical.items()};olddata['physical_failure']=olddata['contact_failure']
            old=outcome_labels(olddata,speed=case['speed'],goal_s=goals[route],working_roll=b['working_roll_rad'],
                horizon=cfg['training_future_phase_C']['episode_s'],hold_s=b['hold_s'])
            for key in ['task_recovered','performance_success','final_hold','working_failure']:
                if metrics[key]!=old[key]:raise ValueError(f'original label mismatch case{i}: {key}')
            rows.append(dict(case_index=i,**case,**metrics))
        atomic_json(output/'results.json',rows);atomic_json(output/'receipt.json',dict(checked_cases=len(rows),matching_original_labels=True,
            source_sha256=sources,physics_steps=0,optimizer_steps=0,
            scope='Parity on actual saved collection only; no assertion that current training has completed or that task success improved.'))
        update(phase='completed',stage='actual_saved_task_labels_matched',completed_cases=len(rows))
    except BaseException as exc:
        update(phase='error',error=repr(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--collection',type=int,required=True);a=p.parse_args();audit(a.training.resolve(),a.output.resolve(),a.collection)
