"""Authenticated reuse of the one completed, Actor-unchanged warmup batch.

This deliberately rejects generic mid-PPO recovery or partially consumed second
updates. It does not restart the failed run or collect any physical trajectory.
"""
import json
from pathlib import Path
from .preference_best import digest,config_hash


def validate_first_warmup_states(done,pending,*,epochs):
    if (done['next_collection']!=1 or done['actor_steps']!=0 or done['critic_steps']!=epochs
            or done['optimizer_steps']!=[0,epochs,epochs] or done['settings']['beta']!=0.
            or done['settings']['epochs']!=epochs):
        raise ValueError('completed task-only state is not the first Critic-only boundary')
    if (pending['next_collection']!=0 or pending['actor_steps']!=0 or pending['critic_steps']!=0
            or pending['optimizer_steps']!=[0,0,0] or pending['settings']['beta']!=1.
            or pending['settings']['epochs']!=epochs):
        raise ValueError('physical-credit state was partially consumed or is not the original pending boundary')


def build_first_warmup_recovery(source,target_plan,config):
    import torch
    from .survival_online_learner import OnlineSurvivalLearner
    source=Path(source).resolve();old=json.loads((source/'plan.json').read_text())
    status=json.loads((source/'status.json').read_text())
    if status['phase']!='error' or status['completed_updates']!=0 or status['actor_steps']!=0:
        raise ValueError('recovery requires the retained first-batch error with no Actor steps')
    if 'collection old value mismatch' not in status.get('error',''):
        raise ValueError('this recovery is restricted to the diagnosed value-roundoff error')
    for key in ['cases','seed','learner_seed','updates','critic_only_updates','epochs','arms','fixed_denominator']:
        if old[key]!=target_plan[key]:raise ValueError('recovery changes declared experiment '+key)
    if config_hash(json.loads((source/'effective_config.json').read_text()))!=config_hash(config):
        raise ValueError('recovery configuration changed')
    col=source/'collections/update_0001';complete=json.loads((col/'collection_complete.json').read_text())
    if not complete['all_physical_tails_complete']:raise ValueError('physical tail incomplete')
    initial=json.loads((source/'checkpoint_pair_0000.json').read_text())
    if initial['plan_sha256']!=digest(source/'plan.json'):raise ValueError('source pair/plan changed')
    physical_record=next(x for x in initial['models'] if x['arm']=='physical_credit')
    physical=Path(physical_record['checkpoint'])
    if digest(physical)!=physical_record['checkpoint_sha256']:raise ValueError('source initial checkpoint changed')
    done=col/'task_only_post_update.pt';task_result=col/'task_only_learner_result.json'
    done_state=torch.load(done,map_location='cpu',weights_only=False)
    pending_state=torch.load(physical,map_location='cpu',weights_only=False)
    validate_first_warmup_states(done_state,pending_state,epochs=old['epochs'])
    identities=json.loads((col/'sampling_identity.json').read_text())['models']
    for name,saved in [('task_only',done_state),('physical_credit',pending_state)]:
        obj=OnlineSurvivalLearner.restore(config,saved)
        if obj.fingerprint(sampling_only=True)!=identities[name]['sampling_sha256']:
            raise ValueError('Actor/std changed; original warmup data cannot be reused')
        if name=='physical_credit' and obj.freeze_collection()['collection_id']!=identities[name]['collection_id']:
            raise ValueError('pending value/collection identity changed')
    result=json.loads(task_result.read_text())
    if not result['warmup'] or result['collection_index']!=0 or result['actor_steps']!=0 or result['critic_steps']!=old['epochs']:
        raise ValueError('completed source optimizer journal does not match saved state')
    files={name:sha for name,sha in complete['raw_chunks'].items()}
    for name,sha in complete['dataset_sha256'].items():files[name+'_dataset.npz']=sha
    for name in ['initial_carry.pkl','final_carry.pkl','sampling_identity.json','collection_complete.json']:files[name]=digest(col/name)
    for name,sha in files.items():
        if digest(col/name)!=sha:raise ValueError('source collection input changed '+name)
    rows=[r for group in complete['case_summaries'].values() for r in group]
    if len(rows)!=target_plan['total_envs'] or any(r['outcome'] not in ['contact','survived'] for r in rows):
        raise ValueError('complete known case labels required')
    if sum(r['physical_ticks'] for r in rows)!=status['active_lower_ticks']:
        raise ValueError('source physical cost receipt differs')
    return dict(kind='first_complete_critic_only_batch',source_run=str(source),source_plan_sha256=digest(source/'plan.json'),
        source_status_sha256=digest(source/'status.json'),source_collection=str(col),collection_files=files,
        initial_learners={'task_only':str(done),'physical_credit':str(physical)},
        initial_learner_sha256={'task_only':digest(done),'physical_credit':digest(physical)},
        successful_task_result=str(task_result),successful_task_result_sha256=digest(task_result),
        inherited_counters={k:status[k] for k in ['active_lower_ticks','allocated_lower_slots','legal_decisions_collected','actor_training_samples']},
        inherited_collection_elapsed_s=complete['elapsed_collection_s'],
        optimizer_reuse='load task-only completed4 Critic steps; execute only unconsumed physical-credit4 steps; no Actor replay',
        next_new_noise_update_index=1)
