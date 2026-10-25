"""Freeze a completed training prefix and inspect telemetry without new physics/updates."""
import argparse,json,os,time,csv
from pathlib import Path
from sttw_control.preference_best import atomic_json,digest
from sttw_control.survival_training_review import join_collection,summarize_strata,gradient_geometry


def review(training,output,*,through,gradient_updates):
    output.mkdir(parents=True,exist_ok=False)
    status=dict(phase='running',pid=os.getpid(),stage='read_metadata',physics_steps=0,optimizer_steps=0,through_update=through)
    def update(**kw):
        status.update(kw,checked_unix=time.time());atomic_json(output/'status.json',status)
    update()
    from prepare_short_recovery_campaign import register
    register(output/'ACTIVE_RUN.json','STTW_CONTROL 偏好防摔训练中期只读分解',output/'status.json')
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        from sttw_control.survival_online_learner import OnlineSurvivalLearner
        from sttw_control.survival_learning import dual_ppo_loss
        state=json.loads((training/'status.json').read_text());plan=json.loads((training/'plan.json').read_text())
        if not 1<=through<=state['completed_updates'] or any(i<3 or i>through for i in gradient_updates):
            raise ValueError('only a fully completed prefix and declared Actor collection diagnostics allowed')
        cfg=json.loads((training/'effective_config.json').read_text());sources={};rows=[];learners=[]
        def read(path):
            sources[str(path.resolve())]=digest(path);return json.loads(path.read_text())
        read(training/'plan.json');read(training/'effective_config.json')
        atomic_json(output/'plan.json',dict(project='STTW_CONTROL',purpose=__doc__,training_run=str(training),
            through_update=through,gradient_updates=gradient_updates,physical_steps=0,optimizer_steps=0,
            data_role='on_policy_training_telemetry',no_checkpoint_selection=True,
            diagnostic_scope='Same training seed, related repeated task starts; no independent endpoint efficacy claim.'))
        for i in range(1,through+1):
            directory=training/'collections'/f'update_{i:04d}'
            rows.extend(join_collection(plan['cases'],read(directory/'collection_complete.json'),update=i))
            for arm in plan['arms']:
                # Arms may be a mapping; keys identify actual learner records.
                result=read(directory/(arm+'_learner_result.json'))
                learners.append(dict(update=i,arm=arm,**{k:result[k] for k in
                    ['warmup','samples','actor_steps','critic_steps','task_advantage_center','task_advantage_scale',
                     'exact_gaussian_KL_mean','likelihood_ratio_clip_fraction']}))
        strata=summarize_strata(rows,dt_s=cfg['timing']['lower_dt_s'])
        atomic_json(output/'episodes.json',rows);atomic_json(output/'strata.json',strata);atomic_json(output/'learner_diagnostics.json',learners)
        with (output/'strata.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(strata[0]));writer.writeheader();writer.writerows(strata)
        gradients=[]
        for i in gradient_updates:
            update(stage='read_only_gradients',current_collection=i)
            directory=training/'collections'/f'update_{i:04d}';identity=read(directory/'sampling_identity.json')
            complete=read(directory/'collection_complete.json')
            for arm in ('task_only','physical_credit'):
                record=next(r for r in identity['source_pair']['models'] if r['arm']==arm)
                cp=Path(record['checkpoint']);dataset=directory/(arm+'_dataset.npz')
                if digest(cp)!=record['checkpoint_sha256'] or digest(dataset)!=complete['dataset_sha256'][arm]:
                    raise ValueError('changed frozen sampling checkpoint or actual batch')
                sources[str(cp)]=digest(cp);sources[str(dataset)]=digest(dataset)
                native=torch.load(cp,map_location='cpu',weights_only=False);obj=OnlineSurvivalLearner.restore(cfg,native)
                before=obj.fingerprint()
                if before!=identity['models'][arm]['state_sha256']:raise ValueError('wrong collection-native state')
                with np.load(dataset) as z:data={k:torch.tensor(z[k],dtype=torch.float32) for k in z.files}
                advantage=data['task_advantage'];advantage=(advantage-advantage.mean())/advantage.std(unbiased=False).clamp_min(1e-8)
                mean=obj.actor(data['obs']);distribution=torch.distributions.Normal(mean,obj.log_std.exp())
                logp=distribution.log_prob(data['action']).sum(-1)
                losses=dual_ppo_loss(new_log_prob=logp,old_log_prob=data['old_logp'],task_advantage=advantage,
                    task_values=data['old_task'],task_returns=data['task_returns'],physical_values=data['old_physical'],
                    physical_targets=data['physical_targets'],old_physical_values=data['old_physical'],
                    beta=1.,fixed_denominator=plan['fixed_denominator'],clip=plan['clip'])
                geom=gradient_geometry(losses['task_actor_loss'],losses['physical_actor_loss'],obj.groups[0])
                first=[r['vector_start'] for r in complete['case_summaries'][arm]]
                target=data['physical_targets'];prediction=data['old_physical']
                first_brier=float(((prediction[first]-target[first])**2).mean())
                gradients.append(dict(update=i,arm=arm,**geom,episode_first_value_brier=first_brier,
                    legal_action_weighted_brier=float(((prediction-target)**2).mean()),
                    old_logp_recheck_max_abs=float((logp.detach()-data['old_logp']).abs().max()),
                    max_probability=float(prediction.max()),min_probability=float(prediction.min()),
                    interpretation='Instantaneous gradients on this sampled data before clipping/Adam, not causal benefit or final policy quality.'))
                if obj.fingerprint()!=before or any(p.grad is not None for p in obj.groups[0]):
                    raise ValueError('read-only diagnostic mutated learner')
        atomic_json(output/'gradient_diagnostics.json',gradients)
        atomic_json(output/'receipt.json',dict(source_sha256=sources,episode_rows=len(rows),completed_updates=through,
            new_physics_steps=0,optimizer_updates=0,selection='none',future_training_status_not_frozen=True))
        update(phase='completed',stage='read_only_training_prefix_analyzed')
    except BaseException as exc:
        update(phase='error',error=repr(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--through',type=int,required=True);p.add_argument('--gradient-updates',type=int,nargs='+',required=True)
    a=p.parse_args();review(a.training.resolve(),a.output.resolve(),through=a.through,gradient_updates=a.gradient_updates)
