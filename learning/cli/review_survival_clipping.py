"""Quantify surrogate noncommutativity on authenticated saved batches; no updates."""
import argparse,json,time,os
from pathlib import Path
from sttw_control.preference_best import atomic_json,digest
from sttw_control.survival_training_review import clipping_comparison


def review(training,output):
    output.mkdir(parents=True,exist_ok=False)
    status=dict(phase='running',pid=os.getpid(),physics_steps=0,optimizer_steps=0,stage='read_saved_batches')
    def update(**kw):
        status.update(kw,checked_unix=time.time());atomic_json(output/'status.json',status)
    update()
    from prepare_short_recovery_campaign import register
    register(output/'ACTIVE_RUN.json','STTW_CONTROL 已存PPO裁剪语义只读核对',output/'status.json')
    try:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        from sttw_control.survival_online_learner import OnlineSurvivalLearner
        cfg=json.loads((training/'effective_config.json').read_text());plan=json.loads((training/'plan.json').read_text())
        sources={str(training/'plan.json'):digest(training/'plan.json'),str(training/'effective_config.json'):digest(training/'effective_config.json')}
        rows=[]
        atomic_json(output/'plan.json',dict(project='STTW_CONTROL',purpose=__doc__,training=str(training),
            collections=[3,19],arm='physical_credit',beta=1.,fixed_denominator=plan['fixed_denominator'],
            scope='Actual final-inner-epoch ratios on original frozen training data; no surrogate replacement, model selection or efficacy evaluation.',
            code_sha256={str(p.resolve()):digest(p) for p in [Path(__file__),Path('learning/src/sttw_control/survival_training_review.py')]}))
        for i in [3,19]:
            directory=training/'collections'/f'update_{i:04d}';pairpath=training/f'checkpoint_pair_{i:04d}.json'
            pair=json.loads(pairpath.read_text());complete=json.loads((directory/'collection_complete.json').read_text())
            model=next(r for r in pair['models'] if r['arm']=='physical_credit');cp=Path(model['checkpoint'])
            datafile=directory/'physical_credit_dataset.npz'
            if digest(cp)!=model['checkpoint_sha256'] or digest(datafile)!=complete['dataset_sha256']['physical_credit']:
                raise ValueError('changed completed native model or original batch')
            obj=OnlineSurvivalLearner.restore(cfg,torch.load(cp,map_location='cpu',weights_only=False));before=obj.fingerprint()
            if before!=model['state_sha256']:raise ValueError('native content identity mismatch')
            with np.load(datafile) as z:data={k:torch.tensor(z[k],dtype=torch.float32) for k in z.files}
            adv=data['task_advantage'];adv=(adv-adv.mean())/adv.std(unbiased=False).clamp_min(1e-8)
            with torch.no_grad():
                dist=torch.distributions.Normal(obj.actor(data['obs']),obj.log_std.exp())
                ratio=(dist.log_prob(data['action']).sum(-1)-data['old_logp']).exp()
            stats=clipping_comparison(ratio.numpy(),adv.numpy(),(data['physical_targets']-data['old_physical']).numpy(),
                beta=1.,clip=plan['clip'],denominator=plan['fixed_denominator'])
            rows.append(dict(collection=i,arm='physical_credit',**stats))
            for p in [cp,datafile,pairpath,directory/'collection_complete.json']:sources[str(p.resolve())]=digest(p)
            if before!=obj.fingerprint():raise ValueError('read-only analysis mutated learner')
            update(completed_collections=len(rows))
        atomic_json(output/'results.json',rows);atomic_json(output/'receipt.json',dict(source_sha256=sources,physics_steps=0,optimizer_steps=0))
        update(phase='completed',stage='saved_ratios_compared_no_policy_change')
    except BaseException as exc:
        update(phase='error',error=repr(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();review(a.training.resolve(),a.output.resolve())
