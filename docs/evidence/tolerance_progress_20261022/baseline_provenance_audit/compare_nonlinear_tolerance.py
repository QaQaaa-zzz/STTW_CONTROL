"""One fixed nonlinear-form check, same physical scales and grouped DEV splits."""
from pathlib import Path
import argparse,json,hashlib,time,os
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    path=p.parse_args().plan.resolve();plan=json.loads(path.read_text());out=path.parent
    from sttw_control.preference_best import atomic_json
    status=dict(phase='running',stage='loading',pid=os.getpid(),classifier_fits=0,
        policy_updates=0,new_physics_steps=0)
    def update(**kw):
        status.update(kw,checked_unix=time.time());atomic_json(out/'status.json',status)
    update()
    try:
        from torch.utils.tensorboard import SummaryWriter
        from sttw_control.conditional_tolerance import fit_conditional_model,predict_conditional,force_crossing
        for file,sha in plan['source_sha256'].items():
            assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==sha
        dataset=json.loads(Path(plan['dataset']).read_text());rows=dataset['rows']
        x=np.asarray([r['features']['raw_state'][:-1] for r in rows])
        amplitude=np.asarray([abs(r['force_N']) for r in rows]);target=np.asarray([r['fell'] for r in rows],float)
        families=sorted({r['family'] for r in rows});assert len(rows)==198 and len(families)==18
        writer=SummaryWriter(plan['tensorboard_logdir']);exports={};scores={};details={};index=0
        def fit(kind,train,test,tag):
            nonlocal index
            model=fit_conditional_model(x[train],amplitude[train],target[train],kind=kind,
                state_scale=plan['state_scale'],max_centers=24,kernel_width=1.,l2=1.)
            model.update(train_indices=train.tolist(),test_indices=test.tolist())
            exports[tag]=model;prediction=predict_conditional(model,x[test],amplitude[test])
            index+=1;writer.add_scalar('fit/train_log_loss',model['train_log_loss'],index)
            writer.add_scalar('fit/parameter_count',model['parameter_count'],index);writer.flush()
            details[tag]=dict(indices=test.tolist(),parameter_count=model['parameter_count'],
                nearest_center_distance=[None if not np.isfinite(v) else float(v) for v in prediction['nearest_center_distance']],
                maximum_kernel=[None if not np.isfinite(v) else float(v) for v in prediction['maximum_kernel']])
            update(stage='fitting',classifier_fits=index,fit_tag=tag)
            return prediction['probability']
        candidate={}
        for kind in ['linear','rbf']:
            train=np.asarray([i for i,r in enumerate(rows) if r['dataset']!='transient60'])
            test=np.asarray([i for i,r in enumerate(rows) if r['dataset']=='transient60'])
            prediction=np.full(len(rows),np.nan)
            prediction[test]=fit(kind,train,test,kind+'/steady_to_transient')
            scores[kind+'/steady_to_transient']=prediction
            prediction=np.full(len(rows),np.nan)
            for family in families:
                train=np.asarray([i for i,r in enumerate(rows) if r['family']!=family])
                test=np.asarray([i for i,r in enumerate(rows) if r['family']==family])
                assert not {rows[i]['snapshot_sha256'] for i in train}&{rows[i]['snapshot_sha256'] for i in test}
                prediction[test]=fit(kind,train,test,kind+'/family_'+family)
            scores[kind+'/leave_family_out']=prediction
            fit(kind,np.arange(len(rows)),np.arange(len(rows)),kind+'/all_data_candidate')
            candidate[kind]=exports[kind+'/all_data_candidate']
            crossing=force_crossing(candidate[kind],x,probability=.5)
            details[kind+'/all_data_candidate']['F50_N']=[None if not np.isfinite(v) else float(v) for v in crossing['force_N']]
            details[kind+'/all_data_candidate']['within_training_amplitude']=crossing['within_training_amplitude'].tolist()
            details[kind+'/all_data_candidate']['warning']='In-sample statistical crossing only, never certified or accepted for deployment.'
        writer.close()
        atomic_json(out/'model_exports.json',exports);atomic_json(out/'candidate_models.json',candidate)
        atomic_json(out/'support_and_crossings.json',details)
        np.savez_compressed(out/'predictions.npz',**scores,actual_contact=target)
        summary={}
        for name,prediction in scores.items():
            for maskname,mask in [('all',np.ones(len(rows),bool)),('nonzero',amplitude>0)]:
                selected=np.flatnonzero(np.isfinite(prediction)&mask);y=target[selected].astype(bool);p=prediction[selected]
                positive=p[y];negative=p[~y]
                summary[name+'/'+maskname]=dict(cases=len(selected),actual_contact=int(y.sum()),
                    false_safe=int((y&(p<.5)).sum()),conservative=int((~y&(p>=.5)).sum()),
                    AUC=float(np.mean([float(a>b)+.5*float(a==b) for a in positive for b in negative])),
                    brier=float(np.mean((p-y)**2)),
                    log_loss=float(-np.mean(np.where(y,np.log(np.clip(p,1e-12,1)),np.log(np.clip(1-p,1e-12,1))))))
        atomic_json(out/'evaluation.json',dict(summary=summary,classifier_fits=index,
            policy_updates=0,new_physics_steps=0,scope='Fixed retrospective DEV model-form check, no policy improvement.'))
        assert index==40
        update(phase='completed',stage='fixed_nonlinear_comparison_complete',classifier_fits=index)
        print(json.dumps(summary,indent=2))
    except BaseException as exc:
        update(phase='error',stage='analysis_error',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
