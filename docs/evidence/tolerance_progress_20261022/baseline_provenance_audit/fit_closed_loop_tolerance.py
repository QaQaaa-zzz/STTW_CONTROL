"""Fixed small conditional classifiers on authenticated historical lower-only data.

Force is a hypothetical query to the capability model, not a future measurement
supplied to the Actor. These DEV fits never change an existing policy.
"""
from pathlib import Path
import argparse
import json
import hashlib
import os
import time
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    planfile=p.parse_args().plan.resolve();plan=json.loads(planfile.read_text());out=planfile.parent
    from sttw_control.preference_best import atomic_json
    status=dict(phase='running',stage='assembling',pid=os.getpid(),policy_updates=0,
                new_physics_steps=0,classifier_fits=0)
    def update(**kw):
        status.update(kw,checked_unix=time.time());atomic_json(out/'status.json',status)
    update()
    try:
        for source,sha in plan['source_sha256'].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=sha:
                raise ValueError('declared analysis source changed: '+source)
        from scipy.optimize import minimize
        from scipy.special import expit
        from torch.utils.tensorboard import SummaryWriter
        from sttw_control.balance_probe import classify_contact_balance
        from sttw_control.tolerance_state_features import unstable_features
        sources,rows,lower_ids=[],[],set()
        snapshot_hashes={}
        def read(path):
            path=Path(path);raw=path.read_bytes()
            sources.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
            return json.loads(raw)
        for spec in plan['datasets']:
            run=Path(spec['run']);scan=read(run/spec['plan_file'])
            manifest=read(run/'manifest.json')
            assert scan['pulse_duration_s']==.1 and scan['observation_s']==4.
            assert scan['termination']=='contact_only_v1'
            assert manifest['no_upper_inference'] and manifest['no_path_domain_termination']
            decl=read(scan['source_declaration']);cc=decl['source']['controller'];ac=decl['source']['actuator']
            lower_ids.add(decl['source']['candidate']['actor_sha256'])
            prefixes={r['state_id']:r for r in read(spec['prefix_file'])['cases']}
            observed=read(run/'results.json')['cases']
            assert len(observed)==len(scan['cases'])==spec['expected_cases']
            for case,result,snapshot in zip(scan['cases'],observed,scan['case_snapshots']):
                prefix=prefixes[case['state_id']]
                path=run/(result['name']+'.npz');raw=path.read_bytes()
                sources.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()))
                with np.load(path) as d:
                    label=classify_contact_balance(d,horizon_s=4.)
                    assert label['balance_label']==result['balance_label']
                    assert label['active_ticks']==result['active_ticks']
                    assert label['balance_label'] in ('FLOOR_CONTACT','NO_FLOOR_CONTACT')
                    assert np.isfinite(d['applied_force_impulse']).all()
                if snapshot not in snapshot_hashes:
                    snapshot_hashes[snapshot]=hashlib.sha256(Path(snapshot).read_bytes()).hexdigest()
                    sources.append(dict(path=snapshot,sha256=snapshot_hashes[snapshot]))
                state=unstable_features(speed_proxy=prefix['wheel_proxy_m_s'],steer=prefix['steer_rad'],
                    roll=prefix['roll_rad'],roll_rate=prefix['roll_rate_rad_s'],
                    disturbance=prefix['controller_disturbance'],controller=cc)
                v=state['model_speed_proxy'];force=float(case['force_N']);side=float(np.sign(force))
                delta=prefix['steer_rad'];lam=state['lambda_u'];b=state['b_u'];T=.3
                lo=max(-ac['steer_rate_limit'],(-ac['steer_limit']-delta)/T)
                hi=min(ac['steer_rate_limit'],(ac['steer_limit']-delta)/T)
                q=-np.expm1(-lam*T)/lam;K=.7192241548921705*(-np.expm1(-lam*.1))/lam
                fplus=(-q*min(b*lo,b*hi)-state['z_u'])/K
                fminus=(state['z_u']+q*max(b*lo,b*hi))/K
                cap=fplus if force>0 else fminus if force<0 else min(fplus,fminus)
                rawstate=[v,v*v,side*delta,side*prefix['steer_rate_rad_s'],side*prefix['roll_rad'],
                    side*prefix['roll_rate_rad_s'],side*prefix['controller_disturbance'],
                    case['reference'][0]-v,side*(case['reference'][1]-delta)]
                features=dict(speed_force=[v,v*v,abs(force)],
                    raw_state=rawstate+[abs(force)],
                    generic_augmented=rawstate+[v*side*prefix['roll_rad'],v*side*prefix['roll_rate_rad_s'],abs(force)],
                    physical_augmented=rawstate+[side*state['z_u'],cap,abs(force)])
                assert all(np.isfinite(x).all() for x in features.values())
                rows.append(dict(dataset=spec['name'],state_id=case['state_id'],
                    family=prefix.get('source_state_id',case['state_id']),snapshot=snapshot,
                    snapshot_sha256=snapshot_hashes[snapshot],
                    force_N=force,reference=case['reference'],fell=label['balance_label']=='FLOOR_CONTACT',
                    active_ticks=label['active_ticks'],features=features,analytic_cap_N=cap,
                    original_npz=str(path)))
        assert len(lower_ids)==1 and len(rows)==198
        atomic_json(out/'dataset.json',dict(rows=rows,sources=sources,lower_actor_sha256=next(iter(lower_ids)),
            policy_updates=0,new_physics_steps=0,scope='Historical DEV; shared ancestor. Complete 4s no-contact versus actual contact; no task labels.'))
        models=plan['models'];names=sorted(set(r['family'] for r in rows))
        allidx=np.arange(len(rows));y=np.asarray([r['fell'] for r in rows],float)
        predictions={};exports={};fit_index=0
        writer=SummaryWriter(str(out/'tensorboard'/'conditional_models'))
        def fit_predict(model,train,test,tag):
            nonlocal fit_index
            X=np.asarray([r['features'][model] for r in rows]);mean=X[train].mean(0);scale=X[train].std(0)
            scale=np.where(scale>1e-8,scale,1.)
            xt=(X[train]-mean)/scale;xx=(X[test]-mean)/scale
            def objective(theta):
                logit=xt@theta[:-1]+theta[-1];p=expit(logit)
                loss=np.logaddexp(0,logit).sum()-y[train]@logit+.5*np.square(theta[:-1]).sum()
                grad=np.r_[xt.T@(p-y[train])+theta[:-1],(p-y[train]).sum()]
                return loss,grad
            bounds=[(None,None)]*(xt.shape[1]+1);bounds[-2]=(0,None)
            opt=minimize(objective,np.zeros(xt.shape[1]+1),jac=True,method='L-BFGS-B',bounds=bounds,
                options=dict(maxiter=500,ftol=1e-10,gtol=1e-7))
            if not opt.success:raise RuntimeError('fixed classifier fit did not converge: '+str(opt.message))
            score=expit(xx@opt.x[:-1]+opt.x[-1]);fit_index+=1
            unpenalized=float(np.mean(np.logaddexp(0,xt@opt.x[:-1]+opt.x[-1])-y[train]*(xt@opt.x[:-1]+opt.x[-1])))
            writer.add_scalar('fit/train_log_loss',unpenalized,fit_index)
            writer.add_scalar('fit/optimizer_iterations',opt.nit,fit_index);writer.flush()
            update(stage='fitting',classifier_fits=fit_index,fit_tag=tag)
            exports[tag]=dict(mean=mean.tolist(),scale=scale.tolist(),theta=opt.x.tolist(),
                train_indices=train.tolist(),test_indices=test.tolist(),train_log_loss=unpenalized,
                positive_force_slope_enforced=True,optimizer_iterations=int(opt.nit))
            return score
        for model in models:
            train=np.asarray([i for i,r in enumerate(rows) if r['dataset']!='transient60'])
            test=np.asarray([i for i,r in enumerate(rows) if r['dataset']=='transient60'])
            pred=np.full(len(rows),np.nan);pred[test]=fit_predict(model,train,test,model+'/steady_to_transient')
            predictions[model+'/steady_to_transient']=pred
            pred=np.full(len(rows),np.nan)
            for family in names:
                train=np.asarray([i for i,r in enumerate(rows) if r['family']!=family])
                test=np.asarray([i for i,r in enumerate(rows) if r['family']==family])
                assert not {rows[i]['snapshot_sha256'] for i in train}&{rows[i]['snapshot_sha256'] for i in test}
                pred[test]=fit_predict(model,train,test,model+'/family_'+family)
            predictions[model+'/leave_family_out']=pred
        writer.close()
        def metrics(score,mask):
            ix=np.flatnonzero(mask&np.isfinite(score));actual=y[ix].astype(bool);pred=score[ix]>=.5
            positive=score[ix][actual];negative=score[ix][~actual]
            auc=float(np.mean([(a>b)+.5*(a==b) for a in positive for b in negative])) if len(positive) and len(negative) else None
            return dict(cases=len(ix),actual_contact=int(actual.sum()),false_safe=int((actual&~pred).sum()),
                conservative=int((~actual&pred).sum()),correct=int((actual==pred).sum()),AUC=auc,
                brier=float(np.mean((score[ix]-actual)**2)))
        summary={}
        for name,score in predictions.items():
            summary[name]=dict(all=metrics(score,np.ones(len(rows),bool)),
                              nonzero=metrics(score,np.asarray([r['force_N']!=0 for r in rows])))
        formula=np.asarray([float(abs(r['force_N'])>r['analytic_cap_N']) for r in rows])
        for subset in ['steady108','steady30','transient60']:
            summary['fixed_formula/'+subset]=metrics(formula,np.asarray([r['dataset']==subset for r in rows]))
        atomic_json(out/'model_exports.json',exports)
        np.savez_compressed(out/'predictions.npz',**predictions,actual_contact=y)
        atomic_json(out/'evaluation.json',dict(summary=summary,classifier_fits=fit_index,
            families=len(names),policy_updates=0,new_physics_steps=0,
            scope='Retrospective conditional capability scores; not calibrated real-world failure probabilities, a certified force boundary or policy improvement.'))
        update(phase='completed',stage='fixed_models_evaluated',classifier_fits=fit_index,cases=len(rows),families=len(names))
        print(json.dumps(summary,indent=2))
    except BaseException as exc:
        update(phase='error',stage='analysis_error',error_type=type(exc).__name__,error=str(exc));raise


if __name__=='__main__':main()
