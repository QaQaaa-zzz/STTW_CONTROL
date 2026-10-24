"""Old-DEV-only joint baseline; freeze predictions separately from outcome access."""
import argparse,json,pickle,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,check_grad
from scipy.special import expit
from sttw_control.preference_best import atomic_json,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--evaluate',action='store_true');a=p.parse_args()
    root=a.run.resolve();out=root/'joint_baseline_supplement'
    if a.evaluate:
        from audit_operating_point_margin import auc
        frozen=json.loads((out/'frozen_predictions.json').read_text())
        primary=json.loads((root/'analysis/analysis.json').read_text())
        rows=[]
        for r in primary['scored_nonzero_cases']:
            prediction=next(x for x in frozen['predictions'] if x['side']==r['side'] and x['state_id']==r['state_id'] and x['force_N']==r['force_N'])
            rows.append(dict(**r,joint_score=prediction['score']))
        assert len(rows)==32
        def stats(rs):
            return dict(n=len(rs),safe=sum(x['safe'] for x in rs),formula_auc=auc([x['safe'] for x in rs],[x['formula_margin_N'] for x in rs]),joint_auc=auc([x['safe'] for x in rs],[x['joint_score'] for x in rs]))
        atomic_json(out/'evaluation.json',dict(overall=stats(rows),strata={key:[dict(value=v,**stats([x for x in rows if x[key]==v])) for v in sorted({x[key] for x in rows})] for key in ['side','speed','alpha','force_N']},cases=rows,predictions_sha256=digest(out/'frozen_predictions.json'),primary_sha256=digest(root/'analysis/analysis.json'),evaluated_unix=time.time()))
        return
    out.mkdir(exist_ok=False)
    old=Path('runs/operating_point_formula_audit_v1/analysis.json').resolve()
    oldrows=json.loads(old.read_text())['cases']
    from sttw_control.path_command_policy import actor_feature_spec
    config=Path('runs/reserve_feature_expansion_v1/physics_config.json').resolve()
    spec=actor_feature_spec(json.loads(config.read_text()))
    receipts={str(x):digest(x) for x in [old,config,Path(__file__).resolve()]}
    cache={}
    def features(file,force):
        file=Path(file)
        if str(file) not in cache:
            with file.open('rb') as f:s=pickle.load(f)
            cache[str(file)]=np.asarray(s.history.frames)[-1]
            receipts[str(file)]=digest(file)
        last=cache[str(file)];names=spec.frame_names
        v=last[names.index('rear_speed_proxy_m_s')]
        return np.array([v,v*v]+[np.sign(force)*last[names.index(k)] for k in ['roll_rad','roll_rate_rad_s','steer_rad','eso_equilibrium_shift_rad']],dtype=float)
    # The old audit records snapshot identity separately by input hashes.
    matched=json.loads(Path('runs/zero_middle_matched_analysis_v1/analysis.json').read_text())['cases']
    X=[];y=[]
    for r in oldrows:
        m=next(x for x in matched if x['route']==r['route'] and x['requested_speed']==r['speed'] and x['fraction']==r['fraction'] and x['model']==r['model'] and x['force_N']==r['force_N'])
        X.append(features(m['snapshot'],r['force_N']));y.append(r['safe'])
    X=np.asarray(X);y=np.asarray(y,float);mean=X.mean(0);scale=X.std(0);scale=np.where(scale>1e-10,scale,1.)
    Z=np.column_stack([np.ones(len(X)),(X-mean)/scale])
    def objective(w):
        logits=Z@w
        return np.sum(np.logaddexp(0,logits)-y*logits)+.5*np.dot(w[1:],w[1:])
    def gradient(w):return Z.T@(expit(Z@w)-y)+np.r_[0.,w[1:]]
    grad_error=check_grad(objective,gradient,np.zeros(7))
    if grad_error>1e-4:raise ValueError('Gradient check failed')
    fit=minimize(objective,np.zeros(7),jac=gradient,method='BFGS',options={'gtol':1e-8,'maxiter':1000})
    if np.max(np.abs(gradient(fit.x)))>1e-6:raise ValueError('Optimizer not converged')
    predictions=[]
    for side in ['left','right']:
        file=root/side/'prefix_forecasts.json';forecasts=json.loads(file.read_text());receipts[str(file)]=digest(file)
        for r in forecasts['cases']:
            for force in [-80.,80.]:
                x=features(r['snapshot'],force);score=float(np.r_[1.,(x-mean)/scale]@fit.x)
                predictions.append(dict(side=side,state_id=r['state_id'],force_N=force,features=x.tolist(),score=score))
    atomic_json(out/'frozen_predictions.json',dict(saved_unix=time.time(),status='post-start supplemental; frozen before root inspected prospective labels; automated primary analyzer already read them',features=['speed_proxy','speed_proxy_squared','signed_roll','signed_roll_rate','signed_steer','signed_ESO_shift'],old_dev_rows=48,mean=mean.tolist(),scale=scale.tolist(),weights=fit.x.tolist(),gradient_check_error=grad_error,max_gradient=float(np.max(np.abs(gradient(fit.x)))),predictions=predictions,input_sha256=receipts,new_physics_ticks=0,new_PPO_updates=0,statistical_model_fits=1,regularization=1.))


if __name__=='__main__':main()
