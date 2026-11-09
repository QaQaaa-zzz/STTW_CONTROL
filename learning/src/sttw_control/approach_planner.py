"""Frozen endpoint-prediction / downstream-capability planner.

This first implementation predicts an endpoint, not a spatial dynamics rollout.
Scores are empirical estimates, not safety guarantees or calibrated probabilities.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def logistic_fit(x, y):
    x=np.column_stack((np.ones(len(x)),x)); y=np.asarray(y)
    penalty=np.ones(x.shape[1]);penalty[0]=0
    def loss(w):
        z=x@w
        return np.logaddexp(0,z).sum()-y@z+.5*np.sum(penalty*w*w),x.T@(expit(z)-y)+penalty*w
    fit=minimize(loss,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B')
    if not fit.success:raise RuntimeError(fit.message)
    return fit.x


def fit_model(rows, indices, entry_indices, candidates, identity):
    x=np.array([r['state'] for r in rows])[:,indices]
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1
    z=(x-mean)/scale;one=np.eye(len(candidates))[[r['action'] for r in rows]]
    design=np.column_stack((one,(one[:,:,None]*z[:,None,:]).reshape(len(rows),-1)))
    reached=np.array([r['entry'] is not None for r in rows])
    entry=np.array([r['entry'] for r in rows if r['entry'] is not None])[:,entry_indices]
    emean=entry.mean(0);escale=entry.std(0);escale[escale<1e-8]=1
    normalized=(entry-emean)/escale
    ridge=np.linalg.solve(design[reached].T@design[reached]+np.eye(design.shape[1]),design[reached].T@normalized)
    gate=logistic_fit(design,[r['prefix_qualified'] for r in rows])
    labels=[r['suffix_complete'] for r in rows if r['entry'] is not None]
    capability=logistic_fit(normalized,labels)
    return dict(schema='approach_endpoint_v1',indices=indices,entry_indices=entry_indices,candidates=candidates,
                mean=mean.tolist(),scale=scale.tolist(),entry_mean=emean.tolist(),entry_scale=escale.tolist(),
                endpoint_weights=ridge.tolist(),gate_weights=gate.tolist(),capability_weights=capability.tolist(),
                support_states=z.tolist(),maximum_support_rms=3.,keep_probability_tolerance=.02,
                identity=identity,training_rows=len(rows),capability_positive_count=int(sum(labels)),
                scope='Ridge endpoint prediction plus L2 logistic prefix gate and downstream suffix score; no spatial rollout or safety guarantee')


class ApproachPlanner:
    def __init__(self, model, *, identity):
        if model['schema']!='approach_endpoint_v1' or model['identity']!=identity:
            raise ValueError('Approach planner model/runtime identity mismatch')
        self.model=model

    def choose(self,state,stage,current):
        m=self.model;z=(np.asarray(state)[m['indices']]-m['mean'])/m['scale']
        support=float(np.sqrt(np.mean((np.asarray(m['support_states'])-z)**2,axis=1)).min())
        n=len(m['candidates']);one=np.eye(n)
        design=np.column_stack((one,(one[:,:,None]*z[None,None,:]).reshape(n,-1)))
        endpoint=design@np.asarray(m['endpoint_weights'])
        gate=expit(np.column_stack((np.ones(n),design))@m['gate_weights'])
        suffix=expit(np.column_stack((np.ones(n),endpoint))@m['capability_weights'])
        score=gate*suffix;selected=int(score.argmax())
        previous=next((i for i,c in enumerate(m['candidates']) if abs(c['speed_m_s']-current['speed_m_s'])<1e-6 and abs(c['alpha']-current['alpha'])<1e-6),None)
        if previous is not None and score[selected]-score[previous]<=m['keep_probability_tolerance']:selected=previous
        supported=bool(np.isfinite(z).all() and support<=m['maximum_support_rms'])
        return dict(status='reference_selected' if supported else 'unknown',candidate=m['candidates'][selected] if supported else current,
                    support_rms=support,estimates=[dict(candidate=c,score=float(score[i]),prefix_score=float(gate[i]),suffix_score=float(suffix[i]),
                    predicted_entry=(endpoint[i]*m['entry_scale']+m['entry_mean']).tolist()) for i,c in enumerate(m['candidates'])],
                    fallback=None if supported else 'retain_current_reference',scope=m['scope'])
