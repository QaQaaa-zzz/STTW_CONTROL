"""Small conditional force-response models, not a safety certificate.

State features already encode a FIXED query direction and reference contract.
Amplitude is nonnegative; force crossings are statistical score contours only.
No controller/Actor is changed or called by this CPU fitting module.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit


def logistic_objective(theta,design,target,*,l2):
    z=design@theta;residual=expit(z)-target
    loss=np.sum(np.logaddexp(0,z)-target*z)+.5*l2*np.square(theta[:-1]).sum()
    gradient=design.T@residual
    gradient[:-1]+=l2*theta[:-1]
    return float(loss),gradient


def _state(x,scale):
    x=np.asarray(x,float);scale=np.asarray(scale,float)
    if x.ndim!=2 or scale.shape!=(x.shape[1],) or not np.isfinite(x).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('finite matrix and positive per-feature physical scales required')
    return x/scale


def _kernels(x,centers,width):
    distance=np.sum((x[:,None,:]-centers[None,:,:])**2,axis=-1)
    return np.exp(-distance/(2*width*width)),np.sqrt(np.min(distance,axis=1))


def _design(model,x,amplitude):
    x=_state(x,model['state_scale']);a=np.asarray(amplitude,float)
    if a.shape!=(len(x),) or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError('one finite nonnegative amplitude per fixed-direction state required')
    a=a/model['amplitude_scale_N']
    if model['kind']=='linear':
        return np.c_[x,a,np.ones(len(x))],np.full(len(x),np.nan),np.full(len(x),np.nan)
    kernel,distance=_kernels(x,np.asarray(model['centers']),model['kernel_width'])
    return np.c_[kernel,a[:,None]*kernel,a,np.ones(len(x))],distance,np.max(kernel,axis=1)


def fit_conditional_model(x,amplitude,target,*,kind,state_scale,max_centers=24,
                          kernel_width=1.,amplitude_scale_N=64.,l2=1.):
    normalized=_state(x,state_scale);target=np.asarray(target,float)
    if kind not in ('linear','rbf') or target.shape!=(len(normalized),) or not np.isin(target,[0.,1.]).all() or len(np.unique(target))!=2:
        raise ValueError('known model kind and both binary outcome classes required')
    if not np.isfinite([kernel_width,amplitude_scale_N,l2]).all() or min(kernel_width,amplitude_scale_N,l2)<=0 or type(max_centers)!=int or max_centers<1:
        raise ValueError('positive fixed model configuration required')
    model=dict(schema='conditional_force_score_v1',kind=kind,state_scale=list(state_scale),
        amplitude_scale_N=amplitude_scale_N,kernel_width=kernel_width,centers=[])
    if kind=='rbf':
        unique=np.unique(normalized,axis=0)
        chosen=[0];nearest=np.sum((unique-unique[0])**2,axis=1)
        while len(chosen)<min(max_centers,len(unique)):
            nxt=int(np.argmax(nearest));chosen.append(nxt)
            nearest=np.minimum(nearest,np.sum((unique-unique[nxt])**2,axis=1))
        model['centers']=unique[chosen].tolist()
    design,_,_=_design(model,x,amplitude)
    constrained=[design.shape[1]-2] if kind=='linear' else list(range(len(model['centers']),design.shape[1]-1))
    bounds=[(0.,None) if i in constrained else (None,None) for i in range(design.shape[1])]
    fit=minimize(lambda theta:logistic_objective(theta,design,target,l2=l2),
        np.zeros(design.shape[1]),jac=True,method='L-BFGS-B',bounds=bounds,
        options=dict(maxiter=1000,ftol=1e-10,gtol=1e-7))
    if not fit.success:raise RuntimeError('conditional fit did not converge: '+str(fit.message))
    z=design@fit.x
    model.update(theta=fit.x.tolist(),constrained_indices=constrained,parameter_count=len(fit.x),
        train_log_loss=float(np.mean(np.logaddexp(0,z)-target*z)),optimizer_iterations=int(fit.nit),
        maximum_training_amplitude_N=float(np.max(amplitude)),l2=l2,
        scope='Conditional score for fixed query direction/reference; force crossing is not physical certification.')
    return model


def predict_conditional(model,x,amplitude):
    theta=np.asarray(model['theta'],float)
    if not np.isfinite(theta).all() or np.any(theta[model['constrained_indices']]<0):
        raise ValueError('finite coefficients and nonnegative force slopes required')
    design,distance,kernel=_design(model,x,amplitude)
    return dict(probability=expit(design@theta),logit=design@theta,
                nearest_center_distance=distance,maximum_kernel=kernel)


def force_crossing(model,x,*,probability=.5):
    """Fixed-direction p contour. NaN slope crossing is unidentified, not infinite safety."""
    if not 0<probability<1:raise ValueError('probability must lie in (0,1)')
    n=len(x);zero=predict_conditional(model,x,np.zeros(n))['logit']
    at_scale=predict_conditional(model,x,np.full(n,model['amplitude_scale_N']))['logit']
    slope=(at_scale-zero)/model['amplitude_scale_N'];positive=slope>1e-12
    force=np.full(n,np.nan)
    force[positive]=np.maximum(0.,(logit(probability)-zero[positive])/slope[positive])
    return dict(force_N=force,positive_slope=positive,already_above_at_zero=zero>=logit(probability),
                within_training_amplitude=positive&(force<=model['maximum_training_amplitude_N']))
