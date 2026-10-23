"""Small fixed-kernel acquisition model for empirical recovery opportunities.

Scores rank simulation queries, never certify safety or command the robot.
"""
import math
import numpy as np

BASE_FIELDS = ('requested_speed_m_s', 'alpha', 'follower_curvature_inv_m',
               'roll_rad', 'roll_rate_rad_s', 'steer_rad', 'cross_track_m',
               'controller_disturbance', 'remaining_s')

def query_features(features, force, duration, *, hybrid=False,
                   gamma=.815560907125473, effort_budget=.030839085298575167):
    values = [features[k] for k in BASE_FIELDS] + [force, duration]
    if hybrid:
        lam = features['lambda_u']; b = features['b_u']; z = features['z_u']
        if lam <= 0 or effort_budget <= 0 or b == 0:
            raise ValueError('invalid analytic model')
        shift = gamma * (-math.expm1(-lam*duration)) / lam * force
        W = b*b * (-math.expm1(-2*lam*min(1., features['remaining_s']))) / (2*lam)
        values += [z, shift, b, abs(z+shift)/math.sqrt(effort_budget*W)]
    x = np.asarray(values, float)
    if duration <= 0 or not np.isfinite(x).all():
        raise ValueError('invalid query')
    return x

def fit_memory(X, labels, *, k=8):
    X = np.asarray(X, float); y = np.asarray(labels, float)
    if X.ndim != 2 or y.shape != (len(X),) or not len(X):
        raise ValueError('invalid training dimensions')
    if not np.isfinite(X).all() or not np.isin(y, [0., 1.]).all():
        raise ValueError('nonfinite inputs or nonbinary labels')
    if k < 1: raise ValueError('k must be positive')
    mean = X.mean(0); scale = X.std(0); scale = np.where(scale > 1e-8, scale, 1.)
    return dict(mean=mean.tolist(), scale=scale.tolist(),
                X=((X-mean)/scale).tolist(), labels=y.tolist(), k=int(k))

def acquisition_score(model, X):
    X = np.asarray(X, float)
    bank = np.asarray(model['X']); mean = np.asarray(model['mean']); scale = np.asarray(model['scale'])
    if X.ndim != 2 or X.shape[1] != bank.shape[1] or not np.isfinite(X).all():
        raise ValueError('invalid query dimensions or values')
    distances = np.mean((((X-mean)/scale)[:, None, :]-bank[None, :, :])**2, axis=-1)
    idx = np.argsort(distances, axis=1, kind='stable')[:, :min(model['k'], len(bank))]
    d = np.take_along_axis(distances, idx, axis=1)
    # Common per-query shift prevents underflow, without changing normalized weights.
    weights = np.exp(-.5*(d-d.min(axis=1, keepdims=True)))
    labels = np.asarray(model['labels'])[idx]
    return (weights*labels).sum(1)/weights.sum(1)
