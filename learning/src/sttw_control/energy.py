"""Sampled actuator mechanical work; never a battery-energy estimator."""
import numpy as np


def mechanical_work(trace):
    keys=('time','actuator_force','actuator_velocity','qpos')
    if any(k not in trace for k in keys):
        return {'available':False,'reason':'missing actuator or position samples'}
    t,f,v,q=(np.asarray(trace[k],dtype=float) for k in keys)
    if (t.ndim!=1 or len(t)<2 or f.ndim!=2 or f.shape!=v.shape or len(f)!=len(t)
        or q.ndim!=2 or q.shape[0]!=len(t) or q.shape[1]<2
        or not all(np.isfinite(x).all() for x in (t,f,v,q)) or np.any(np.diff(t)<=0)):
        return {'available':False,'reason':'invalid or insufficient aligned samples'}
    power=f*v
    if not np.isfinite(power).all():return {'available':False,'reason':'nonfinite mechanical power'}
    # Clip each actuator before integration, preventing cross-channel cancellation.
    positive=np.trapezoid(np.maximum(power,0.),t,axis=0)
    negative=np.trapezoid(np.maximum(-power,0.),t,axis=0)
    distance=float(np.linalg.norm(np.diff(q[:,:2],axis=0),axis=1).sum())
    return {'available':True,'positive_work_j':float(positive.sum()),
            'negative_work_magnitude_j':float(negative.sum()),
            'positive_work_per_actuator_j':positive.tolist(),
            'negative_work_per_actuator_j':negative.tolist(),
            'positive_work_per_planar_meter_j_m':float(positive.sum()/distance) if distance>1e-9 else None,
            'planar_distance_m':distance,'window_start_seconds':float(t[0]),'window_end_seconds':float(t[-1]),
            'scope':'whole observed episode; sampled trapezoidal mechanical work, not battery energy or recovered braking energy'}
