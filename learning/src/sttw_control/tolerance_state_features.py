"""Cheap interpretable model features, NOT calibrated force tolerances."""
import math
import numpy as np

def unstable_features(*,speed_proxy,steer,roll,roll_rate,disturbance,controller):
    """Frozen-speed, constant-ESO model coordinate from controller._system.

    Use the controller's rear-wheel speed proxy, not silently true body speed.
    No gamma or effort budget is inferred; output has no Newton margin.
    """
    c=controller
    vals=[speed_proxy,steer,roll,roll_rate,disturbance]+[c[k] for k in ['mass','cg_height','cg_forward','wheelbase','trail','gravity','caster','minimum_speed']]
    if not np.isfinite(vals).all():raise ValueError('nonfinite model feature input')
    if min(c[k] for k in ['mass','cg_height','wheelbase','gravity','minimum_speed'])<=0:raise ValueError('invalid physical model parameters')
    v=max(c['minimum_speed'],float(speed_proxy));m=c['mass'];h=c['cg_height'];a=c['cg_forward'];b=c['wheelbase'];g=c['gravity'];inertia=m*h*h+.043929*m/5.434
    m1=-m*a*h*v*math.cos(c['caster'])/b
    raw_m2=-(m*v*v*h-m*a*c['trail']*g)*math.cos(c['caster'])/b
    m2=(-1. if raw_m2<0 else 1.)*max(abs(raw_m2),.1)
    a1=m1/inertia;a2=m2/inertia;a4=m*g*h/inertia;lam=math.sqrt(a4);bu=a1+a2/lam
    return dict(a1=a1,a2=a2,a4=a4,lambda_u=lam,b_u=bu,static_input_gain_abs=abs(bu),z_u=(a2/lam)*steer+lam*roll+roll_rate+disturbance/lam,model_speed_proxy=v)

FRAME_NAMES=('roll_rad','roll_rate_rad_s','steer_rad','steer_rate_rad_s','true_forward_speed_m_s','yaw_rate_rad_s','rear_speed_proxy_m_s','front_speed_proxy_m_s','raw_speed_ref_m_s','raw_steer_ref_rad','raw_speed_rate_m_s2','raw_steer_rate_rad_s','previous_governed_speed_m_s','previous_governed_steer_rad','previous_final_steer_command','previous_final_rear_command','previous_bounded_steer_residual','previous_bounded_rear_residual','eso_equilibrium_shift_rad','raw_roll_reference_rad')

def snapshot_features(snapshot,*,speed_request,onset_s,controller,horizon_s=20.):
    frame=np.asarray(snapshot.history.frames[-1],float)
    if frame.shape!=(20,) or not np.isfinite(frame).all():raise ValueError('finite raw20 frame required')
    if not bool(np.asarray(snapshot.history.mask[-1])):raise ValueError('newest history frame is not valid')
    if not np.isfinite([speed_request,onset_s,horizon_s]).all() or not 0<=onset_s<horizon_s:raise ValueError('invalid task deadline')
    if abs(float(snapshot.tick)*controller['dt']-onset_s)>1e-5:raise ValueError('snapshot clock does not match declared onset')
    out=dict(zip(FRAME_NAMES,frame.tolist()))
    out.update(alpha=float(snapshot.alpha),requested_speed_m_s=float(speed_request),remaining_s=horizon_s-onset_s,cross_track_m=float(snapshot.projection.cross_track),path_progress_m=float(snapshot.projection.progress),follower_curvature_inv_m=float(snapshot.follower.kappa),lookahead_m=float(snapshot.follower.lookahead),controller_disturbance=float(snapshot.physical.controller.disturbance))
    for i,name in enumerate(['speed','steer']):out[f'offset_{name}']=float(snapshot.offsets[i]);out[f'filtered_{name}']=float(snapshot.filtered[i])
    out.update(unstable_features(speed_proxy=frame[6],steer=frame[2],roll=frame[0],roll_rate=frame[1],disturbance=out['controller_disturbance'],controller=controller))
    if not all(math.isfinite(x) for x in out.values()):raise ValueError('nonfinite snapshot feature')
    return out
