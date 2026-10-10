"""Read-only phase/platform diagnostics; never modifies legacy selection/rewards."""
import numpy as np
from .lower_tracking_audit import intervals
DT=.005

def kappa(delta):return np.cos(25*np.pi/180)*np.tan(delta)/.408

def small_window(d,case,cfg):
    t=np.asarray(d['time']);mask=np.zeros(len(t),bool)
    if not case.startswith('post_return_small'):return mask
    command=np.asarray(d['limited_command']);rates=np.asarray(d['raw_rates'])
    hits=np.flatnonzero((t>=cfg['post_small_target_start_s']-1e-6)&(abs(abs(command[:,1])-cfg['post_small_target'])<1e-6)&(abs(rates[:,1])<.02))
    if len(hits):mask=t>=t[hits[0]]+cfg['settle_s']-1e-6
    return mask

def phase_score(d,small):
    err=np.column_stack([d['actual_forward_speed'],d['actual_delta']]).astype(float)-d['limited_command']
    tol=np.tile([.05,.02],(len(err),1));tol[small,1]=.01
    return float(np.mean((err/tol)**2)+2*np.mean(np.asarray(d['peak_roll'])>.302))

def channel_stats(error,t,mask,tolerance):
    ix=np.flatnonzero(mask)
    if not len(ix):return None
    e=np.asarray(error)[ix];a=abs(e);maximum=int(ix[np.argmax(a)])
    exceed=intervals(mask&(abs(error)>tolerance),np.asarray(t),DT)
    return dict(samples=int(len(ix)),duration_s=len(ix)*DT,bias=float(np.mean(e)),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(a)),p95_abs=float(np.quantile(a,.95)),max_abs=float(max(a)),max_time_s=float(t[maximum]),positive_fraction=float(np.mean(e>0)),negative_fraction=float(np.mean(e<0)),zero_fraction=float(np.mean(e==0)),longest_exceed_s=max((r['duration_s'] for r in exceed),default=0.),exceed_intervals=exceed)

def platform_statistics(d,small):
    t=np.asarray(d['time'],float);cmd=np.asarray(d['limited_command'],float);rates=np.asarray(d['raw_rates']);n=len(t)
    err=np.column_stack([d['actual_forward_speed'],d['actual_delta']]).astype(float)-cmd
    quiet=np.all(abs(rates)<1e-5,axis=1);rows=[];i=0
    while i<n:
        if not quiet[i]:i+=1;continue
        j=i+1
        while j<n and quiet[j] and np.allclose(cmd[j],cmd[i],atol=1e-6,rtol=0):j+=1
        full=np.zeros(n,bool);full[i:j]=True;settled=full&(t>=t[i]+.5-1e-6)
        tau=.01 if np.any(small&settled) else .02
        stats=lambda mask:{name:channel_stats(err[:,c],t,mask,tol) for name,c,tol in [('speed',0,.05),('steer',1,np.where(small,.01,.02))]}
        f,s=stats(full),stats(settled)
        worst=max(s['speed']['rmse']/.05,s['steer']['rmse']/tau) if settled.sum()*DT>=1.-1e-6 else None
        row=dict(start_s=float(t[i]),end_exclusive_s=float(t[j-1]+DT),command=cmd[i].tolist(),steer_tolerance=tau,full=f,settled=s,worst_normalized_rmse=worst,worst_status='computed' if worst is not None else 'N/A: fewer than1s settled samples')
        if abs(cmd[i,1])<1e-6 and 'yaw_unwrapped' in d:
            yaw=np.asarray(d['yaw_unwrapped'],float);r=np.r_[np.nan,np.diff(yaw)/DT];valid=settled&np.isfinite(r)
            row['zero_steer_yaw_rate_bias']=float(np.mean(r[valid])) if valid.any() else None
        rows.append(row);i=j
    return rows

def yaw_decomposition(d,*,initial_yaw=None,initial_error=None):
    required=['limited_command','actual_forward_speed','actual_delta','yaw_unwrapped','reference_yaw_unwrapped','e_psi_unwrapped']
    missing=[k for k in required if k not in d]
    if missing or initial_yaw is None or initial_error is None:
        return dict(status='unavailable',missing=missing+(['initial_yaw/initial_error'] if initial_yaw is None or initial_error is None else [])),{}
    cmd=np.asarray(d['limited_command'],float);v=np.asarray(d['actual_forward_speed'],float);delta=np.asarray(d['actual_delta'],float)
    yaw=np.asarray(d['yaw_unwrapped'],float);actual_rate=np.diff(np.r_[initial_yaw,yaw])/DT
    A=kappa(cmd[:,1])*(cmd[:,0]-v);B=v*(kappa(cmd[:,1])-kappa(delta));C=v*kappa(delta)-actual_rate
    pieces=np.cumsum(np.column_stack([A,B,C])*DT,axis=0);reconstructed=initial_error+pieces.sum(axis=1)
    error=np.asarray(d['e_psi_unwrapped'],float)
    return dict(status='computed',initial_error=float(initial_error),contribution_end=pieces[-1].tolist(),measured_end=float(error[-1]),reconstructed_end=float(reconstructed[-1]),closure_abs_error=float(abs(reconstructed[-1]-error[-1])),closure_max_abs_error=float(np.max(abs(reconstructed-error))),C_interpretation='kinematic/sampling/contact residual; not a measured tire slip rate'),dict(A=A,B=B,C=C,world_yaw_rate=actual_rate,cumulative=pieces,reconstructed_error=reconstructed,measured_error=error)
