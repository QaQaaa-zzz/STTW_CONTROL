"""Prespecified gates use actual transitions, not reference targets or rewards."""
import numpy as np

def rmse(x):return float(np.sqrt(np.mean(np.square(x)))) if len(x) else None

def held(mask,n=100):
    mask=np.asarray(mask,bool)
    if len(mask)<n:return None
    hits=np.flatnonzero(np.convolve(mask.astype(int),np.ones(n,int),'valid')==n)
    return int(hits[0]+n-1) if len(hits) else None

def baseline_metrics(x,rows):
    rows=np.asarray(rows);time=x['time'];segments=[]
    for i,row in enumerate(rows):
        end=rows[i+1,0] if i+1<len(rows) else 16.
        # Poststep measurement vs command at interval start; full last .5s.
        start=time-.005;mask=(start>=end-.5-1e-9)&(start<end-1e-9)
        ev=rmse((x['actual_forward_speed']-x['limited_command'][:,0])[mask])
        ed=rmse((x['actual_delta']-x['limited_command'][:,1])[mask])
        segments.append(dict(end_s=float(end),samples=int(mask.sum()),speed_rmse=ev,steer_rmse=ed,
            passed=bool(mask.sum()==100 and ev<=.12 and ed<=.04)))
    peak=float(np.max(x['peak_roll']));failure=bool(np.any(x['physical_failure']))
    return dict(passed=all(s['passed'] for s in segments) and peak<=.30 and not failure and len(time)==3200,
        segments=segments,peak_roll=peak,physical_failure=failure,actual_ticks=len(time))

def recovery_metrics(x):
    time=x['time'];start=time-.005;c=x['limited_command']
    onset=np.flatnonzero((start>=5-1e-9)&(np.abs(c[:,1])<=.005))
    if not len(onset):return dict(passed=False,start_s=None,completion_s=None)
    i=onset[0];t0=float(start[i]);window=(start>=t0-1e-9)&(time<=t0+8+1e-9)
    good=(np.abs(x['e_psi_unwrapped'])<=.05)&(np.abs(x['actual_forward_speed']-c[:,0])<=.10)&(np.abs(x['actual_delta']-c[:,1])<=.03)&~x['physical_failure']&window
    end=held(good)
    return dict(passed=end is not None,start_s=t0,completion_s=float(time[end]) if end is not None else None,
        final_heading_error=float(x['e_psi_unwrapped'][-1]))

def tracking_metrics(x,window=None):
    start=x['time']-.005;m=np.ones(len(start),bool) if window is None else (start>=window[0]-1e-9)&(start<window[1]-1e-9)
    ev=x['actual_forward_speed']-x['limited_command'][:,0];ed=x['actual_delta']-x['limited_command'][:,1]
    return dict(speed_rmse=rmse(ev[m]),steer_rmse=rmse(ed[m]),
        overspeed_peak=float(np.maximum(ev,0).max()),overspeed_seconds=float(np.sum(ev>0)*.005),overspeed_integral=float(np.maximum(ev,0).sum()*.005),
        underspeed_peak=float(np.maximum(-ev,0).max()),underspeed_integral=float(np.maximum(-ev,0).sum()*.005),
        peak_roll=float(np.max(x['peak_roll'])),peak_roll_rate=float(np.max(x['peak_roll_rate'])),
        physical_failure=bool(np.any(x['physical_failure'])),fallback=bool(np.any(x['fallback'])),
        last_time_s=float(x['time'][-1]),lateral_peak=float(np.max(np.abs(x['lateral']))),
        final_along=float(x['along'][-1]),recovery=recovery_metrics(x))

def core_gate(b,g0,g1,raw_conflict_seen):
    x0=tracking_metrics(g0,(3,5));x1=tracking_metrics(g1,(3,5));reasons=[]
    if np.max(b['peak_roll'])<=.30 and not raw_conflict_seen:
        return dict(passed=False,classification=['not_a_conflict'],G0=x0,G1=x1)
    for name,x in [('G0',x0),('G1',x1)]:
        if x['physical_failure'] or x['fallback'] or x['peak_roll']>.302 or x['last_time_s']<16:
            reasons.append(name+':physical_or_operation_or_fallback_failure')
        if not x['recovery']['passed']:reasons.append(name+':recovery_blocked')
    ed0,ed1=x0['steer_rmse'],x1['steer_rmse'];ev0,ev1=x0['speed_rmse'],x1['speed_rmse']
    if any(v is None for v in [ed0,ed1,ev0,ev1]) or not (ed0<=.8*ed1 and ed1-ed0>=.01 and ev1<=.8*ev0 and ev0-ev1>=.05 and ed0<=.05 and ev1<=.15):
        reasons.append('priority_not_separated')
    t0=g0['time']-.005;pre=(t0>=1.5-1e-9)&(t0<2-1e-9);conf=(t0>=3-1e-9)&(t0<5-1e-9)
    vpre=float(np.mean(g0['actual_forward_speed'][pre])) if pre.any() else None
    brake=held((g0['actual_forward_speed']<=vpre-.15)&conf,40) if vpre is not None else None
    if brake is None:reasons.append('no_active_speed_drop')
    t1=g1['time']-.005;m=(t1>=3-1e-9)&(t1<5-1e-9)
    reduction=float(np.mean(np.abs(g1['limited_command'][m,1])-np.abs(g1['actual_delta'][m]))) if m.any() else None
    if reduction is None or reduction<.03:reasons.append('no_steer_reduction')
    return dict(passed=not reasons,classification=reasons,G0=x0,G1=x1,pre_speed_G0=vpre,active_brake=brake is not None,steer_reduction_G1=reduction)

def governor_nominal_metrics(x):
    mask=x['raw_feasible']&(x['mode']!=2)
    zero=bool(np.all(x['applied_residual'][mask]==0))
    return dict(passed=zero and len(x['time'])==3200 and not x['physical_failure'].any() and not x['fallback'].any() and float(np.max(x['peak_roll']))<=.30,
        raw_feasible_nonrecover_zero_residual=zero,zero_check_samples=int(mask.sum()),
        operating_bound_passed=float(np.max(x['peak_roll']))<=.30)

def random_gate(x):
    complete=len(x['time'])==3200
    good=(x['time']>=10)&(np.abs(x['e_psi_unwrapped'])<=.05)&(np.abs(x['actual_forward_speed']-x['limited_command'][:,0])<=.10)&(np.abs(x['actual_delta']-x['limited_command'][:,1])<=.03)
    final_hold=bool(complete and np.all(good[-100:]))
    return dict(passed=complete and final_hold and not x['physical_failure'].any() and not x['fallback'].any() and np.max(x['peak_roll'])<=.302,
        final_half_second_hold=final_hold,final_heading_error=float(x['e_psi_unwrapped'][-1]))
