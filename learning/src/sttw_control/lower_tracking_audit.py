#!/usr/bin/env python3
"""Audit saved STTW compact NPZ. No physics, model inference, or reanchoring.

Input keys: case__method__field, optional case__shared__field for raw/time.
Aligned same-index comparison: each command is held over the interval whose
post-physics actual state is recorded in the same row. Never time-shift scores.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

REQUIRED = ('limited_command','governed','actual_forward_speed','actual_delta','time')

def intervals(mask: np.ndarray, time: np.ndarray, dt: float) -> list[dict]:
    b=np.asarray(mask,dtype=bool)
    edges=np.flatnonzero(np.diff(np.r_[False,b,False].astype(np.int8)))
    return [{'start_s':float(time[s]),'end_exclusive_s':float(time[e-1]+dt),
             'duration_s':float((e-s)*dt),'start_index':int(s),'end_index_exclusive':int(e)}
            for s,e in zip(edges[::2],edges[1::2])]

def held(mask: np.ndarray, dt: float, seconds: float) -> np.ndarray:
    out=np.zeros(len(mask),bool); count=0
    for i,m in enumerate(mask):
        count=count+1 if m else 0
        out[i]=count*dt>=seconds-1e-10
    return out

def stats(x: np.ndarray, mask: np.ndarray, time: np.ndarray, dt: float, tol: float) -> dict:
    indices=np.flatnonzero(mask)
    if len(indices)==0: return {'count':0,'status':'no_samples'}
    a=x[indices]; j=int(indices[np.argmax(np.abs(a))])
    excursions=intervals(mask & (np.abs(x)>tol),time,dt)
    return {'count':len(indices),'bias':float(a.mean()),'rmse':float(np.sqrt(np.mean(a*a))),
            'mae':float(np.mean(np.abs(a))),'p95_abs':float(np.quantile(np.abs(a),.95)),
            'max_abs':float(abs(x[j])),'max_abs_interval_start_s':float(time[j]),
            'tolerance':tol,'fraction_exceeding':float(np.mean(np.abs(a)>tol)),
            'longest_exceeding_s':max((i['duration_s'] for i in excursions),default=0.),
            'all_exceeding_intervals':excursions}

def run(source: Path, destination: Path) -> dict:
    destination.mkdir(parents=True,exist_ok=True)
    with np.load(source,allow_pickle=False) as z:
        groups={}
        for key in z.files:
            parts=key.split('__',2)
            if len(parts)!=3: continue
            case,method,field=parts
            groups.setdefault((case,method),{})[field]=np.array(z[key])
    output={'source':str(source),'new_physics_steps':0,'alignment':'same interval; no score lag compensation',
            'scope':'recorded states only; not proof of tracking for every possible command/state','cases':{}}
    arrays={}; rows=[]
    for (case,method),own in sorted(groups.items()):
        if method in ('shared','raw','reference') or 'actual_delta' not in own: continue
        common=next((groups[(case,m)] for m in ('shared','raw','reference') if (case,m) in groups),{})
        d={**common,**own}
        # B0 can supply shared raw/time, but never another method's governed or actual state.
        for field in ('time','limited_command'):
            if field not in d and (case,'B0') in groups and field in groups[(case,'B0')]:
                d[field]=groups[(case,'B0')][field]
        missing=[k for k in REQUIRED if k not in d]
        if missing:
            output['cases'][f'{case}/{method}']={'status':'missing_fields','missing':missing}; continue
        t=np.asarray(d['time'],float).reshape(-1); n=len(t)
        if n<2 or np.any(np.diff(t)<=0): raise ValueError(f'{case}/{method}: invalid time')
        dt=float(np.median(np.diff(t)))
        if not np.allclose(np.diff(t),dt,atol=2e-6,rtol=2e-3):
            raise ValueError(f'{case}/{method}: nonuniform trace; handle segments explicitly')
        if abs(dt-.005)>2e-6:raise ValueError('expected 5ms saved intervals')
        dt=.005
        raw=np.asarray(d['limited_command'],float); gov=np.asarray(d['governed'],float)
        actual=np.column_stack((np.asarray(d['actual_forward_speed']).reshape(-1),np.asarray(d['actual_delta']).reshape(-1))).astype(float)
        if raw.shape!=(n,2) or gov.shape!=(n,2) or actual.shape!=(n,2):
            raise ValueError(f'{case}/{method}: incompatible arrays')
        if not all(np.isfinite(v).all() for v in (t,raw,gov,actual)): raise ValueError('nonfinite trace')
        upper=gov-raw; lower=actual-gov; total=actual-raw
        error=float(np.max(np.abs(total-upper-lower)))
        if error>1e-8: raise AssertionError('error decomposition failed')
        raw_rate=np.vstack((np.zeros((1,2)),np.diff(raw,axis=0)/dt))
        gov_rate=np.vstack((np.zeros((1,2)),np.diff(gov,axis=0)/dt))
        quiet=held((np.abs(gov_rate[:,0])<=.1)&(np.abs(gov_rate[:,1])<=.02),dt,.3)
        valid=np.ones(n,bool)
        failure=np.asarray(d.get('physical_failure',np.zeros(n,bool))).reshape(-1).astype(bool)
        if len(failure)!=n: raise ValueError('failure length mismatch')
        if failure.any(): valid[np.flatnonzero(failure)[0]+1:]=False
        masks={'all':valid,'command_1_6':valid&(t>=1)&(t<6),
               'stable_2_4':valid&(t>=2)&(t<4),'governed_quiet_0p3s':valid&quiet,
               'governed_transient':valid&~quiet}
        if 'g' in d:
            has_turn=np.maximum.accumulate(np.abs(raw[:,1])>.01)
            masks['post_turn_recovery']=valid&has_turn&(np.asarray(d['g']).reshape(-1)>.5)
        result={'status':'computed','count':n,'dt':dt,'observed_end_exclusive_s':float(t[-1]+dt),
                'physical_failure':bool(failure.any()),'decomposition_max_error':error,'windows':{}}
        for name,mask in masks.items():
            values={}
            for j,channel,tol in ((0,'speed_m_s',.1),(1,'steer_rad',.04)):
                values[channel]={label:stats(x[:,j],mask,t,dt,tol)
                                 for label,x in (('upper_correction',upper),('lower_tracking_error',lower),('task_error',total))}
                v=values[channel]['lower_tracking_error']
                if v['count']:
                    rows.append({'case':case,'method':method,'window':name,'channel':channel,
                                 **{k:v[k] for k in ('count','bias','rmse','p95_abs','max_abs','max_abs_interval_start_s','fraction_exceeding','longest_exceeding_s')}})
            result['windows'][name]=values
        if 'wheel_speed_proxy' in d and 'final_command' in d:
            wheel=np.asarray(d['wheel_speed_proxy']).reshape(-1); motor=np.asarray(d['final_command'])[:,1]*.1
            result['motor_proxy_note']='0.1*rear shaft rate is a proxy; proxy minus chassis speed is not a validated contact slip ratio.'
            result['motor_proxy_error']=stats(wheel-motor,valid,t,dt,.1)
            result['proxy_minus_chassis']=stats(wheel-actual[:,0],valid,t,dt,.1)
        if 'phi' in d:
            phi=np.abs(np.asarray(d['phi']).reshape(-1))
            peak=np.maximum(phi,np.asarray(d.get('peak_roll',phi)).reshape(-1))
            result['safety']={'observed_peak_abs_roll_rad':float(peak[valid].max()),
                'using_intra_tick_peak':'peak_roll' in d,
                'all_over_0p302_intervals':intervals(valid&(peak>.302),t,dt)}
            arrays[f'{case}__{method}__peak_abs_roll']=peak
        severe_quiet=held((np.abs(gov_rate[:,0])<=.1)&(np.abs(gov_rate[:,1])<=.02),dt,.5)
        severe=[]
        for channel,j,tol in [('speed',0,.15),('steer',1,.04)]:
            for segment in intervals(valid&severe_quiet&(np.abs(lower[:,j])>tol),t,dt):
                if segment['duration_s']<=.5:continue
                sl=slice(segment['start_index'],segment['end_index_exclusive'])
                segment.update(channel=channel,governed_min=gov[sl].min(axis=0).tolist(),governed_max=gov[sl].max(axis=0).tolist(),
                    lower_error_min=lower[sl,j].min().item(),lower_error_max=lower[sl,j].max().item(),
                    final_command_clip_fraction=float(np.mean(d['final_command_clipped'][sl])) if 'final_command_clipped' in d else None,
                    peak_roll=float(np.max(d['peak_roll'][sl])) if 'peak_roll' in d else None)
                severe.append(segment)
        result['severe_persistent_after_quiet_0p5s']=severe
        if 'latent_z' in d:
            z=np.tanh(d['latent_z']);proposal=raw+np.column_stack([z[:,0]*np.where(z[:,0]>=0,.25,1.),.2*z[:,1]])
            arrays[f'{case}__{method}__proposal']=proposal
        for field,value in [('raw',raw),('governed',gov),('actual',actual)]:arrays[f'{case}__{method}__{field}']=value
        output['cases'][f'{case}/{method}']=result
        for k,v in (('time',t),('upper_correction',upper),('lower_tracking_error',lower),('task_error',total),('governed_quiet_mask',quiet)):
            arrays[f'{case}__{method}__{k}']=v
    with (destination/'lower_tracking_report.json').open('w') as f: json.dump(output,f,indent=2,ensure_ascii=False,allow_nan=False)
    if rows:
        with (destination/'lower_tracking_summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(destination/'lower_tracking_errors.npz',**arrays)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=run(a.input,a.output);print(f"Audited {len(r['cases'])} stored trace groups; no new simulation.")
