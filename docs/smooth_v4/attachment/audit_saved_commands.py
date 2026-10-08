"""Offline command-lineage and smoothness audit; zero physics/model calls.

Usage: python audit_saved_commands.py --npz .../timeseries.npz \
       --evidence .../evidence.json --output command_audit.json

Input schema is taken from d269f71: case__method__field. Shared method token is
resolved by suffix; if ambiguous, abort rather than pick a convenient array.
No claim that a generic waveform metric alone diagnoses control quality.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def shared(z, case, field):
    keys=[k for k in z.files if k.startswith(case+'__') and k.endswith('__'+field)]
    if not keys:
        raise KeyError(f'Missing shared {case}/{field}')
    x=z[keys[0]]
    if any(not np.array_equal(x,z[k]) for k in keys[1:]):
        raise ValueError(f'Ambiguous shared values: {keys}')
    return x


def contiguous_runs(mask):
    ids=np.flatnonzero(mask)
    if not len(ids):return []
    return np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)


def stats(x,dt,mask):
    """Within-mask transitions only; remove mask-edge jumps from derivatives."""
    x=np.asarray(x,dtype=float)
    if x.ndim!=1:raise ValueError('one channel expected')
    edge=mask[1:] & mask[:-1]
    rate=np.diff(x)/dt
    edge2=edge[1:] & edge[:-1]
    acc=np.diff(rate)/dt
    selected=rate[edge]
    direction=(np.abs(rate)>.05) & edge
    adjacent=direction[1:]&direction[:-1]
    switches=int(np.sum(adjacent & (np.sign(rate[1:])!=np.sign(rate[:-1]))))
    durations=float(np.sum(edge)*dt)
    hf=tot=0.
    for ids in contiguous_runs(mask):
        if len(ids)<max(64,round(1.2/dt)):continue
        a=x[ids];q=np.arange(len(a));a=a-np.polyval(np.polyfit(q,a,1),q)
        power=np.abs(np.fft.rfft(a*np.hanning(len(a))))**2
        freq=np.fft.rfftfreq(len(a),dt)
        hf+=float(power[freq>=5].sum());tot+=float(power[freq>0].sum())
    def rms(a):return None if not a.size else float(np.sqrt(np.mean(a*a)))
    return {'samples':int(mask.sum()),'valid_interval_seconds':durations,
            'total_variation':float(np.sum(np.abs(np.diff(x)[edge]))),
            'variation_per_second':None if durations==0 else float(np.sum(np.abs(np.diff(x)[edge]))/durations),
            'rate_rms':rms(selected),'rate_p95_abs':None if not selected.size else float(np.quantile(abs(selected),.95)),
            'acceleration_rms':rms(acc[edge2]),'direction_reversals_adjacent_rate_above_0p05':switches,
            'above_5Hz_power_fraction':None if tot<1e-16 else hf/tot}


def analyze(npz:Path, evidence:dict):
    action=evidence['selected_configuration']['action']
    out={'scope':'saved traces only; no new rollout', 'source':str(npz),'cases':{}}
    with np.load(npz,allow_pickle=False) as z:
        cases=list(evidence['protocol']['cases'])
        for case in cases:
            t=shared(z,case,'time').reshape(-1).astype(float)
            raw=shared(z,case,'limited_command').astype(float)
            if raw.shape!=(len(t),2):raise ValueError(f'{case}: command shape')
            dt=float(np.median(np.diff(t)))
            if not np.isclose(dt,.005,atol=2e-6):raise ValueError(f'{case}: unexpected time step {dt}')
            raw_rate=np.vstack([np.zeros(2),np.diff(raw,axis=0)/dt])
            # Require raw command to have been quiet for at least 0.4 s.
            quiet=(abs(raw_rate[:,0])<=.1+1e-5)&(abs(raw_rate[:,1])<=.02+1e-5)
            stable=np.zeros(len(t),bool)
            for ids in contiguous_runs(quiet):
                if len(ids)>80:stable[ids[80:]]=True
            rows={}
            for method in ('B0','pi_alpha0','pi_alpha1'):
                prefix=case+'__'+method+'__'
                req=[prefix+f for f in ['governed','latent_z','actual_delta','actual_forward_speed']]
                if any(k not in z.files for k in req):
                    rows[method]={'state':'missing_required_fields','missing':[k for k in req if k not in z.files]};continue
                governed=z[prefix+'governed'].astype(float)
                latent=z[prefix+'latent_z'].astype(float)
                actual=z[prefix+'actual_delta'].reshape(-1).astype(float)
                offsets=governed-raw
                a=np.tanh(latent)
                targets=np.column_stack([np.where(a[:,0]>=0,action['speed_positive_scale_m_s'],action['speed_negative_scale_m_s'])*a[:,0],
                                         action['steer_scale_rad']*a[:,1]])
                # Endpoint snapshots at 50 Hz, do not differentiate sample-held noise at 200 Hz.
                ix=np.arange(3,len(t),4);mask=stable[ix];pdt=4*dt
                q={}
                for name,x in [('proposed_steer_offset',targets[ix,1]),('executed_steer_offset',offsets[ix,1]),
                               ('governed_steer_reference',governed[ix,1]),('actual_steer',actual[ix])]:
                    q[name]=stats(x,pdt,mask)
                q['max_abs_executed_offsets']=np.max(np.abs(offsets),axis=0).tolist()
                q['protected_vs_proposed_offset_rmse']=np.sqrt(np.mean((targets-offsets)**2,axis=0)).tolist()
                if prefix+'cap_fraction' in z.files:
                    q['original_global_cost_cap_fraction']=float(np.mean(z[prefix+'cap_fraction']))
                quiet_error=actual-raw[:,1]
                q['quiet_actual_steer_rmse']=None if not stable.any() else float(np.sqrt(np.mean(quiet_error[stable]**2)))
                rows[method]=q
            out['cases'][case]={'dt':dt,'shared_raw_same_for_all_methods':True,
                'raw_rate_max_abs':np.max(abs(raw_rate),axis=0).tolist(),'methods':rows}
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--npz',type=Path,required=True)
    ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=analyze(args.npz,json.loads(args.evidence.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Saved offline report:',args.output)

if __name__=='__main__':main()
