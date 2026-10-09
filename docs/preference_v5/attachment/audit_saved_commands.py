"""Read existing NPZ; do not simulate, optimize, or modify its contents.
Usage: python audit_saved_commands.py /path/to/timeseries.npz --output audit.json
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def longest_run(mask,dt):
    padded=np.r_[False,np.asarray(mask,bool),False]
    edges=np.flatnonzero(padded[1:]!=padded[:-1])
    return float(np.max(edges[1::2]-edges[::2])*dt) if len(edges) else 0.

def stats(x):
    x=np.asarray(x,float)
    if x.size==0:return {'count':0}
    if not np.isfinite(x).all():return {'count':int(x.size),'nonfinite':True}
    return {'count':int(x.size),'mean':float(x.mean()),'mean_abs':float(np.abs(x).mean()),
            'minimum':float(x.min()),'maximum':float(x.max()),
            'p05':float(np.quantile(x,.05)),'p95':float(np.quantile(x,.95)),
            'rms':float(np.sqrt(np.mean(x*x)))}

def main(path:Path,output:Path):
    out={'source':str(path),'new_physics_steps':0,'methods':{},'warnings':[]}
    with np.load(path,allow_pickle=False) as z:
        keys=set(z.files)
        prefixes=sorted(k[:-len('__actual_forward_speed')] for k in keys if k.endswith('__actual_forward_speed'))
        if not prefixes:raise ValueError('No case__method__actual_forward_speed keys. Use schema in evidence.json.')
        for prefix in prefixes:
            case=prefix.split('__')[0]
            v=np.asarray(z[prefix+'__actual_forward_speed']).reshape(-1)
            n=len(v)
            def field(name):
                choices=[prefix+'__'+name,case+'__shared__'+name,case+'__common__'+name,case+'__'+name]
                for key in choices:
                    if key in keys and len(z[key])>=n:return np.asarray(z[key])[:n],key
                others=sorted(k for k in keys if k.startswith(case+'__') and k.endswith('__'+name) and len(z[k])>=n)
                if others:
                    a=np.asarray(z[others[0]])[:n]
                    if all(np.allclose(a,np.asarray(z[k])[:n],rtol=0,atol=1e-5,equal_nan=False) for k in others):
                        return a,others[0]
                return None,None
            t,tk=field('time'); raw,rk=field('limited_command'); gov,gk=field('governed')
            if t is None or raw is None or gov is None:
                out['methods'][prefix]={'unavailable':'time/raw/governed missing or ambiguous; do not infer'};continue
            t=t.reshape(-1);dt=float(np.median(np.diff(t)))
            d=gov-raw
            record={'fields':{'time':tk,'raw':rk,'governed':gk},'steps':n,'end_time':float(t[-1]),'windows':{}}
            for label,(lo,hi) in {'all_saved':(float(t[0]),float(t[-1]+dt)),
                                  'legacy_1_6':(1.,6.),'steady_2_4':(2.,4.),'last_half_second':(t[-1]+dt-.5,t[-1]+dt)}.items():
                mask=(t>=lo-1e-7)&(t<hi-1e-7)
                if not mask.any():continue
                w={'speed_error':stats((v-raw[:,0])[mask]),'upper_dv':stats(d[mask,0]),'upper_ddelta':stats(d[mask,1])}
                for thresh in [-.15,-.30]:
                    negative=mask&(d[:,0]<=thresh)
                    w[f'dv_le_{thresh}']={'seconds':float(negative.sum()*dt),'longest_seconds':longest_run(negative,dt)}
                for f in ['actual_delta','phi','e_psi_unwrapped']:
                    a,k=field(f)
                    if a is not None:
                        values=a.reshape(-1)
                        if f=='actual_delta':values=values-raw[:,1]
                        w[f]=stats(values[mask])
                latent,lk=field('latent_z')
                if latent is not None:
                    a=np.tanh(latent)
                    request=np.stack([a[:,0]*np.where(a[:,0]>=0,.25,1.),.2*a[:,1]],axis=-1)
                    w['requested_dv']=stats(request[mask,0]);w['requested_ddelta']=stats(request[mask,1])
                    w['request_minus_governed_dv']=stats((request[:,0]-d[:,0])[mask])
                record['windows'][label]=w
            record['warning']='Final total residual alone cannot uniquely identify upper vs lower contribution. No control-chain attribution without separate logged signals.'
            out['methods'][prefix]=record
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('npz',type=Path);p.add_argument('--output',type=Path,default=Path('saved_command_audit.json'))
    a=p.parse_args();main(a.npz,a.output)
