"""Independent frozen V1 schema and evaluator-owned causal v/delta inputs."""
import json
import math
from pathlib import Path
import numpy as np

CONFIG_PATH=Path(__file__).resolve().parents[2]/'configs/teleop_pref_governor_v1.json'

def load_teleop_config(path=CONFIG_PATH):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError(f'duplicate config field: {k}')
            out[k]=v
        return out
    raw=json.loads(Path(path).read_text(),object_pairs_hook=unique)
    import hashlib
    if hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()!='10292fd64f7581a85911c18a041c02dfb755e04bdeb3b457788b4585bbffdb92':raise ValueError('frozen V1 schema modified; new protocol version required')
    canonical=json.loads(CONFIG_PATH.read_text())
    # V1 is a frozen experiment protocol, not a generic tunable TaskConfig.
    # Reject unknown fields AND altered values; a changed protocol needs a new version.
    def check(a,b,p):
        if isinstance(b,dict):
            if not isinstance(a,dict) or a.keys()!=b.keys():raise ValueError(f'{p}: missing/unknown fields')
            for k in b:check(a[k],b[k],f'{p}.{k}')
        elif isinstance(b,list):
            if not isinstance(a,list) or len(a)!=len(b):raise ValueError(f'{p}: invalid list')
            for i,(x,y) in enumerate(zip(a,b)):check(x,y,f'{p}[{i}]')
        elif isinstance(b,bool):
            if type(a) is not bool or a!=b:raise ValueError(f'{p}: frozen flag')
        elif isinstance(b,(int,float)):
            if isinstance(a,bool) or not isinstance(a,(int,float)) or not math.isfinite(a) or a!=b:raise ValueError(f'{p}: frozen numeric value')
        elif type(a)!=type(b) or a!=b:raise ValueError(f'{p}: frozen value')
    check(raw,canonical,'config')
    return raw

def validate_alpha(alpha):
    if isinstance(alpha,bool) or not np.isfinite(alpha) or alpha not in (0,1):
        raise ValueError('alpha must be finite and exactly 0 or 1')
    return int(alpha)

def limit_command(current,target,dt=.005):
    current=np.asarray(current,dtype=float);target=np.asarray(target,dtype=float)
    if current.shape!=(2,) or target.shape!=(2,) or not np.isfinite([current,target]).all():
        raise ValueError('command must contain finite speed m/s and steer rad')
    target=np.clip(target,[2.,-.3],[2.6,.3])
    return current+np.clip(target-current,-dt*np.array([.5,.3]),dt*np.array([.5,.3]))

def random_schedule(seed):
    rng=np.random.Generator(np.random.PCG64(seed));rows=[[0.,2.3,0.]];t=2.
    while t<10:
        v=rng.uniform(2.,2.6);d=rng.uniform(-.3,.3);duration=rng.uniform(.8,1.6)
        rows.append([t,v,d]);t+=duration
    rows.append([10.,2.3,0.])
    return np.asarray(rows)

def command_stream(rows,ticks=3200,dt=.005):
    """Only evaluator holds schedule. c[0]=(2.3,0); advance after logging tick."""
    rows=np.asarray(rows,float);c=np.array([2.3,0.]);out=[]
    for k in range(ticks):
        target=rows[np.searchsorted(rows[:,0],k*dt,side='right')-1,1:]
        out.append(np.r_[target,c]);c=limit_command(c,target,dt)
    return np.asarray(out)
