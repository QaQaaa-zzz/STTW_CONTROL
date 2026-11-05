"""Offline JAX oracle for frozen LOCAL300; no simulation or optimization."""
import argparse
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace
# Only the unused Torch export helpers in direct_command_policy require these
# imports. This oracle invokes its unchanged JAX frame function, not Torch code.
sys.modules['torch'] = types.ModuleType('torch')
sys.modules['torch'].nn = types.ModuleType('torch.nn')
import jax
import jax.numpy as j
import numpy as np
from sttw_control.frozen_local_lower import FrozenLocalLower
from sttw_control.controller import ControllerConfig, initial_controller
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
manifest=json.load(open('/home/qy/STTW_CONTROL/runs/model_registry/middle_lineage_20261024_v1/manifest.json'))
record=manifest['common_frozen_lower']['actor'];data=Path(record['original']).read_bytes()
assert hashlib.sha256(data).hexdigest()==record['sha256']
original=pickle.loads(data)
np.savez(a.output/'actor.npz',**{f'{layer}_{name}':np.asarray(value) for layer,values in original['params'].items() for name,value in values.items()})
portable=dict(original=str((a.output/'actor.npz').resolve()),sha256=hashlib.sha256((a.output/'actor.npz').read_bytes()).hexdigest())
params=jax.tree.map(j.asarray,original);lower=FrozenLocalLower(params);cc=ControllerConfig()
rng=np.random.default_rng(20261105);n=48;lanes=3
m=rng.normal(size=(n,lanes,7)).astype('float32');speed=rng.uniform(1,3,(n,lanes)).astype('float32')
ref=rng.uniform(-1,1,(n,lanes,2)).astype('float32')*np.array([.5,.2],dtype='float32')+np.array([2.3,0],dtype='float32')
prev=ref-rng.normal(0,.001,ref.shape).astype('float32');command=rng.normal(size=ref.shape).astype('float32')
applied=rng.normal(size=ref.shape).astype('float32');yaw=rng.normal(size=speed.shape).astype('float32');dist=rng.normal(size=speed.shape).astype('float32')
reset=np.zeros((n,lanes),bool);reset[17,1]=True;reset[31,0]=True
states=[lower.initial(None) for _ in range(lanes)];actions=[];observations=[]
@jax.jit
def step(s,m,v,r,pr,cmd,d):
    physical=SimpleNamespace(governor=SimpleNamespace(current_reference=pr),actuator=SimpleNamespace(previous=cmd),controller=initial_controller(cc).replace(disturbance=d))
    new,act,obs,_=lower.prepare_local(s,m,None,v,r,physical,cc)
    return new,act,obs
for tick in range(n):
    aa=[];oo=[]
    for lane in range(lanes):
        if reset[tick,lane]:states[lane]=lower.initial(None)
        states[lane],act,obs=step(states[lane],m[tick,lane],speed[tick,lane],ref[tick,lane],prev[tick,lane],command[tick,lane],dist[tick,lane])
        aa.append(np.asarray(act));oo.append(np.asarray(obs))
        states[lane]=lower.finish_local(states[lane],j.asarray(applied[tick,lane]),j.asarray(yaw[tick,lane]))
    actions.append(aa);observations.append(oo)
np.savez_compressed(a.output/'oracle.npz',measurement=m,speed=speed,reference=ref,previous=prev,command=command,applied=applied,yaw=yaw,disturbance=dist,reset=reset,action=actions,observation=observations)
from dataclasses import asdict
(a.output/'identity.json').write_text(json.dumps(dict(source_actor=record,actor=portable,controller=asdict(cc),ticks=n,lanes=lanes,physics_steps=0,optimizer_updates=0),indent=2))
print('saved',a.output)
