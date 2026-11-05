"""Frozen H75 JAX inference/correction oracle; no physics or optimization."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from sttw_control.path_command_policy import make_path_actor,correction_tick
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
root=Path('/home/qy/STTW_CONTROL/runs/model_registry/middle_lineage_20261024_v1')
m=json.loads((root/'manifest.json').read_text());node=next(x for x in m['nodes'] if x['id']=='STTW_MIDDLE_FULLTAIL_PPO75_U77')
cdata=(root/node['resolved_config']['path']).read_bytes();assert hashlib.sha256(cdata).hexdigest()==node['resolved_config']['sha256']
c=json.loads(cdata);data=Path(node['actor']['original']).read_bytes();assert hashlib.sha256(data).hexdigest()==node['actor']['sha256']
params=pickle.loads(data);arrays={}
for key,value in params['params'].items():
    if isinstance(value,dict):arrays.update({f'{key}_{k}':np.asarray(v) for k,v in value.items()})
    else:arrays[key]=np.asarray(value)
np.savez(a.output/'actor.npz',**arrays)
params=jax.tree.map(j.asarray,params);net=make_path_actor(c)
rng=np.random.default_rng(20261105);n=80;alphas=c['network']['alphas'];lanes=len(alphas)
obs=rng.uniform(-5,5,(n,lanes,351)).astype('float32');obs[:,:,336]=alphas
nominal=rng.uniform([.1,-.6],[4.,.6],(n,lanes,2)).astype('float32')
reset=np.zeros((n,lanes),bool);reset[29,1]=True;reset[53,0]=True
latent=jax.jit(jax.vmap(jax.vmap(lambda o:net.apply(params,o))))(j.asarray(obs))
correct=jax.jit(jax.vmap(lambda f,o,z,r:correction_tick(f,o,z,r,c)))
f=j.zeros((lanes,2));o=f;filtered=[];offsets=[];governed=[];flags=[]
for tick in range(n):
    f=j.where(j.asarray(reset[tick])[:,None],0,f);o=j.where(j.asarray(reset[tick])[:,None],0,o)
    f,o,g,flag=correct(f,o,latent[tick],j.asarray(nominal[tick]));filtered.append(np.asarray(f));offsets.append(np.asarray(o));governed.append(np.asarray(g))
    flags.append(np.stack([np.asarray(flag[k]) for k in ('reference_clipped','rate_clipped','policy_fault')],axis=-1))
np.savez_compressed(a.output/'oracle.npz',observation=obs,latent=np.asarray(latent),nominal=nominal,reset=reset,filtered=filtered,offsets=offsets,governed=governed,flags=flags)
(a.output/'config.json').write_bytes(cdata)
(a.output/'identity.json').write_text(json.dumps(dict(source_actor=node['actor'],source_config=node['resolved_config'],actor_sha256=hashlib.sha256((a.output/'actor.npz').read_bytes()).hexdigest(),ticks=n,lanes=lanes,alphas=alphas,physics_steps=0,optimizer_updates=0),indent=2))
print('saved',a.output)
