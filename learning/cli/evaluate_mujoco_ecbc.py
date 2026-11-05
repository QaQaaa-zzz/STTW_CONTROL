"""Reuse the original TeleopEnv and saved Local300 commands for zero-residual GPU evaluation."""
import argparse,json,pickle,time,os
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
import jax
import jax.numpy as jp
from dataclasses import asdict
from sttw_control.teleop_env import TeleopEnv
from sttw_control.runtime import configure_compilation_cache
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True,exist_ok=False)
configure_compilation_cache();env=TeleopEnv(backend='mjx');bank=pickle.load(open(cfg['initial_bank'],'rb'));sample=jax.tree.map(lambda x:x[cfg['initial_bank_index']],bank);cases=cfg['cases'];states=jax.tree.map(lambda x:jp.repeat(jp.asarray(x)[None],len(cases),axis=0),sample)
streams=jp.asarray(np.stack([np.load(Path(cfg['reference_streams'])/(case+'.npy')) for case in cases],axis=1));steps=round(cfg['seconds']/.005)
(a.output/'contract.json').write_text(json.dumps(dict(settings=cfg,device=str(jax.devices()),model=env.bundle.identity,controller=asdict(env.cc),actuator=asdict(env.ac),role='paired zero residual ECBC+ESO; native MJX; no policy/optimizer',budget_physics_lane_steps=steps*len(cases)*25),indent=2))
def one(state,command):
 active=~state.failed;state=state.replace(raw=command)
 state,log=env._step(state,command,True,exact_governed=True)
 log['active_tick']=active
 return state,log
@jax.jit
def chunk(state,commands):
 return jax.lax.scan(lambda st,cmd:jax.vmap(one)(st,cmd),state,commands)
started=time.time();chunks=[]
for k in range(0,steps,50):
 states,log=chunk(states,streams[k:k+50,:,1:3]);jax.block_until_ready(states);chunks.append(jax.device_get(log))
 if k==0 or (k+50)%400==0:print(json.dumps(dict(step=k+50,elapsed_s=time.time()-started,alive=int((~states.failed).sum()))),flush=True)
logs=jax.tree.map(lambda *xs:np.concatenate(xs),*chunks)
for i,case in enumerate(cases):
 d={key:val[:,i] for key,val in logs.items()};valid=d['active_tick'];d={key:val[valid] for key,val in d.items()};d['time']=np.arange(len(d['actual_xy']))*.005;np.savez_compressed(a.output/(case+'.npz'),**d)
(a.output/'status.json').write_text(json.dumps(dict(stage='complete',cases=cases,survived=np.asarray(~states.failed).tolist(),physical_lane_steps=steps*len(cases)*25,optimizer_updates=0,elapsed_s=time.time()-started),indent=2))
