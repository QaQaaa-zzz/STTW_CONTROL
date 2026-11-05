"""Reuse native H75/Local300 path-control execution for a matched flat panel."""
import argparse,json,pickle,time,os
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION','highest')
import numpy as np
import jax
import jax.numpy as jp
from sttw_control.path_command_training import load_environment
from sttw_control.path_command_policy import PathCommandActor,architecture_identity
from sttw_control.geometric_path import build_path
from sttw_control.direct_command_training import flatten_logs
from sttw_control.runtime import configure_compilation_cache
p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=10);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);configure_compilation_cache()
cfg=json.loads((a.bundle/'config.json').read_text());identity=json.loads((a.bundle/'identity.json').read_text());source=json.loads(a.source.read_text());env,snapshot=load_environment(cfg,source)
path=build_path(identity['route_name'],identity['world_anchor'],cfg);speed=jp.float32(2.3);alphas=jp.array([0,.25,.5,.75,1,0]);zero=jp.array([False]*5+[True]);states=jax.vmap(lambda x:env.reset(snapshot,path,speed,x))(alphas)
env.set_log_template(jax.tree.map(lambda x:x[0],states),path,speed)
params=jax.tree.map(jp.asarray,pickle.load(open(identity['source_actor']['source'],'rb')));net=PathCommandActor(architecture=architecture_identity(cfg))
(a.output/'contract.json').write_text(json.dumps(dict(bundle=str(a.bundle.resolve()),identity=identity,source=str(a.source.resolve()),model=env.physics.bundle.identity,alphas=np.asarray(alphas).tolist(),zero_middle=np.asarray(zero).tolist(),initial_bank=source['prepared_bank'],initial_bank_index=source['bank_index'],initial_physical_tick=int(snapshot.physical_tick),device=str(jax.devices()),seconds=a.seconds,physics_dt=.0002,lower_dt=.005,upper_dt=.02,role='new matched friction0.8 H75 DEV reference; not reuse of historical friction10 traces'),indent=2))
def one(state,is_zero):
 obs,_,fault=env.observation(state,path,speed,a.seconds);state=state.replace(fault=state.fault|fault);z=net.apply(params,obs);z=jp.where(is_zero,0.,z)
 return env.policy_step(state,z,path,speed,a.seconds)
@jax.jit
def chunk(states):
 return jax.lax.scan(lambda x,_:jax.vmap(one)(x,zero),states,None,length=50)
start=time.time();chunks=[]
for i in range(round(a.seconds)):
 states,log=chunk(states);jax.block_until_ready(states);chunks.append(jax.device_get(log));print(json.dumps(dict(seconds=i+1,elapsed_s=time.time()-start,alive=int((~states.physical.failed&~states.fault&~states.domain_exit).sum()))),flush=True)
logs=flatten_logs(jax.tree.map(lambda *x:np.concatenate(x),*chunks));labels=['alpha0','alpha025','alpha05','alpha075','alpha1','zero'];counts=[]
for i,name in enumerate(labels):
 lane={key:value[:,i].reshape((-1,)+value.shape[3:]) for key,value in logs.items()};valid=lane['active_tick'].astype(bool);lane={key:value[valid] for key,value in lane.items()};lane['time']=np.arange(valid.sum())*.005;counts.append(int(valid.sum()));np.savez_compressed(a.output/(name+'.npz'),**lane)
(a.output/'status.json').write_text(json.dumps(dict(stage='complete',survived=np.asarray(~states.physical.failed&~states.fault&~states.domain_exit).tolist(),active_control_ticks=counts,allocated_physics_lane_steps=round(a.seconds/.0002)*6,optimizer_updates=0,elapsed_s=time.time()-start),indent=2))
