#!/usr/bin/env python3
"""One causal prefix replay and paired two-second inherited-state recovery."""
import argparse,os,sys,json,pickle,time,hashlib
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import jax,jax.numpy as jp
from sttw_control.direct_command_env import DirectCommandEnv
from sttw_control.lower_command import LowerCommandEnv,LowerState
from sttw_control.network import ResidualActor
from sttw_control.lower_command_validation import summarize_parent_recovery
from sttw_control.runtime import configure_compilation_cache
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args()
source=Path(a.source);run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
def write(name,d):(out/name).write_text(json.dumps(d,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def flatten(d,prefix=''):
 result={}
 for k,v in d.items():
  if isinstance(v,dict):result.update(flatten(v,prefix+k+'_'))
  else:result[prefix+k]=np.asarray(v)
 return result
configure_compilation_cache(out/'jax_cache')
write('declaration.json',dict(source=str(source),best_run=str(run),prefix_ticks=1263,paired_ticks=400,budget_control_transitions=2063,history='last ten causal real pre-step states, commands and executed actions from old controller',command=[2.3,0.],no_training=True))
try:
 spec=json.loads((source/'resolved_config.json').read_text());spec['lower_internal_diagnostics']=False
 cfg=json.loads((run/'config.json').read_text());best=json.loads((run/'training/best_model.json').read_text())
 assert sha(best['checkpoint'])==best['checkpoint_sha256']
 old=DirectCommandEnv(spec);local=LowerCommandEnv(cfg)
 # Both methods use the very same source plant, controller and actuator.
 local.physics=old.physics;local.cc=old.cc;local.ac=old.ac
 with (source/'initial_full_state.pkl').open('rb') as f:s=jax.tree.map(jp.asarray,pickle.load(f))
 initial=s
 frame0=jp.zeros((10,20));mask0=jp.zeros(10)
 def causal_frame(s,frames,mask):
  ls=LowerState(s.physical,frames,mask,s.rows,s.slew,s.command_rates,s.previous_bounded/jp.array([1.5,10.]),s.previous_bounded,s.yaw_rate,jp.int32(0),s.env_id,s.episode_index)
  return local.record(ls)
 def prefix_step(c,_):
  s,frames,mask=c;ls=causal_frame(s,frames,mask)
  ns,log=old._tick(s,jp.zeros(2),False)
  ns=jax.lax.cond(ns.tick%4==0,old.record_frame,lambda x:x,ns)
  kept={k:log[k] for k in ('actual_xy','actual_delta','actual_forward_speed','phi','governed','lower_action')}
  return (ns,ls.frames,ls.mask),kept
 write('status.json',dict(state='compiling_prefix',pid=os.getpid()))
 print('Compiling one original prefix to6.315s',flush=True)
 prefix=jax.jit(lambda s,f,m:jax.lax.scan(prefix_step,(s,f,m),None,length=1263))
 (s,frames,mask),logs=prefix(s,frame0,mask0);jax.block_until_ready(s)
 logs=jax.device_get(logs);np.savez_compressed(out/'prefix_overlap.npz',**logs)
 saved=np.load(source/'trace.npz');overlap={k:float(np.max(abs(logs[k]-saved[k][:1263]))) for k in ('actual_xy','actual_delta','actual_forward_speed','phi','governed')}
 write('prefix_overlap.json',overlap)
 assert overlap['actual_delta']<.001 and overlap['actual_forward_speed']<.001 and overlap['actual_xy']<.005,overlap
 with (out/'old_full_state_6p315.pkl').open('wb') as f:pickle.dump(jax.device_get(s),f)
 ls=causal_frame(s,frames,mask);ls=ls.replace(previous_action=jp.asarray(logs['lower_action'][-1]))
 # The source command is already2.3/0. No plant/ESO/actuator or original path resets.
 command=jp.array([2.3,0.]);assert np.allclose(s.physical.raw,command,atol=1e-5)
 rows=jp.zeros_like(s.rows).at[:,0].set(99.).at[0].set(jp.array([0.,2.3,0.]))
 s=s.replace(rows=rows,command_rates=jp.zeros(2),physical=s.physical.replace(raw=command))
 stream=jp.tile(jp.array([0.,2.3,0.,0.,0.]),(401,1)).at[:,0].set(jp.arange(401)*.005)
 ls=ls.replace(rows=rows,issued_stream=stream,rates=jp.zeros(2),physical=s.physical,tick=jp.int32(0))
 with (out/'local_full_state_6p315.pkl').open('wb') as f:pickle.dump(jax.device_get(ls),f)
 # Force diagnostics only observe physical substeps; old Actor diagnostics disabled.
 old.physics.spec={**old.physics.spec,'lower_internal_diagnostics':True}
 with (run/'training/best_actor.pkl').open('rb') as f:params=jax.tree.map(jp.asarray,pickle.load(f))
 net=ResidualActor(tuple(cfg['hidden_sizes']),activation='elu')
 def old_step(s,_):
  ns,log=old._tick(s,jp.zeros(2),False)
  ns=jax.lax.cond(ns.tick%4==0,old.record_frame,lambda x:x,ns)
  return ns,log
 def new_step(s,_):
  obs,_=local.observation(s);ns,_,_,log=local.step(s,net.apply(params,obs,return_logits=True));return ns,log
 initial_roll=float(old.physics.observe(s.physical.data)[0][0]);results={}
 for name,state,step in [('R196',s,old_step),('best300',ls,new_step)]:
  write('status.json',dict(state='evaluating',method=name,pid=os.getpid(),prefix_ticks=1263))
  print('Paired recovery '+name,flush=True)
  fn=jax.jit(lambda state:jax.lax.scan(step,state,None,length=400));end,logs=fn(state);jax.block_until_ready(end)
  d=flatten(jax.device_get(logs));d['time']=np.arange(400)*.005;d['source_time']=d['time']+6.315
  failed=np.flatnonzero(d['physical_failure'])
  if failed.size:d={k:v[:failed[0]+1] for k,v in d.items()}
  np.savez_compressed(out/(name+'.npz'),**d)
  with (out/(name+'_final_full_state.pkl')).open('wb') as f:pickle.dump(jax.device_get(end),f)
  results[name]=summarize_parent_recovery(d,initial_roll=initial_roll)
  results[name]['force_peak_abs']=np.maximum(abs(d['actuator_force_min']),abs(d['actuator_force_max'])).max(axis=0).tolist()
  results[name]['force_limit_substeps']=d['actuator_force_limit_substeps'].sum(axis=0).tolist()
 write('results.json',dict(best_update=best['source_update'],best_checkpoint_sha256=best['checkpoint_sha256'],prefix_overlap=overlap,methods=results,adoption='not_adopted',two_consecutive_late_validation='only300passed',upper_comparison='not_run'))
 write('status.json',dict(state='completed',prefix_ticks=1263,paired_ticks=800,training_transitions=0));print(json.dumps(results),flush=True)
except BaseException as e:
 write('status.json',dict(state='error',type=type(e).__name__,message=str(e)));raise
