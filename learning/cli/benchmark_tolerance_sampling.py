"""Short restored-input GPU benchmark; synthetic terminal ticks are engineering only."""
import argparse,json,time,os
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from sttw_control.path_command_training import PathBatch,load_environment
from sttw_control.path_command_statistics import reduce_statistics
from sttw_control.tolerance_sampling import make_packed_advance
from sttw_control.preference_best import atomic_json
p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
status=dict(state='running',phase='running',pid=os.getpid(),engineering_only=True)
def update(**kw):
 status.update(kw);status['checked_unix']=time.time();atomic_json(out/'status.json',status)
try:
 update(stage='loading')
 from sttw_control.runtime import configure_compilation_cache
 configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
 plan=json.loads(Path(a.plan).read_text());cfg=json.loads(Path(plan['config']).read_text());source=json.loads(Path(plan['parent_manifest']).read_text())['source']
 env,snapshot=load_environment(cfg,source,'terminal_contract_v2');batch=PathBatch(env,snapshot,cfg,plan['seed'],plan['alpha'])
 sample=batch.reset(j.int32(0),j.int32(0));env.set_log_template(sample.state,sample.route.path,sample.route.speed)
 n=512;ep=jax.jit(jax.vmap(batch.reset))(j.arange(n,dtype=j.int32),j.zeros(n,j.int32));jax.block_until_ready(ep);actions=j.zeros((n,2))
 def old(ep,z):
  end,obs,r,flags,_,stats=jax.vmap(batch.advance_one)(ep,z);stats['diagnostics']=reduce_statistics(stats['diagnostics']);return end,obs,r,flags,stats
 funcs={};compile_s={}
 for name,fn in [('original',old),('packed',make_packed_advance(batch))]:
  update(stage='compiling_'+name);start=time.monotonic();funcs[name]=jax.jit(fn).lower(ep,actions).compile();compile_s[name]=time.monotonic()-start
 results={}
 for case,states in [('no_reset',ep),('reset17',ep.replace(state=ep.state.replace(tick=j.where(j.arange(n)<17,round(batch.horizon/.005)-4,ep.state.tick))))]:
  update(stage='benchmark_'+case);outputs={};timings={}
  for name,fn in funcs.items():
   outputs[name]=fn(states,actions);jax.block_until_ready(outputs[name]);timings[name]=[]
  for repeat in range(3):
   for name in (['original','packed'] if repeat%2==0 else ['packed','original']):
    start=time.monotonic();value=funcs[name](states,actions);jax.block_until_ready(value);timings[name].append(time.monotonic()-start)
  differences=[];discrete_equal=True;finite=True
  for x,y in zip(jax.tree.leaves(outputs['original']),jax.tree.leaves(outputs['packed'])):
   x,y=np.asarray(x),np.asarray(y)
   if np.issubdtype(x.dtype,np.inexact):
    finite=finite and bool(np.isfinite(x).all() and np.isfinite(y).all());differences.append(float(np.max(np.abs(x-y))) if x.size else 0.)
   else:discrete_equal=discrete_equal and bool(np.array_equal(x,y))
  results[case]=dict(seconds=timings,median_seconds={k:float(np.median(v)) for k,v in timings.items()},all_discrete_equal=discrete_equal,all_finite=finite,max_all_float_leaf_difference=max(differences),reward_max_abs_difference=float(np.max(np.abs(np.asarray(outputs['original'][2])-np.asarray(outputs['packed'][2])))),actual_reset_count=int(j.sum(outputs['packed'][3]['done'] & ~outputs['packed'][3]['engineering_fault'])))
  atomic_json(out/'results.json',dict(batch=n,compile_seconds=compile_s,cases=results,engineering_only=True,synthetic_terminal_ticks=True,lower_ticks_executed=2*len(results)*4*n*4))
 update(state='completed',phase='completed',stage='done')
except BaseException as e:
 update(state='error',phase='error',error=f'{type(e).__name__}: {e}');raise
