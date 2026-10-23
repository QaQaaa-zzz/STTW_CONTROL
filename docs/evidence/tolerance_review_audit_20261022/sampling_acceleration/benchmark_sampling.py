import sys,os,time,json,pickle,importlib.util
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from sttw_control.path_command_training import load_environment,PathBatch
from sttw_control.short_recovery import ShortRecoveryBatch
from sttw_control.tolerance_training import experiment_config
from sttw_control.preference_best import atomic_json
from sttw_control.runtime import configure_compilation_cache
sys.path.insert(0,str(Path.cwd()/'learning/cli'))
from prepare_short_recovery_campaign import register
out=Path('runs/sampling_acceleration_review_v1_attempt3').resolve();out.mkdir(exist_ok=False)
state=dict(state='running',phase='running',pid=os.getpid(),training_steps=0,maximum_transition_executions=8*10)
def update(**kw):
 state.update(kw);state['checked_unix']=time.time();atomic_json(out/'status.json',state)
update(stage='loading');register(out/'ACTIVE_RUN.json','STTW bounded sampling equivalence benchmark',out/'status.json')
try:
 configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
 source=Path('/tmp/sttw_reference_short_recovery.py').read_text();(out/'reference_short_recovery.py').write_text(source)
 spec=importlib.util.spec_from_file_location('sttw_control._reference_recovery',out/'reference_short_recovery.py')
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 from sttw_control.short_recovery import RecoveryEpisode
 module.RecoveryEpisode=RecoveryEpisode
 bankdir=Path('runs/short_recovery_train_v1/banks/alpha0');manifest=json.loads((bankdir/'manifest.json').read_text());cfg=manifest['config']
 with (bankdir/'bank.pkl').open('rb') as f:bank=pickle.load(f)
 env,snapshot=load_environment(cfg,manifest['source'],'terminal_contract_v2');base=PathBatch(env,snapshot,cfg,910601,0)
 args=dict(guided=False,seed=910602,reserve_config=experiment_config(cfg,'dynamic'))
 old=module.ShortRecoveryBatch(base,bank,**args);new=ShortRecoveryBatch(base,bank,**args)
 ep=jax.jit(jax.vmap(old.reset))(j.arange(8),j.zeros(8,j.int32));jax.block_until_ready(ep)
 single=jax.tree.map(lambda x:x[0],ep);env.set_log_template(single.state,single.route.path,single.route.speed)
 z=jax.random.normal(jax.random.PRNGKey(8201),(8,2))*.1
 funcs={};compile_seconds={}
 for name,fn in [('reference',jax.vmap(old.advance_one)),('optimized',new.advance_batch)]:
  update(stage='compiling_'+name);start=time.monotonic();funcs[name]=jax.jit(fn).lower(ep,z).compile();compile_seconds[name]=time.monotonic()-start
  jax.block_until_ready(funcs[name](ep,z))
 reports=[];maxdiff=0.;ticks=0
 for boundary in [False,True]:
  state_ep=ep
  if boundary:state_ep=ep.replace(origin_tick=ep.origin_tick-j.where(j.arange(8)%2==0,796,0))
  outputs={};times={}
  for name in ['reference','optimized','optimized','reference']:
   update(stage='checking_'+name,boundary=boundary)
   start=time.monotonic();result=funcs[name](state_ep,z);jax.block_until_ready(result);elapsed=time.monotonic()-start
   times.setdefault(name,[]).append(elapsed);outputs[name]=jax.device_get(result);ticks+=int(np.sum(outputs[name][5]['active_ticks']))
  a,b=outputs['reference'],outputs['optimized']
  aa,ta=jax.tree.flatten(a);bb,tb=jax.tree.flatten(b);assert ta==tb
  differences=[]
  for index,(x,y) in enumerate(zip(aa,bb)):
   x,y=np.asarray(x),np.asarray(y)
   if x.dtype.kind in 'biu':np.testing.assert_array_equal(x,y,err_msg=str(index))
   elif x.size:
    assert np.isfinite(x).all() and np.isfinite(y).all()
    delta=float(np.max(np.abs(x-y)));maxdiff=max(maxdiff,delta)
    if delta: differences.append(dict(leaf=index,max_abs_difference=delta))
  metrics={}
  for label,x,y,bound in [
   ('qpos',a[0].state.physical.data.qpos,b[0].state.physical.data.qpos,2e-5),
   ('qvel',a[0].state.physical.data.qvel,b[0].state.physical.data.qvel,2e-4),
   ('actor_obs',a[1][0],b[1][0],2e-4),
   ('critic_obs',a[1][1],b[1][1],2e-4),
   ('reward',a[2],b[2],2e-5),
   ('impulse',a[5]['applied_force_impulse'],b[5]['applied_force_impulse'],1e-5)]:
   metrics[label]=float(np.max(np.abs(x-y)))
   assert metrics[label]<=bound,(label,metrics[label],bound)
  reports.append(dict(boundary=boundary,seconds=times,segment_timeouts=int(a[3]['segment_timeout'].sum()),leaves_checked=len(aa),numerical_differences=differences,control_differences=metrics))
 atomic_json(out/'results.json',dict(reference_revision='c422c63',num_envs=8,actions='fixed PRNGKey8201 normal*.1',compile_seconds=compile_seconds,reports=reports,max_abs_difference=maxdiff,active_lower_ticks=ticks,comparison='all leaves finite; exact discrete; explicit channel tolerances, not bitwise identity; shared GPU, not full rollout speed estimate',training_steps=0))
 update(state='completed',phase='completed',stage='verified',active_lower_ticks=ticks)
 print((out/'results.json').read_text(),flush=True)
except BaseException as exc:
 update(state='error',phase='error',error=f'{type(exc).__name__}: {exc}');raise
