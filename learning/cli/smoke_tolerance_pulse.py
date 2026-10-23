"""GPU engineering smoke for actual-pulse curriculum; no learning."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,json,time
from pathlib import Path
import jax
import jax.numpy as j
import numpy as np
from sttw_control.path_command_training import PathBatch,load_environment
from sttw_control.tolerance_pulse import PulseBatch
from sttw_control.tolerance_training import experiment_config
from sttw_control.preference_best import atomic_json
from sttw_control.runtime import configure_compilation_cache

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.time();ticks=0
 def status(state,**kw):atomic_json(out/'status.json',dict(state=state,phase=state,pid=os.getpid(),checked_unix=time.time(),elapsed_s=time.time()-start,active_lower_ticks=ticks,maximum_lower_ticks=768,training_steps=0,**kw))
 status('running',stage='loading')
 try:
  if not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU required')
  campaign=Path('runs/tolerance_shaping_train_v1');cfg=json.loads((campaign/'baseline_config.json').read_text());plan=json.loads((campaign/'baseline_alpha0/plan.json').read_text());parent=json.loads(Path(plan['parent_manifest']).read_text());configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache');env,snapshot=load_environment(cfg,parent['source'],'terminal_contract_v2');base=PathBatch(env,snapshot,cfg,910401,0);sample=base.reset(j.int32(0),j.int32(0));env.set_log_template(sample.state,sample.route.path,sample.route.speed);reserve_cfg=experiment_config(cfg,'dynamic');results=[]
  for guided in [False,True]:
   batch=PulseBatch(base,guided=guided,pulse_seed=910402,reserve_config=reserve_cfg)
   ep=jax.jit(jax.vmap(batch.reset))(j.arange(8),j.zeros(8,j.int32));ep=ep.replace(pulse_state=ep.pulse_state.replace(enabled=j.arange(8)>=2,onset_tick=j.full((8,),20,j.int32)))
   advance=jax.jit(jax.vmap(lambda e:batch.advance_one(e,j.zeros(2),auto_reset=False)))
   status('running',stage='compiling',guided=guided)
   for k in range(12):
    ep,obs,reward,flags,final,stats=advance(ep);jax.block_until_ready(ep);ticks+=int(np.sum(stats['active_ticks']))
    if np.any(flags['engineering_fault']) or np.any(flags['physical_failure']) or np.any(flags['domain_exit']):raise RuntimeError('smoke terminal fault')
    if not np.isfinite(np.asarray(reward)).all():raise RuntimeError('nonfinite reward')
    status('running',stage='rollout',guided=guided,completed_intervals=k+1)
   impulse=np.asarray(ep.impulse);expected=.1*np.asarray(ep.pulse_state.pulse[:,2:]);error=float(np.max(np.abs(impulse-expected)))
   np.testing.assert_allclose(impulse[:2],0,atol=1e-8)
   if not np.all(np.linalg.norm(impulse[2:],axis=1)>0):raise RuntimeError('missing pulse')
   if error>.002:raise RuntimeError('gross impulse mismatch')
   results.append(dict(guided=guided,force=np.asarray(ep.pulse_state.force).tolist(),impulse=impulse.tolist(),expected=expected.tolist(),maximum_impulse_error_Ns=error))
  atomic_json(out/'results.json',dict(scope='engineering only; zero policy corrections; onset .1s duration .1s simulation .24s; task horizon20s unchanged',results=results,active_lower_ticks=ticks))
  status('completed',stage='verified')
 except Exception as e:
  status('error',stage='failed',error=repr(e));raise
if __name__=='__main__':main()
