"""Real GPU snippet smoke; no learner updates or performance claim."""
import argparse,json,sys,os,time,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import jax
import jax.numpy as j
from sttw_control.recovery_bank import load_parent
from sttw_control.path_command_training import load_environment,PathBatch
from sttw_control.path_command_policy import PathCommandActor
from sttw_control.tolerance_training import experiment_config
from sttw_control.short_recovery import ShortRecoveryBatch
from sttw_control.preference_best import atomic_json
from sttw_control.runtime import configure_compilation_cache


def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();ticks=0
 def update(phase='running',**kw):atomic_json(out/'status.json',dict(phase=phase,state=phase,pid=os.getpid(),checked_unix=time.time(),elapsed_s=time.monotonic()-started,active_lower_ticks=ticks,maximum_lower_ticks=12800,training_steps=0,**kw))
 update(stage='loading')
 try:
  bankdir=Path(a.bank);manifest=json.loads((bankdir/'manifest.json').read_text());cfg=manifest['config'];record=manifest['parent']
  configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
  env,snapshot=load_environment(cfg,manifest['source'],'terminal_contract_v2')
  with (bankdir/'bank.pkl').open('rb') as f:bank=pickle.load(f)
  with Path(record['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
  actor=PathCommandActor();results=[]
  for guided in [False,True]:
   base=PathBatch(env,snapshot,cfg,910601,0);batch=ShortRecoveryBatch(base,bank,guided=guided,seed=910602,reserve_config=experiment_config(cfg,'dynamic'))
   episodes=jax.jit(jax.vmap(batch.reset))(j.arange(8),j.zeros(8,j.int32))
   # Two explicitly nominal engineering lanes; all remaining sampled as designed.
   episodes=episodes.replace(force=episodes.force.at[:2].set(0.),force_index=episodes.force_index.at[:2].set(0),pulse=episodes.pulse.at[:2,2:].set(0.))
   if not bool(j.all(episodes.state.tick>0)):raise RuntimeError('original task clock was reset')
   sample=jax.tree.map(lambda x:x[0],episodes);env.set_log_template(sample.state,sample.route.path,sample.route.speed)
   def step(ep,_):
    ob=jax.vmap(batch.observation)(ep);z=jax.vmap(lambda o:actor.apply(params,o))(ob[0])
    nxt,obs,r,flags,final,stats=jax.vmap(batch.advance_one)(ep,z)
    return nxt,(r,flags,stats['pulse'],stats['recovery'],final[1],stats['active_ticks'])
   execute=jax.jit(lambda ep:jax.lax.scan(step,ep,None,length=10))
   timeouts=completed=0;max_error=0.;fired=0
   for start in range(0,200,10):
    episodes,logs=execute(episodes);jax.block_until_ready(episodes)
    r,flags,pulse,rec,finalcritic,active=jax.device_get(logs);ticks+=int(active.sum())
    if flags['engineering_fault'].any() or not np.isfinite(r).all():raise RuntimeError('snippet engineering fault')
    timeouts+=int(flags['segment_timeout'].sum());completed+=int(flags['done'].sum());fired+=int(pulse['fired_now'].sum())
    full=flags['done']&(rec['elapsed_ticks']>=20)
    if full.any():max_error=max(max_error,float(np.max(np.abs(pulse['completed_impulse'][full]-pulse['completed_expected_impulse'][full]))))
    update(stage='snippet',guided=guided,completed_intervals=start+10)
   if timeouts<1:raise RuntimeError('no real timeout boundary exercised')
   if fired<1:raise RuntimeError('no nonzero recovery pulse')
   if max_error>.002:raise RuntimeError('gross completed pulse impulse error')
   results.append(dict(guided=guided,segment_timeouts=timeouts,completed=completed,pulses=fired,max_impulse_error_Ns=max_error,original_clocks_preserved=True))
  atomic_json(out/'results.json',dict(results=results,scope='GPU engineering snippets, original20s horizon; no training or effect claim',active_lower_ticks=ticks));update('completed',stage='verified')
 except BaseException as exc:update('error',error=f'{type(exc).__name__}: {exc}');raise
if __name__=='__main__':main()
