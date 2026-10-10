#!/usr/bin/env python3
"""Exactly two zero-upper stage-B diagnostics, never training or retry tuning."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import argparse,json,pickle,sys,time,hashlib,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import jax
import jax.numpy as j
from dataclasses import asdict
from sttw_control.path_command_env import PathCommandEnv
from sttw_control.geometric_path import build_path
from sttw_control.frozen_local_lower import FrozenLocalLower
from sttw_control.preference_best import digest,atomic_json,config_hash
from sttw_control.direct_command_training import flatten_logs
from sttw_control.lower_command import SCALES
from sttw_control.runtime import configure_compilation_cache

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='learning/configs/STTW_Path_Feedback_V1.json');p.add_argument('--source',default='docs/evidence/select300_integration_20261010');p.add_argument('--output',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
 cfg=json.loads(Path(a.config).read_text());source=json.loads((Path(a.source)/'manifest.json').read_text());candidate=source['candidate'];out=Path(a.output)
 if out.exists():raise ValueError('immutable run directory must not exist')
 out.mkdir(parents=True);start=time.monotonic();ticks=0;completed=[]
 def status(state,**kw):atomic_json(out/'status.json',dict(state=state,pid=os.getpid(),control_ticks=ticks,maximum_control_ticks=6000,training_steps=0,completed=completed,elapsed_s=time.monotonic()-start,**kw))
 status('initializing')
 try:
  assert candidate['source_update']==cfg['scope']['lower_checkpoint']==300
  assert digest(candidate['actor'])==candidate['actor_sha256'];assert digest(candidate['checkpoint'])==candidate['checkpoint_sha256']
  assert digest(candidate['config'])==candidate['config_file_sha256']
  lowercfg=json.loads(Path(candidate['config']).read_text());np.testing.assert_array_equal(np.asarray(SCALES),np.asarray(lowercfg['observation_scales'],np.float32))
  bankpath=Path(source['prepared_bank']);assert digest(bankpath)==source['prepared_bank_sha256']
  protected_paths=[Path(candidate['checkpoint']),Path(candidate['actor']),Path(candidate['original_protocol_best']['checkpoint'])]
  for u in candidate['upper'].values():protected_paths.extend([Path(u['actor']),Path(u['checkpoint']),Path(u['actor']).parents[2]/'best_model.json'])
  protected_paths.append(Path(candidate['config']).parent/'training/best_model.json')
  protected={str(p):digest(p) for p in protected_paths if p.exists()}
  with Path(candidate['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
  upper=Path(source['upper_run']);spec=json.loads((upper/'alpha0/frozen_config_stage52.json').read_text())
  env=PathCommandEnv(cfg,spec,FrozenLocalLower(params))
  for key,live in [('model',env.physics.bundle.identity),('controller',asdict(env.cc)),('actuator',asdict(env.ac))]:assert config_hash(source[key])==config_hash(live),key
  with bankpath.open('rb') as f:bank=pickle.load(f)
  snapshot=jax.tree.map(lambda x:j.asarray(x[source['bank_index']]),bank)
  _,pose,_=env.physics.observe(snapshot.data)
  configure_compilation_cache(out/'jax_cache')
  code={str(p):digest(p) for p in sorted(Path('learning/src/sttw_control').glob('path_*.py'))}
  code['learning/src/sttw_control/geometric_path.py']=digest('learning/src/sttw_control/geometric_path.py');code[__file__]=digest(__file__)
  manifest=dict(schema=env.schema,task_mode='geometric_path',base_commit=cfg['source_commit'],head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_files_sha256=code,config=cfg,config_sha256=digest(a.config),candidate=candidate,plant_spec=spec,model=env.physics.bundle.identity,controller=asdict(env.cc),actuator=asdict(env.ac),loaded_timing=dict(physics_dt=float(env.physics.model.opt.timestep),substeps=env.physics.substeps,lower_dt=env.cc.dt,upper_dt=spec['plant']['policy_dt_s']),prepared_bank=str(bankpath),prepared_bank_sha256=digest(bankpath),bank_index=source['bank_index'],seed=source['seed'],maximum_control_ticks=6000,training_steps=0,phase_C='not_run',upper='constant zero latent; no old actor loaded',initial_pose=np.asarray(pose).tolist(),protected=protected,command=sys.argv)
  atomic_json(out/'manifest.json',manifest)
  if not a.execute:status('prepared_not_run');return
  for case in cfg['stage_B_baseline']['cases']:
   name=case['name'];horizon=case['horizon_s'];v=j.asarray(case['v_user'],dtype=j.float32)
   # Positive declared error means vehicle left of route: translate route right once.
   path=build_path(name,np.asarray(pose),cfg,lateral_offset=-case['lateral_initial_error_m'])
   np.savez_compressed(out/f'{name}_path.npz',s=np.asarray(path.s),xy=np.asarray(path.xy),heading=np.asarray(path.heading),curvature=np.asarray(path.curvature),turn_end=np.asarray(path.turn_end),goal=np.asarray(path.goal))
   s=env.reset(snapshot,path,v);jax.block_until_ready(s)
   obs,critic,fault=env.observation(s,path,v,horizon);assert obs.shape==(351,) and critic.shape==(352,) and not bool(fault)
   before=np.asarray(s.projection.progress);env.observation(s,path,v,horizon);np.testing.assert_array_equal(s.projection.progress,before)
   with (out/f'{name}_initial_state.pkl').open('wb') as f:pickle.dump(jax.device_get(s),f)
   env.set_log_template(s,path,v);status('running',case=name,stage='compile');print('compile '+name,flush=True)
   execute=jax.jit(lambda st:env.policy_step(st,j.zeros(2),path,v,horizon)).lower(s).compile()
   chunks=[];reason='horizon';planned=round(horizon/.02)
   for step in range(planned):
    s,logs=execute(s);jax.block_until_ready(s);host=jax.device_get(logs);chunks.append(host);ticks+=int(host['active_tick'].sum());assert ticks<=6000
    if (step+1)%50==0:status('running',case=name,stage='physics',policy_intervals=step+1,planned_intervals=planned);print(f'{name} {(step+1)*.02:g}/{horizon}s total_ticks={ticks}',flush=True)
    if bool(s.physical.failed)|bool(s.fault)|bool(s.domain_exit):
     reason='physical_failure' if bool(s.physical.failed) else 'policy_fault' if bool(s.fault) else 'follower_domain_exit';break
   all_logs=jax.tree.map(lambda *xs:np.concatenate(xs),*chunks);flat=flatten_logs(all_logs);valid=flat['active_tick'];flat={k:val[valid] for k,val in flat.items()}
   flat.update(initial_xy=np.asarray(pose[:2]),initial_pose=np.asarray(pose),initial_path_error=np.asarray(case['lateral_initial_error_m']),horizon_s=np.asarray(horizon),stop_reason=np.asarray(reason),lower_checkpoint_update=np.asarray(300))
   np.savez_compressed(out/f'{name}.npz',**flat)
   with (out/f'{name}_final_state.pkl').open('wb') as f:pickle.dump(jax.device_get(s),f)
   completed.append(dict(case=name,control_ticks=len(flat['time']),reason=reason));status('running',stage='saved')
  for p,sha in protected.items():assert digest(p)==sha,p
  status('completed',protected_unchanged=True,phase_C='not_run',adoption='not_adopted')
 except BaseException as e:
  status('error',error_type=type(e).__name__,message=str(e));raise
if __name__=='__main__':main()
