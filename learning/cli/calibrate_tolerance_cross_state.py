#!/usr/bin/env python3
"""Batched frozen-policy cross-state pulse calibration; no optimizer."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys,json,pickle,time,argparse,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import jax
import jax.numpy as j
from sttw_control.path_command_training import load_environment
from sttw_control.path_command_policy import PathCommandActor
from sttw_control.geometric_path import build_path
from sttw_control.preference_best import atomic_json,digest
from sttw_control.runtime import configure_compilation_cache
from sttw_control.tolerance_batch_layout import contexts,lanes,case_name,FORCES,ONSETS,prefix_indices
from calibrate_directional_tolerance import BASE,PACKAGE,KEEP

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest.json').exists():raise FileExistsError('immutable run exists')
 started=time.monotonic();completed=[];ticks=0;rows=lanes();n=len(rows);budget=n*4000*len(ONSETS)
 def status(phase,**kw):atomic_json(out/'status.json',dict(phase=phase,state=phase,pid=os.getpid(),checked_unix=time.time(),elapsed_s=time.monotonic()-started,completed_cases=len(completed),declared_cases=n*len(ONSETS),active_lower_ticks=ticks,maximum_panel_lower_ticks=budget,training_steps=0,**kw))
 status('initializing')
 try:
  cfg=json.loads((BASE/'learning/configs/STTW_Path_Coverage_V2.json').read_text());source=json.loads((BASE/'docs/evidence/select300_integration_20261010/manifest.json').read_text());pkg=json.loads(PACKAGE.read_text());actors=[]
  for alpha in [0,1]:
   ident=pkg['model_selection'][f'alpha{alpha}'];assert digest(ident['actor'])==ident['actor_sha256'];assert digest(ident['checkpoint'])==ident['checkpoint_sha256']
   with open(ident['actor'],'rb') as f:actors.append(jax.tree.map(j.asarray,pickle.load(f)))
  cache=configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache');env,snapshot=load_environment(cfg,source,'terminal_contract_v2');pose=np.asarray(env.physics.helpers.pose(snapshot.data));pathbank={name:build_path(name,pose,cfg) for name in ['left90_R2','right90_R2']}
  assert pathbank['left90_R2'].xy.shape==pathbank['right90_R2'].xy.shape
  paths=jax.tree.map(lambda *xs:j.stack(xs),*[pathbank[c['route']] for c in rows]);params=jax.tree.map(lambda *xs:j.stack(xs),*[actors[c['alpha']] for c in rows]);speeds=j.asarray([c['speed'] for c in rows],j.float32);alphas=j.asarray([c['alpha'] for c in rows],j.float32);absolute_start=float(snapshot.physical_tick)*env.cc.dt
  reset=jax.jit(jax.vmap(lambda path,v,alpha:env.reset(snapshot,path,v,alpha)))
  initial=reset(paths,speeds,alphas);single=jax.tree.map(lambda x:x[0],initial);env.set_log_template(single,pathbank[rows[0]['route']],speeds[0]);actor=PathCommandActor()
  manifest=dict(protocol='directional_cross_state_v1',contexts=contexts(),lane_layout=rows,pulse_times_s=list(ONSETS),forces_N=list(FORCES),duration_s=.1,horizon_s=20.,force_direction='world horizontal lateral fixed at pulse onset heading',force_point='subtree COM equivalent wrench on chassis',terminal=cfg['stage_B_baseline'],checkpoint_identities=pkg['model_selection'],lower=source['candidate'],physical_identity=env.physics.bundle.identity,absolute_start=absolute_start,batch=n,scan=50,budget=budget,training_steps=0,preflight_declared_lower_ticks=8*n,compilation_cache=cache,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256={str(x):digest(x) for x in [Path(__file__),Path('learning/src/sttw_control/tolerance_batch_layout.py'),Path('learning/src/sttw_control/teleop_env.py'),Path('learning/src/sttw_control/path_command_env.py')]},data_split='2.3 m/s calibration; 2.8 m/s speed-transfer, no fitting on held speed',prefix='each context uses one genuine zero-lane full state at onset; duplicate pre-onset SIMD cost retained',success='common physical/domain/work/goal/final1s rules at original20s deadline')
  atomic_json(out/'manifest.json',manifest)
  for name,path in pathbank.items():np.savez_compressed(out/f'{name}_path.npz',xy=np.asarray(path.xy),s=np.asarray(path.s),goal=np.asarray(path.goal))
  def one(state,par,path,v,pulse):
   obs,_,fault=env.observation(state,path,v,20.);state=state.replace(fault=state.fault|fault);end,log=env.policy_step(state,actor.apply(par,obs),path,v,20.,pulse);return end,{k:log[k] for k in KEEP}
  advance=jax.vmap(one)
  status('running',stage='preflight')
  no_force=jax.jit(lambda states:jax.vmap(lambda st,pa,pt,v:one(st,pa,pt,v,None))(states,params,paths,speeds))
  zero_force=jax.jit(lambda states:advance(states,params,paths,speeds,j.zeros((n,5))))
  base,_=no_force(initial);zero,_=zero_force(initial);jax.block_until_ready(zero)
  errors=[float(np.max(abs(np.asarray(x,dtype=float)-np.asarray(y,dtype=float)))) for x,y in zip(jax.tree.leaves(base),jax.tree.leaves(zero)) if np.asarray(x).size];assert max(errors)==0.,max(errors)
  atomic_json(out/'preflight.json',dict(zero_force_full_state_max_error=max(errors),lower_ticks=8*n,backend='mjx',batch=n))
  def chunk(states,pulses):
   return jax.lax.scan(lambda st,_:advance(st,params,paths,speeds,pulses),states,None,length=50)
  execute=jax.jit(chunk);branch_index=j.asarray(prefix_indices())
  for onset in ONSETS:
   states=initial;pulses=j.zeros((n,5));saved=[];directions=None
   for sec in range(20):
    if sec==int(onset):
     states=jax.tree.map(lambda x:x[branch_index],states)
     saved=[{k:arr[:,np.asarray(branch_index)] for k,arr in block.items()} for block in saved]
     for ci,c in enumerate(contexts()):
      st=jax.tree.map(lambda x:x[ci*len(FORCES)],states)
      with (out/f"{c['route']}_v{c['speed']:g}_a{c['alpha']}_t{onset:g}_snapshot.pkl").open('wb') as f:pickle.dump(jax.device_get(st),f)
     headings=np.asarray(jax.vmap(lambda st:env.physics.helpers.pose(st.physical.data)[2])(states));directions=np.stack([-np.sin(headings),np.cos(headings),np.zeros(n)],axis=1);forces=np.asarray([c['force_N'] for c in rows])[:,None]*directions;pulses=j.asarray(np.column_stack([np.full(n,absolute_start+onset),np.full(n,absolute_start+onset+.1),forces]),j.float32)
    status('running',stage='panel',onset=onset,seconds=sec)
    states,logs=execute(states,pulses);jax.block_until_ready(states);host=jax.device_get(logs);ticks+=int(np.sum(host['active_tick']));saved.append(host)
    if np.any(host['policy_fault']) or np.any(host['lower_fault']):raise RuntimeError('policy/lower engineering fault')
    assert ticks<=budget
    status('running',stage='panel',onset=onset,seconds=sec+1);print(f'onset={onset:g} seconds={sec+1}/20 ticks={ticks}',flush=True)
   all_logs={k:np.concatenate([block[k] for block in saved],axis=0) for k in KEEP}
   for idx,c in enumerate(rows):
    d={k:arr[:,idx].reshape((-1,)+arr.shape[3:]) for k,arr in all_logs.items()};mask=d['active_tick'];d={k:val[mask] for k,val in d.items()};name=case_name(c,onset);np.savez_compressed(out/f'{name}.npz',**d);b=cfg['stage_B_baseline'];count=len(d['time']);tail=d['time']+.005>19.+1e-5
    physical=bool(np.any(d['physical_failure']));domain=bool(np.any(d['domain_exit']));work=bool(np.any(d['peak_roll']>b['working_roll_rad']));hold=bool(count==4000 and tail.sum()==200 and np.all(abs(d['path_cross_track'][tail])<=b['final_y_error_m']) and np.all(abs(d['path_heading_error'][tail])<=b['final_path_heading_error_rad']) and np.all(abs(d['actual_forward_speed'][tail]-c['speed'])<=b['final_speed_error_m_s']));goal=bool(np.any((d['goal_section_signed_distance']>=0)&(d['path_progress']>=float(pathbank[c['route']].goal))))
    impulse=d['applied_force_impulse'].sum(axis=0);expected=c['force_N']*.1*directions[idx];fully_applied=count>=round((onset+.1)/.005)
    if fully_applied:np.testing.assert_allclose(impulse,expected,rtol=.004,atol=1e-5)
    completed.append(dict(**c,name=name,onset_s=onset,success=not(physical or domain or work) and hold and goal,physical_failure=physical,domain_failure=domain,working_failure=work,final_hold=hold,goal=goal,ticks=count,peak_roll=float(d['peak_roll'].max()) if count else None,impulse=impulse.tolist(),expected_impulse=expected.tolist(),pulse_complete=fully_applied))
   atomic_json(out/'results.json',dict(cases=completed,scope='finite DEV panel; no continuous boundary certification'))
  status('completed',stage='saved',preflight_lower_ticks=8*n)
 except BaseException as e:status('error',error=f'{type(e).__name__}: {e}');raise
if __name__=='__main__':main()
