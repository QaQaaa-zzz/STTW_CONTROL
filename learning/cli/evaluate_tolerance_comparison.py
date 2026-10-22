#!/usr/bin/env python3
"""Matched final-checkpoint perturbation comparison; no optimization."""
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
from sttw_control.tolerance_comparison import comparison_layout,chunk_schedule,resolve_comparison_actors,paired_counts
from calibrate_directional_tolerance import KEEP

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--campaign',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();campaign=Path(a.campaign).resolve();protocol=json.loads((campaign/'comparison_protocol.json').read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if a.smoke:protocol=dict(protocol,policies=['parent'],speeds=[2.45],onsets=[.1],forces=[0,-2,2],simulation_duration_s=.24,max_lower_ticks=576,engineering_only=True)
 if (out/'manifest.json').exists():raise FileExistsError('immutable run exists')
 started=time.monotonic();completed=[];ticks=0;rows,branch_ids=comparison_layout(protocol);n=len(rows);ONSETS=protocol['onsets'];budget=protocol['max_lower_ticks']
 def status(phase,**kw):atomic_json(out/'status.json',dict(phase=phase,state=phase,pid=os.getpid(),checked_unix=time.time(),elapsed_s=time.monotonic()-started,completed_cases=len(completed),declared_cases=n*len(ONSETS),active_lower_ticks=ticks,maximum_panel_lower_ticks=budget,training_steps=0,**kw))
 status('initializing')
 try:
  if a.smoke:
   plan=json.loads((campaign/'baseline_alpha0/plan.json').read_text());package=json.loads(Path(plan['package']).read_text())
   records={('parent',alpha):package['model_selection'][f'alpha{alpha}'] for alpha in [0,1]}
   for record in records.values():
    if digest(record['actor'])!=record['actor_sha256']:raise ValueError('parent actor changed')
  else:records=resolve_comparison_actors(campaign)
  cfg=json.loads((campaign/'baseline_config.json').read_text())
  parent_plan=json.loads((campaign/'baseline_alpha0/plan.json').read_text())
  source=json.loads(Path(parent_plan['parent_manifest']).read_text())['source'];actors={}
  for key,record in records.items():
   with open(record['actor'],'rb') as f:actors[key]=jax.tree.map(j.asarray,pickle.load(f))
  cache=configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache');env,snapshot=load_environment(cfg,source,'terminal_contract_v2');pose=np.asarray(env.physics.helpers.pose(snapshot.data));pathbank={name:build_path(name,pose,cfg) for name in ['left90_R2','right90_R2']}
  assert pathbank['left90_R2'].xy.shape==pathbank['right90_R2'].xy.shape
  paths=jax.tree.map(lambda *xs:j.stack(xs),*[pathbank[c['route']] for c in rows]);params=jax.tree.map(lambda *xs:j.stack(xs),*[actors[(c['policy'],c['alpha'])] for c in rows]);speeds=j.asarray([c['speed'] for c in rows],j.float32);alphas=j.asarray([c['alpha'] for c in rows],j.float32);absolute_start=float(snapshot.physical_tick)*env.cc.dt
  reset=jax.jit(jax.vmap(lambda path,v,alpha:env.reset(snapshot,path,v,alpha)))
  initial=reset(paths,speeds,alphas);single=jax.tree.map(lambda x:x[0],initial);env.set_log_template(single,pathbank[rows[0]['route']],speeds[0]);actor=PathCommandActor()
  manifest=dict(engineering_only=a.smoke,protocol=protocol,lane_layout=rows,checkpoint_identities={f'{k[0]}_alpha{k[1]}':v for k,v in records.items()},lower=source['candidate'],physical_identity=env.physics.bundle.identity,absolute_start=absolute_start,batch=n,scan_max=50,budget=budget,training_steps=0,preflight_declared_lower_ticks=0,compilation_cache=cache,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),prefix='own-policy genuine zero-force complete state shared only within same context',success='original full-episode physical/domain/work/goal/final1s labels; descriptive metrics retained regardless of adoption thresholds')
  atomic_json(out/'manifest.json',manifest)
  for name,path in pathbank.items():np.savez_compressed(out/f'{name}_path.npz',xy=np.asarray(path.xy),s=np.asarray(path.s),goal=np.asarray(path.goal))
  def one(state,par,path,v,pulse):
   obs,_,fault=env.observation(state,path,v,20.);state=state.replace(fault=state.fault|fault);end,log=env.policy_step(state,actor.apply(par,obs),path,v,20.,pulse);return end,{k:log[k] for k in KEEP}
  advance=jax.vmap(one)
  compiled={};branch_index=j.asarray(branch_ids)
  def executor(length):
   if length not in compiled:
    def chunk(states,pulses):return jax.lax.scan(lambda st,_:advance(st,params,paths,speeds,pulses),states,None,length=length)
    compiled[length]=jax.jit(chunk)
   return compiled[length]
  for onset in ONSETS:
   states=initial;pulses=j.zeros((n,5));saved=[];directions=None
   for start_tick,length,is_branch in chunk_schedule(onset,duration=protocol.get('simulation_duration_s',20.)):
    if is_branch:
     states=jax.tree.map(lambda x:x[branch_index],states)
     saved=[{k:arr[:,np.asarray(branch_index)] for k,arr in block.items()} for block in saved]
     for idx in sorted(set(branch_ids)):
      c=rows[idx];st=jax.tree.map(lambda x:x[idx],states)
      with (out/f"{c['policy']}_{c['route']}_v{c['speed']:g}_a{c['alpha']}_t{onset:g}_snapshot.pkl").open('wb') as f:pickle.dump(jax.device_get(st),f)
     headings=np.asarray(jax.vmap(lambda st:env.physics.helpers.pose(st.physical.data)[2])(states));directions=np.stack([-np.sin(headings),np.cos(headings),np.zeros(n)],axis=1);forces=np.asarray([c['force_N'] for c in rows])[:,None]*directions;pulses=j.asarray(np.column_stack([np.full(n,absolute_start+onset),np.full(n,absolute_start+onset+.1),forces]),j.float32)
    status('running',stage='panel',onset=onset,seconds=start_tick*.02)
    states,logs=executor(length)(states,pulses);jax.block_until_ready(states);host=jax.device_get(logs);ticks+=int(np.sum(host['active_tick']));saved.append(host)
    if np.any(host['policy_fault']) or np.any(host['lower_fault']):raise RuntimeError('policy/lower engineering fault')
    assert ticks<=budget
    status('running',stage='panel',onset=onset,seconds=(start_tick+length)*.02);print(f'onset={onset:g} intervals={start_tick+length}/1000 ticks={ticks}',flush=True)
   all_logs={k:np.concatenate([block[k] for block in saved],axis=0) for k in KEEP}
   for idx,c in enumerate(rows):
    d={k:arr[:,idx].reshape((-1,)+arr.shape[3:]) for k,arr in all_logs.items()};mask=d['active_tick'];d={k:val[mask] for k,val in d.items()};name=f"{c['policy']}_{c['route']}_v{c['speed']:g}_a{c['alpha']}_t{onset:g}_f{c['force_N']:+g}";np.savez(out/f'{name}.npz',**d);b=cfg['stage_B_baseline'];count=len(d['time']);tail=d['time']+.005>19.+1e-5
    physical=bool(np.any(d['physical_failure']));domain=bool(np.any(d['domain_exit']));work=bool(np.any(d['peak_roll']>b['working_roll_rad']));hold=bool(count==4000 and tail.sum()==200 and np.all(abs(d['path_cross_track'][tail])<=b['final_y_error_m']) and np.all(abs(d['path_heading_error'][tail])<=b['final_path_heading_error_rad']) and np.all(abs(d['actual_forward_speed'][tail]-c['speed'])<=b['final_speed_error_m_s']));goal=bool(np.any((d['goal_section_signed_distance']>=0)&(d['path_progress']>=float(pathbank[c['route']].goal))))
    impulse=d['applied_force_impulse'].sum(axis=0);expected=c['force_N']*.1*directions[idx];fully_applied=count>=round((onset+.1)/.005)
    impulse_error=float(np.linalg.norm(impulse-expected));prefix=d['time']<onset-1e-5
    prefix_valid=not bool(np.any((d['physical_failure']|d['domain_exit']|(d['peak_roll']>b['working_roll_rad']))&prefix)) and count>=round(onset/.005)
    completed.append(dict(**c,name=name,onset_s=onset,success=not(physical or domain or work) and hold and goal,physical_failure=physical,domain_failure=domain,working_failure=work,final_hold=hold,goal=goal,ticks=count,peak_roll=float(d['peak_roll'].max()) if count else None,impulse=impulse.tolist(),expected_impulse=expected.tolist(),pulse_complete=fully_applied,impulse_error_Ns=impulse_error,prefix_valid=prefix_valid,speed_rmse=float(np.sqrt(np.mean((d['actual_forward_speed']-c['speed'])**2))) if count else None,cross_track_rmse=float(np.sqrt(np.mean(d['path_cross_track']**2))) if count else None,heading_rmse=float(np.sqrt(np.mean(d['path_heading_error']**2))) if count else None))
   atomic_json(out/'results.json',dict(cases=completed,scope='engineering smoke only; short traces cannot assess task recovery' if a.smoke else 'finite DEV panel; no continuous boundary certification'))
  comparisons={} if a.smoke else {f'{alpha}_{baseline}_vs_{candidate}':paired_counts(completed,baseline,candidate,alpha) for alpha in [0,1] for baseline in ['parent','baseline_update50','static_update50'] for candidate in ['baseline_update50','static_update50','dynamic_update50'] if baseline!=candidate}
  atomic_json(out/'paired_comparisons.json',comparisons)
  status('completed',stage='saved',preflight_lower_ticks=0)
 except BaseException as e:status('error',error=f'{type(e).__name__}: {e}');raise
if __name__=='__main__':main()
