#!/usr/bin/env python3
"""Bounded diagnostic panel: frozen policies, real force pulses, no optimization."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import sys,json,pickle,time,argparse,hashlib,subprocess
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

BASE=Path('/home/qy/STTW_CONTROL/runs/worktrees/path-feedback-v1')
PACKAGE=Path('/home/qy/下载/STTW_单网与双网_核心代码和结果_20261021/02_历史双网络/core/hyperparameters.json')
KEEP=['active_tick','time','actual_xy','reference_xy','actual_forward_speed','v_user','path_cross_track','path_heading_error','path_progress','goal_section_signed_distance','phi','phi_dot','actual_delta','peak_roll','physical_failure','domain_exit','policy_fault','lower_fault','governed','applied_force_impulse','nominal','final_ctrl']

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest.json').exists():raise FileExistsError('immutable execution exists')
 start=time.monotonic();completed=[];ticks=0
 def status(phase,**kw):
  atomic_json(out/'status.json',dict(phase=phase,state=phase,pid=os.getpid(),checked_unix=time.time(),elapsed_s=time.monotonic()-start,completed_cases=len(completed),declared_cases=7,active_lower_ticks=ticks,maximum_panel_lower_ticks=26600,training_steps=0,**kw))
 status('initializing')
 try:
  cfg=json.loads((BASE/'learning/configs/STTW_Path_Coverage_V2.json').read_text());src=json.loads((BASE/'docs/evidence/select300_integration_20261010/manifest.json').read_text());pkg=json.loads(PACKAGE.read_text())
  actors=[]
  for alpha in [0,1]:
   ident=pkg['model_selection'][f'alpha{alpha}'];assert digest(ident['actor'])==ident['actor_sha256'];assert digest(ident['checkpoint'])==ident['checkpoint_sha256']
   with open(ident['actor'],'rb') as f:actors.append(jax.tree.map(j.asarray,pickle.load(f)))
  cache=configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
  env,snapshot=load_environment(cfg,src,'terminal_contract_v2')
  horizon=20.;v=j.asarray(2.6,j.float32);pose=np.asarray(env.physics.helpers.pose(snapshot.data));path=build_path('left90_R2',pose,cfg)
  s=env.reset(snapshot,path,v,j.asarray(0.,j.float32));env.set_log_template(s,path,v)
  absolute_start=float(snapshot.physical_tick)*env.cc.dt
  manifest=dict(protocol='directional_boundary_focus_v1',prefix='single genuine prefix copied at onset including full history; unused SIMD prefix lanes charged as overhead',data_role='development engineering calibration only',horizon_s=horizon,parent_snapshot=str(BASE.parent/'tolerance-calibration/runs/pulse_pilot_v1_attempt2/alpha0_t1_snapshot.pkl'),parent_snapshot_sha256=digest(BASE.parent/'tolerance-calibration/runs/pulse_pilot_v1_attempt2/alpha0_t1_snapshot.pkl'),prefix_reused_ticks=1400,prediction_before_measurement_N=4.721686363220215,pulse_times_s=[1.],duration_s=.1,forces_N=[0.,-.25,-.5,-4.,-4.5,-5.,-5.5],route='left90_R2',v_user=2.6,force_direction='world horizontal lateral fixed at pulse onset heading',force_point='subtree COM equivalent wrench on chassis',deadline='original episode t=20s, final1s hold; no reset after pulse',terminal=cfg['stage_B_baseline'],checkpoint_identities=pkg['model_selection'],lower=src['candidate'],physical_identity=env.physics.bundle.identity,absolute_start=absolute_start,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),patch=subprocess.check_output(['git','diff'],text=True),source_sha256={str(x):digest(x) for x in [Path(__file__),Path('learning/src/sttw_control/teleop_env.py'),Path('learning/src/sttw_control/path_command_env.py')]},batch=7,scan=50,compilation_cache=cache,budget=26600,training_steps=0)
  atomic_json(out/'manifest.json',manifest)
  np.savez_compressed(out/'path.npz',xy=np.asarray(path.xy),s=np.asarray(path.s),curvature=np.asarray(path.curvature),goal=np.asarray(path.goal))
  actor=PathCommandActor()
  def advance(st,params,pulse):
   obs,_,fault=env.observation(st,path,v,horizon);st=st.replace(fault=st.fault|fault)
   end,log=env.policy_step(st,actor.apply(params,obs),path,v,horizon,pulse)
   return end,{k:log[k] for k in KEEP}
  status('running',stage='compile_preflight')
  # Actual MJX check from the complete prepared snapshot, including controller state.
  fn=jax.jit(advance);base,bl=fn(s,actors[0],None);zero,zl=fn(s,actors[0],j.zeros(5));jax.block_until_ready(zero)
  errors=[float(np.max(np.abs(np.asarray(x,dtype=float)-np.asarray(y,dtype=float)))) for x,y in zip(jax.tree.leaves(base),jax.tree.leaves(zero)) if np.asarray(x).size]
  maxerr=max(errors);assert maxerr==0.,('zero pulse changed MJX transition',maxerr)
  atomic_json(out/'preflight.json',dict(zero_force_full_state_max_error=maxerr,lower_ticks=8,backend='mjx',physical_identity_checked=True))
  def chunk(states,params,pulses):
   def step(st,_):return jax.vmap(advance,in_axes=(0,None,0))(st,params,pulses)
   return jax.lax.scan(step,states,None,length=50)
  execute=jax.jit(chunk)
  for alpha in [0]:
   for onset in [1.]:
    with (BASE.parent/'tolerance-calibration/runs/pulse_pilot_v1_attempt2/alpha0_t1_snapshot.pkl').open('rb') as f:state=jax.tree.map(j.asarray,pickle.load(f))
    states=jax.tree.map(lambda x:j.repeat(x[None],7,axis=0),state)
    pulses=j.zeros((7,5));saved=[];status('running',stage='panel',alpha=alpha,onset=onset,seconds=0)
    prefix=dict(np.load(BASE.parent/'tolerance-calibration/runs/pulse_pilot_v1_attempt2/alpha0_t1_F+0.npz'))
    saved=[{k:np.repeat(prefix[k][:200].reshape((50,1,4)+prefix[k].shape[1:]),7,axis=1) for k in KEEP}]
    for sec in range(1,20):
     if sec==int(onset):
      nominal=jax.tree.map(lambda x:x[0],states);heading=float(env.physics.helpers.pose(nominal.physical.data)[2]);direction=np.array([-np.sin(heading),np.cos(heading),0.])
      forces=np.array([0.,-.25,-.5,-4.,-4.5,-5.,-5.5])[:,None]*direction
      pulses=j.asarray(np.column_stack([np.full(7,absolute_start+onset),np.full(7,absolute_start+onset+.1),forces]),dtype=j.float32)
      # Branch from ONE complete real prefix; SIMD lanes need not be bitwise equal.
      # Other unperturbed lanes are diagnostic overhead, not separate prefix evidence.
      states=jax.tree.map(lambda x:j.repeat(x[None],7,axis=0),nominal)
      saved=[{k:np.repeat(val[:,0:1],7,axis=1) for k,val in block.items()} for block in saved]
      for leaf in jax.tree.leaves(states):
       np.testing.assert_array_equal(np.asarray(leaf),np.repeat(np.asarray(leaf)[0:1],7,axis=0))
      with (out/f'alpha{alpha}_t{int(onset)}_snapshot.pkl').open('wb') as f:pickle.dump(jax.device_get(nominal),f)
     states,logs=execute(states,actors[alpha],pulses);jax.block_until_ready(states);host=jax.device_get(logs);saved.append(host);ticks+=int(np.sum(host['active_tick']))
     assert ticks<=26600
     if np.any(host['policy_fault']) or np.any(host['lower_fault']):raise RuntimeError('engineering policy/lower fault')
     status('running',stage='panel',alpha=alpha,onset=onset,seconds=sec+1)
     print(f'alpha={alpha} onset={onset} seconds={sec+1}/20 ticks={ticks}',flush=True)
    all_logs={k:np.concatenate([x[k] for x in saved],axis=0) for k in KEEP}
    for idx,force in enumerate([0.,-.25,-.5,-4.,-4.5,-5.,-5.5]):
     d={k:arr[:,idx].reshape((-1,)+arr.shape[3:]) for k,arr in all_logs.items()};mask=d['active_tick'];d={k:val[mask] for k,val in d.items()}
     name=f'alpha{alpha}_t{int(onset)}_F{force:+g}';np.savez_compressed(out/f'{name}.npz',**d)
     n=len(d['time']);tail=d['time']+.005>19.+1e-5;b=cfg['stage_B_baseline']
     physical=bool(np.any(d['physical_failure']));domain=bool(np.any(d['domain_exit']));work=bool(np.any(d['peak_roll']>b['working_roll_rad']))
     hold=bool(n==4000 and tail.sum()==200 and np.all(abs(d['path_cross_track'][tail])<=b['final_y_error_m']) and np.all(abs(d['path_heading_error'][tail])<=b['final_path_heading_error_rad']) and np.all(abs(d['actual_forward_speed'][tail]-2.6)<=b['final_speed_error_m_s']))
     goal=bool(np.any((d['goal_section_signed_distance']>=0)&(d['path_progress']>=float(path.goal))))
     impulse=d['applied_force_impulse'].sum(axis=0);expected=force*.1*direction
     # Stopping before/during pulse is retained as failure; don't invent unexecuted force.
     fully_applied=n>=round((onset+.1)/.005)
     if fully_applied:np.testing.assert_allclose(impulse,expected,rtol=.004,atol=1e-5)
     row=dict(name=name,alpha=alpha,onset_s=onset,force_N=force,success=not(physical or domain or work) and hold and goal,physical_failure=physical,domain_failure=domain,working_failure=work,final_hold=hold,goal=goal,ticks=n,peak_roll=float(d['peak_roll'].max()),impulse=impulse.tolist(),expected_impulse=expected.tolist(),pulse_complete=fully_applied)
     completed.append(row)
    atomic_json(out/'results.json',dict(cases=completed,scope='finite development force grid; no continuous boundary certification'))
  status('completed',stage='saved',preflight_lower_ticks=8)
 except BaseException as e:
  status('error',stage='calibration',error=f'{type(e).__name__}: {e}');raise
if __name__=='__main__':main()
