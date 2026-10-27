"""Selected-failure diagnostic: sensor gate until domain exit, then explicit held reference."""
from pathlib import Path
import argparse,json,pickle,os,time,hashlib

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--shape-only',action='store_true');a=p.parse_args();plan=json.loads(a.plan.read_text());root=a.plan.parent
 from sttw_control.preference_best import atomic_json
 state=dict(phase='running',pid=os.getpid(),stage='loading',declared_cases=len(plan['cases']),completed_cases=0,active_lower_ticks=0,training_steps=0)
 def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(root/'status.json',state)
 update()
 try:
  import numpy as np
  import jax
  import jax.numpy as j
  from sttw_control.path_command_training import load_environment
  from sttw_control.path_command_policy import make_path_actor,validate_actor_parameters
  from sttw_control.geometric_path import build_path
  from sttw_control.balance_probe import classify_contact_balance
  from sttw_control.reference_reserve_gate import initial_gate,modal_risk,gate_step
  from sttw_control.runtime import configure_compilation_cache
  configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
  for path,sha in plan['source_sha256'].items():
   if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError('source identity changed '+path)
  decl=json.loads(Path(plan['source_declaration']).read_text());cfg=decl['config'];env,initial=load_environment(cfg,decl['source'],'terminal_contract_v2')
  # Scope the diagnostic termination change to this environment instance.
  original=env.physics
  class ContactOnlyPhysics:
   def __getattr__(self,name):return getattr(original,name)
   def _step(self,physical,*args,**kw):
    end,log=original._step(physical,*args,**kw)
    contact=log['contact_failure']|original.helpers._contact_failure(end.data)
    end=end.replace(failed=physical.failed|log['nonfinite']|contact)
    log.update(contact_failure=contact,physical_failure=end.failed)
    return end,log
  env.physics=ContactOnlyPhysics()
  actors={}
  for name,record in plan['actors'].items():
   with Path(record['path']).open('rb') as f:actors[name]=jax.tree.map(j.asarray,pickle.load(f))
   validate_actor_parameters(actors[name],cfg)
  actor=make_path_actor(cfg);paths={};snapshot_cache={}
  pose=np.asarray(env.physics.helpers.pose(initial.data))
  for route in decl['routes']:
   paths[route['name']]=build_path(route['builder'],pose,cfg)
   old=Path(plan['path_evidence_directory'])/(route['name']+'_path.npz')
   with np.load(old) as d:
    if not np.allclose(np.asarray(paths[route['name']].xy),d['xy'],atol=1e-6,rtol=0):raise ValueError('path reconstruction mismatch')
   np.savez_compressed(root/(route['name']+'_path.npz'),xy=np.asarray(paths[route['name']].xy))
  cases=plan['cases'];n=len(cases);snapshots=[]
  for row in cases:
   if row['snapshot'] not in snapshot_cache:
    with Path(row['snapshot']).open('rb') as f:snapshot_cache[row['snapshot']]=jax.tree.map(j.asarray,pickle.load(f))
   s=snapshot_cache[row['snapshot']]
   if bool(s.physical.failed|s.fault|s.domain_exit):raise ValueError('invalid prefix')
   snapshots.append(s)
  stack=lambda values:jax.tree.map(lambda *x:j.stack(x),*values)
  states=stack(snapshots);parameters=stack([actors[r['policy']] for r in cases]);path_batch=stack([paths[r['route']] for r in cases]);speeds=j.asarray([r['requested_speed'] for r in cases]);initial_time=j.asarray([float(s.tick)*env.cc.dt for s in snapshots]);modes=j.asarray([r['gate_mode'] for r in cases]);gates=jax.vmap(lambda _:initial_gate())(modes)
  wheel_radius=float(plan['sensor_contract']['rear_wheel_radius_proxy_m'])
  if wheel_radius!=.1:raise ValueError('sensor proxy must match frozen control contract')
  from sttw_control.preference_history_probe import pulse_directions
  absolute=np.asarray([float(s.physical.physical_tick)*env.cc.dt for s in snapshots]);head=np.asarray([float(env.physics.helpers.pose(s.physical.data)[2]) for s in snapshots]);directions=pulse_directions(cases,head)
  forces=np.asarray([r['force_N'] for r in cases]);pulses=j.asarray(np.column_stack([absolute,absolute+plan['pulse_duration_s'],forces[:,None]*directions]),dtype=j.float32)
  keep=['active_tick','actual_xy','reference_xy','path_cross_track','path_heading_error','actual_forward_speed','phi','phi_dot','actual_delta','peak_roll','physical_failure','nonfinite','contact_failure','lower_fault','policy_fault','domain_exit','governed','nominal','latent_z','applied_force_impulse']
  horizon=cfg['training_future_phase_C']['episode_s'];limit=round(plan['observation_s']/env.cc.dt)
  replay_enabled=any('nominal_program_file' in row for row in cases)
  replay_programs=None
  if replay_enabled:
   if not all('nominal_program_file' in row for row in cases) or limit%4:raise ValueError('one complete publication program per case required')
   programs=[]
   for row in cases:
    file=Path(row['nominal_program_file'])
    if hashlib.sha256(file.read_bytes()).hexdigest()!=row['nominal_program_sha256']:raise ValueError('nominal program changed')
    with np.load(file) as z:program=np.asarray(z['bare_latents'])
    if program.shape!=(limit//4,2) or not np.isfinite(program).all():raise ValueError('complete finite bare-latent program required')
    programs.append(program)
   replay_programs=j.asarray(np.stack(programs))
  env.set_log_template(snapshots[0],paths[cases[0]['route']],speeds[0])
  if a.shape_only:
   actions=[]
   for i,s in enumerate(snapshots):
    obs,_,fault=env.observation(s,paths[cases[i]['route']],speeds[i],horizon)
    action=np.asarray(actor.apply(actors[cases[i]['policy']],obs))
    if bool(fault) or not np.isfinite(action).all():raise ValueError('invalid snapshot/actor observation')
    actions.append(action.tolist())
   atomic_json(root/'shape_receipt.json',dict(phase='completed',cases=n,physics_ticks=0,actions=actions,world_directions=directions.tolist(),task_start_s=np.asarray(initial_time).tolist(),physical_start_s=absolute.tolist()))
   update(phase='shape_completed',stage='zero_integration',completed_cases=0)
   return
  from sttw_control.balance_probe import make_lower_tick
  lower_tick=make_lower_tick(env,snapshots[0].physical.governor.current_reference,contact_only=True)
  def normal(s,gate,par,path,speed,pulse,t0,mode,program=None):
   obs,_,fault=env.observation(s,path,speed,horizon);raw_action=actor.apply(par,obs)
   proposal=raw_action
   if replay_enabled:
    relative_tick=s.tick-j.rint(t0/env.cc.dt).astype(j.int32)
    index=relative_tick//4
    fault=fault|(relative_tick<0)|(relative_tick%4!=0)|(index>=limit//4)
    proposal=program[j.clip(index,0,limit//4-1)]
   measurement,_,_=env.physics.observe(s.physical.data)
   roll,roll_rate,steer,steer_rate,yaw_rate,rear_omega,front_omega=measurement
   risk=modal_risk(speed_proxy=rear_omega*wheel_radius,steer=steer,roll=roll,roll_rate=roll_rate,disturbance=s.physical.controller.disturbance,controller=env.cc,actuator=env.ac)
   gate,action,gate_log=gate_step(gate,proposal,rho=j.where(risk['fault'],j.nan,risk['rho']),roll_rate=roll_rate,mode=mode,dt=cfg['timing']['upper_dt_s'])
   fault=fault|gate_log['fault']
   end,logs=env.policy_step(s.replace(fault=s.fault|fault),action,path,speed,horizon,pulse)
   data={k:logs[k] for k in keep};data.update(reference=logs['governed'],time=logs['time']-t0,fallback_active=j.zeros(4,dtype=bool),upper_active=logs['active_tick'])
   data['nonfinite']=logs['nonfinite']|logs['policy_fault']
   data['gate_active']=j.full(logs['time'].shape,gate_log['active']);data['gate_rho']=j.full(logs['time'].shape,risk['rho']);data['original_policy_latent']=j.broadcast_to(raw_action,logs['time'].shape+(2,));data['executed_policy_latent']=j.broadcast_to(action,logs['time'].shape+(2,))
   if replay_enabled:
    data['nominal_program_latent']=j.broadcast_to(proposal,logs['time'].shape+(2,))
    data['nominal_program_used']=logs['active_tick']
   return (end,gate),data
  template=jax.eval_shape(normal,snapshots[0],initial_gate(),actors[cases[0]['policy']],paths[cases[0]['route']],speeds[0],pulses[0],initial_time[0],modes[0],replay_programs[0] if replay_enabled else None)[1]
  zero_block=jax.tree.map(lambda x:j.zeros(x.shape,x.dtype),template);zero_tick=jax.tree.map(lambda x:j.zeros(x.shape[1:],x.dtype),template)
  def fallback(s,gate,pulse,t0):
   def step(st,_):
    def active(st):
     t=st.tick*env.cc.dt-t0;ref=st.physical.governor.current_reference;physical,lower,log=lower_tick(st.physical,st.lower,pulse,t,ref)
     end=st.replace(physical=physical,lower=lower,tick=st.tick+1,fault=st.fault|log['lower_fault']|log['nonfinite'],yaw_rate=(physical.yaw_unwrapped-st.physical.yaw_unwrapped)/env.cc.dt)
     data=dict(zero_tick)
     for k in ['active_tick','actual_xy','actual_forward_speed','phi','phi_dot','actual_delta','peak_roll','physical_failure','nonfinite','contact_failure','lower_fault','applied_force_impulse']:data[k]=log[k]
     data.update(gate_rho=j.asarray(j.nan,dtype=ref.dtype),original_policy_latent=j.full((2,),j.nan),executed_policy_latent=j.full((2,),j.nan),time=t,reference=ref,governed=ref,nominal=st.physical.raw,reference_xy=j.full((2,),j.nan),path_cross_track=j.asarray(j.nan,dtype=ref.dtype),path_heading_error=j.asarray(j.nan,dtype=ref.dtype),domain_exit=j.bool_(True),fallback_active=j.bool_(True),upper_active=j.bool_(False))
     return end,data
    stop=st.physical.failed|st.fault|((st.tick-j.rint(t0/env.cc.dt).astype(j.int32))>=limit)
    return jax.lax.cond(stop,lambda st:(st,zero_tick),active,st)
   end,logs=jax.lax.scan(step,s,None,length=4)
   return (end,gate),logs
  def one(s,gate,par,path,speed,pulse,t0,mode,program=None):
   def active(carry):
    st,gs=carry
    return jax.lax.cond(st.domain_exit,lambda _:fallback(st,gs,pulse,t0),lambda _:normal(st,gs,par,path,speed,pulse,t0,mode,program),None)
   stop=s.physical.failed|s.fault|((s.tick-j.rint(t0/env.cc.dt).astype(j.int32))>=limit)
   return jax.lax.cond(stop,lambda c:(c,zero_block),active,(s,gate))
  @jax.jit
  def execute(carry):
   if replay_enabled:
    return jax.lax.scan(lambda c,_:jax.vmap(one)(c[0],c[1],parameters,path_batch,speeds,pulses,initial_time,modes,replay_programs),carry,None,length=25)
   return jax.lax.scan(lambda c,_:jax.vmap(one)(c[0],c[1],parameters,path_batch,speeds,pulses,initial_time,modes),carry,None,length=25)
  atomic_json(root/'manifest.json',dict(plan=plan,termination_contract='contact_only_v1',middle_actor_active='until domain exit; then latched last-reference frozen-lower continuation',lower_frozen=True,full_history_preserved=True,path_domain_exit_is_unknown=False,fallback_contract='hold_last_governed_reference_v1; no upper restart, no physical/controller/local reset; upper history stays frozen after handover',physical_identity=env.physics.bundle.identity,reward_scope='not evaluated after handover; fallback has no task-performance claim',gate_uses_future_disturbance_information=False,gate_active='until domain exit only; then frozen held reference'))
  saved=[];update(stage='scanning')
  # At most three inactive slots at domain exit; one extra upper step suffices.
  # Pad to25-step blocks; per-case tick guards enforce the exact physical horizon.
  for block in range(int(np.ceil((limit/4+1)/25))):
   (states,gates),logs=execute((states,gates));jax.block_until_ready(states);logs=jax.device_get(logs)
   flat={k:np.swapaxes(v,1,2).reshape((100,n)+v.shape[3:]) for k,v in logs.items()};saved.append(flat);state['active_lower_ticks']+=int(flat['active_tick'].sum());update(logical_slots=(block+1)*100)
  data={k:np.concatenate([b[k] for b in saved],axis=0) for k in saved[0]};results=[]
  final_states=jax.device_get(states)
  for i,row in enumerate(cases):
   log={k:v[:,i] for k,v in data.items()};name=f'case_{i:03d}';np.savez_compressed(root/(name+'.npz'),**log);label=classify_contact_balance(log,horizon_s=plan['observation_s']);impulse=log['applied_force_impulse'].sum(0)
   with (root/(name+'_final_snapshot.pkl')).open('wb') as f:pickle.dump(jax.tree.map(lambda x:x[i],final_states),f)
   final_fault=bool(np.asarray(states.fault)[i])
   if final_fault:label['balance_label']='ENGINEERING_ERROR'
   results.append(dict(**row,name=name,**label,final_fault=final_fault,fallback_triggered=bool(log['fallback_active'].any()),fallback_start_time_s=float(log['time'][np.flatnonzero(log['fallback_active'])[0]]) if log['fallback_active'].any() else None,domain_exit_observed=bool(log['domain_exit'].any()),policy_fault_observed=bool(log['policy_fault'].any()),impulse_Ns=impulse.tolist(),expected_impulse_Ns=(row['force_N']*directions[i]*plan['pulse_duration_s']).tolist()))
  atomic_json(root/'results.json',dict(cases=results));faults=[r for r in results if r['balance_label']=='ENGINEERING_ERROR'];update(phase='error' if faults else 'completed',stage='labels_ready',completed_cases=n,engineering_error_cases=len(faults))
  if faults:raise RuntimeError('engineering errors saved separately; no negative fall labels')
 except BaseException as e:update(phase='error',error=repr(e));raise
if __name__=='__main__':main()
