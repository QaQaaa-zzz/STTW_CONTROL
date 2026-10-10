#!/usr/bin/env python3
"""Bounded fixed300 diagnostic integration; no training or model registration."""
import argparse,os,sys,json,pickle,time
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import jax,jax.numpy as jp
from dataclasses import asdict
from sttw_control.frozen_local_lower import FrozenLocalLower
from sttw_control.lower_command import SCALES
from sttw_control.direct_command_env import DirectCommandEnv
from sttw_control.direct_command_policy import DirectCommandActor,q_ratio
from sttw_control.direct_command_training import flatten_logs
from sttw_control.fixed_command_panel import load_protocol,schedules
from sttw_control.preference_best import digest,atomic_json,config_hash
from sttw_control.lower_tracking_audit import held,intervals
from sttw_control.runtime import configure_compilation_cache
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--upper-run',required=True);p.add_argument('--lower-run',required=True);a=p.parse_args()
out=Path(a.output);upper=Path(a.upper_run);lower_run=Path(a.lower_run)
assert (out/'integration_candidate.json').exists() and not (out/'manifest.json').exists(),'new declared diagnostic directory required'
start=time.monotonic();completed=[];total_ticks=0;stop=False
candidate=json.loads((out/'integration_candidate.json').read_text());fixed_specs=[json.loads((upper/f'alpha{i}/frozen_config_stage52.json').read_text()) for i in (0,1)]
spec=fixed_specs[0];cfg=json.loads((lower_run/'config.json').read_text());parent_manifest=json.loads((upper/'manifest.json').read_text())
protected={candidate['checkpoint']:candidate['checkpoint_sha256'],candidate['actor']:candidate['actor_sha256'],candidate['original_protocol_best']['checkpoint']:candidate['original_protocol_best']['checkpoint_sha256']}
for u in candidate['upper'].values():protected.update({u['actor']:u['actor_sha256'],u['checkpoint']:u['checkpoint_sha256']})
for pth,sha in protected.items():assert digest(pth)==sha,pth
for key in ['action','network','plant','limits','reference','lower_controller','lower_reference_centered','commands','reward','smooth_v4']:
    assert fixed_specs[0][key]==fixed_specs[1][key],key
assert spec['lower_reference_centered'] is True
np.testing.assert_array_equal(np.asarray(SCALES),np.asarray(cfg['observation_scales'],np.float32))
assert candidate['source_update']==300
with Path(candidate['actor']).open('rb') as f:params=jax.tree.map(jp.asarray,pickle.load(f))
net=FrozenLocalLower(params);env=DirectCommandEnv(spec,lower_adapter=net)
# Substep force observation only, without the old310D internal-capture routine.
env.physics.spec={**env.physics.spec,'lower_internal_diagnostics':True}
bank_path=Path(parent_manifest['prepared_bank']);assert digest(bank_path)==parent_manifest['prepared_bank_sha256']
# Local development preparation is a different artifact; integration uses only
# the exact original upper prepared bank, not the lower training bank.
lower_training_bank_sha=digest(cfg['prepared_bank'])
with bank_path.open('rb') as f:bank=pickle.load(f)
sample=jax.tree.map(lambda x:jp.asarray(x[3]),bank)
old_identity=json.loads((lower_run/'training/R196_reference/identity.json').read_text())['identity']
for key,live in [('model',env.physics.bundle.identity),('controller',asdict(env.cc)),('actuator',asdict(env.ac))]:
    assert config_hash(old_identity[key])==config_hash(live),key
assert env.cc.dt==.005
assert float(env.physics.model.opt.timestep)==old_identity['physics_dt']
assert env.physics.substeps==round(env.cc.dt/old_identity['physics_dt'])
upper_params=[]
for i in (0,1):
    live=json.loads((upper/f'alpha{i}/best_model.json').read_text());assert live['checkpoint_sha256']==candidate['upper'][str(i)]['checkpoint_sha256']
    with Path(live['actor']).open('rb') as f:upper_params.append(jax.tree.map(jp.asarray,pickle.load(f)))
upper_params=jax.tree.map(lambda *x:jp.stack(x),*upper_params);actor=DirectCommandActor();protocol=load_protocol();rows_by_case=dict(schedules(protocol))
configure_compilation_cache(out/'jax_cache')
manifest=dict(role='diagnostic_only',candidate=candidate,upper_run=str(upper.resolve()),prepared_bank=str(bank_path),prepared_bank_sha256=digest(bank_path),bank_index=3,seed=77001,protocol=protocol,lower_training_bank_sha256=lower_training_bank_sha,upper_config_file_sha256={str(i):digest(upper/f'alpha{i}/frozen_config_stage52.json') for i in (0,1)},model=env.physics.bundle.identity,controller=asdict(env.cc),actuator=asdict(env.ac),physics_dt=float(env.physics.model.opt.timestep),substeps=env.physics.substeps,control_dt=.005,policy_dt=.02,maximum_new_control_ticks=10400,training_ticks=0,initialization='matched complete prepared physical bank; causal masks zero at start, one real pre-frame per5ms',stop_rule='physical/policy fault or quiet-settled lower speed>.20/steer>.08 continuously1s; save and stop new combinations',baseline_reuse='V5.2 evaluationbest with identical bank/protocol and upper SHA; check exact raw/reference stream after simulation')
atomic_json(out/'manifest.json',manifest)
def status(**kw):atomic_json(out/'status.json',dict(pid=os.getpid(),completed=completed,control_ticks=total_ticks,training_ticks=0,elapsed_s=time.monotonic()-start,**kw))
def store(case,chunks):
    if not chunks:return {}
    logs=jax.tree.map(lambda *x:np.concatenate(x),*chunks);flat=flatten_logs(logs);traces={};dest=out/case;dest.mkdir(exist_ok=True)
    for i in (0,1):
        d={k:v[:,i].reshape((-1,)+v.shape[3:]) for k,v in flat.items()};valid=d['active_tick'];d={k:v[valid] for k,v in d.items()}
        d.update(checkpoint_update=np.asarray(candidate['upper'][str(i)]['source_update']),lower_checkpoint_update=np.asarray(300),partial=np.asarray(False))
        np.savez_compressed(dest/f'alpha{i}.npz',**d);traces[i]=d
    return traces
try:
    status(state='initializing')
    for case,seconds in [('straight_hold',10),('fast_turn',16)]:
        status(state='running',stage='reset',case=case)
        rows=jp.asarray(rows_by_case[case],jp.float32)
        reset=jax.jit(lambda rows:jax.vmap(lambda alpha:env.reset(sample,jp.int32(77001),jp.int32(0),alpha,rows,jp.array([.5,.3])))(jp.array([0.,1.])))
        states=reset(rows);jax.block_until_ready(states)
        with (out/(case+'_initial_full_states.pkl')).open('wb') as f:pickle.dump(jax.device_get(states),f)
        one=jax.tree.map(lambda x:x[0],states);env.set_log_template(one)
        def chunk(states):
            def step(st,_):
                obs,_,fault,_=jax.vmap(env.observation)(st);z=jax.vmap(lambda p,o:actor.apply(p,o))(upper_params,obs)
                st=st.replace(fault=st.fault|fault)
                end,_,_,logs,_=jax.vmap(lambda s,z:env.policy_step(s,z,False))(st,z)
                return end,logs
            return jax.lax.scan(step,states,None,length=50)
        status(state='running',stage='compile',case=case);print('COMPILE '+case+' pair',flush=True)
        execute=jax.jit(chunk).lower(states).compile();chunks=[]
        for second in range(seconds):
            states,logs=execute(states);jax.block_until_ready(states);logs=jax.device_get(logs);chunks.append(logs)
            total_ticks+=int(np.sum(logs['active_tick']));assert total_ticks<=10400
            traces=store(case,chunks);reasons=[]
            for i,d in traces.items():
                if np.any(d['physical_failure']) or np.any(d['policy_fault']) or np.any(d['lower_fault']):reasons.append(f'{case}/alpha{i} physical/policy/lower failure')
                t=d['time'];err=np.column_stack([d['actual_forward_speed'],d['actual_delta']])-d['governed'];rates=d['lower_governed_rates']
                quiet=held((abs(rates[:,0])<=.1)&(abs(rates[:,1])<=.02),.005,.5)&(t>=1.)
                bad=quiet&((abs(err[:,0])>.2)|(abs(err[:,1])>.08))
                if any(x['duration_s']>=1.-1e-6 for x in intervals(bad,t,.005)):reasons.append(f'{case}/alpha{i} persistent pronounced lower mismatch')
            status(state='running',stage='physics',case=case,seconds_completed=second+1)
            print(f'{case} {second+1}/{seconds}s ticks={total_ticks}',flush=True)
            if reasons:
                stop=True;atomic_json(out/'stop_receipt.json',dict(reasons=reasons,case=case,seconds_saved=second+1,control_ticks=total_ticks))
                with (out/'stop_full_states.pkl').open('wb') as f:pickle.dump(jax.device_get(states),f)
                break
        if stop:break
        with (out/(case+'_final_full_states.pkl')).open('wb') as f:pickle.dump(jax.device_get(states),f)
        for i,d in traces.items():
            baseline=dict(np.load(upper/'evaluationbest'/case/f'alpha{i}.npz'))
            for key in ['time','limited_command','raw_rates','reference_xy','reference_yaw_unwrapped']:
                np.testing.assert_array_equal(d[key],baseline[key])
            completed.append(case+f'/alpha{i}')
    for pth,sha in protected.items():assert digest(pth)==sha,pth
    status(state='stopped_mismatch' if stop else 'completed',stage='saved',protected_identities_unchanged=True,adoption='not_adopted')
except BaseException as e:
    status(state='error',stage='diagnostic',error_type=type(e).__name__,message=str(e));raise
