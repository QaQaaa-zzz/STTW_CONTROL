#!/usr/bin/env python3
"""One original full-state fast_turn replay (12s maximum), diagnostic logging only."""
import argparse,os,sys,json,pickle,time,hashlib,subprocess
from pathlib import Path
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false');os.environ.setdefault('OMP_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
p=argparse.ArgumentParser();p.add_argument('--parent-run',required=True);p.add_argument('--method',choices=['B0','alpha0'],required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.parent_run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
def write(name,d): (out/name).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def flat(d,prefix=''):
    r={}
    for k,v in d.items():
        key=prefix+k
        if isinstance(v,dict):r.update(flat(v,key+'_'))
        else:r[key]=v
    return r
write('status.json',dict(state='initializing',method=a.method,pid=os.getpid(),ticks=0,budget_ticks=2400))
import numpy as np
import jax,jax.numpy as jp
from sttw_control.direct_command_env import DirectCommandEnv
from sttw_control.direct_command_policy import DirectCommandActor
from sttw_control.fixed_command_panel import load_protocol,schedules
from sttw_control.runtime import configure_compilation_cache
configure_compilation_cache(out/'jax_cache')
try:
    spec=json.loads((root/'alpha0/frozen_config_stage51.json').read_text());spec['lower_internal_diagnostics']=True
    manifest=json.loads((root/'manifest.json').read_text());bankpath=Path(manifest['prepared_bank']);assert sha(bankpath)==manifest['prepared_bank_sha256']
    env=DirectCommandEnv(spec);lower=env.lower
    meta=json.loads((Path(lower.provenance['checkpoint'])/'identity.json').read_text());env.diagnostic_mean=jp.asarray(meta['mean']);env.diagnostic_std=jp.asarray(meta['std'])
    with bankpath.open('rb') as f:bank=jax.tree.map(jp.asarray,pickle.load(f))
    sample=jax.tree.map(lambda x:x[3],bank);rows=dict(schedules(load_protocol()))['fast_turn']
    state=env.reset(sample,jp.int32(77001),jp.int32(0),jp.asarray(0.),jp.asarray(rows,dtype=jp.float32),jp.array([.5,.3]));env.set_log_template(state)
    upperpath=root/'alpha0/checkpoints/actor_0100.pkl'
    if a.method=='alpha0':
        with upperpath.open('rb') as f:params=jax.tree.map(jp.asarray,pickle.load(f))
    actor=DirectCommandActor()
    write('identity.json',dict(method=a.method,parent_run=str(root),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        lower=lower.provenance,normalizer=meta,source_task=lower.source,prepared_bank=str(bankpath),prepared_sha256=sha(bankpath),prepared_index=3,
        upper_actor=str(upperpath) if a.method=='alpha0' else None,upper_sha256=sha(upperpath) if a.method=='alpha0' else None,
        env_id=77001,episode=0,rows=rows.tolist(),slew=[.5,.3],dtype=str(state.physical.data.qpos.dtype),jax_x64=jax.config.jax_enable_x64,
        budget_ticks=2400,physics_substeps=60000,actuator_names=[env.physics.model.actuator(i).name for i in range(env.physics.model.nu)],
        actuator_ctrllimited=env.physics.model.actuator_ctrllimited.tolist(),actuator_ctrlrange=env.physics.model.actuator_ctrlrange.tolist(),
        actuator_forcelimited=env.physics.model.actuator_forcelimited.tolist(),actuator_forcerange=env.physics.model.actuator_forcerange.tolist(),
        rules='pre input at t; post physical at t+.005; full original preparation; no restarts or parameter scan; one state commit per tick'))
    write('resolved_config.json',spec);(out/'source.patch').write_bytes(subprocess.check_output(['git','diff']))
    # Preserve full reset and final states, never reconstruct them from qpos/qvel alone.
    with (out/'initial_full_state.pkl').open('wb') as f:pickle.dump(jax.device_get(state),f)
    def chunk(st):
        def step(st,_):
            obs,_,fault,_=env.observation(st)
            z=jp.zeros(2) if a.method=='B0' else actor.apply(params,obs)
            st=st.replace(fault=st.fault|(fault&~st.physical.failed))
            end,_,_,logs,_=env.policy_step(st,z,False)
            return end,logs
        return jax.lax.scan(step,st,None,length=50)
    start=time.monotonic();write('status.json',dict(state='compiling',method=a.method,pid=os.getpid(),ticks=0,budget_ticks=2400))
    fn=jax.jit(chunk).lower(state).compile();compile_s=time.monotonic()-start;chunks=[];start=time.monotonic()
    for second in range(12):
        state,logs=fn(state);jax.block_until_ready(state);chunks.append(jax.device_get(logs))
        status=dict(state='running',method=a.method,pid=os.getpid(),ticks=int(state.tick),budget_ticks=2400,compile_s=compile_s,execute_s=time.monotonic()-start,failed=bool(state.physical.failed),fault=bool(state.fault))
        write('status.json',status);print(json.dumps(status),flush=True)
        if status['failed'] or status['fault']:break
    d=flat(jax.tree.map(lambda *x:np.concatenate(x),*chunks));d={k:v.reshape((-1,)+v.shape[2:]) for k,v in d.items()};active=d['active_tick'];d={k:v[active] for k,v in d.items()}
    np.savez_compressed(out/'trace.npz',**d)
    with (out/'final_full_state.pkl').open('wb') as f:pickle.dump(jax.device_get(state),f)
    old=dict(np.load(root/f'evaluation100/fast_turn/{a.method}.npz'));n=min(len(old['time']),len(d['time']))
    comparison={k:dict(max_abs=float(np.max(np.abs(old[k][:n]-d[k][:n]))),rms=float(np.sqrt(np.mean((old[k][:n]-d[k][:n])**2)))) for k in ['time','actual_delta','actual_forward_speed','phi','actual_xy','final_command','governed']}
    write('original_overlap.json',dict(ticks=n,metrics=comparison,bitwise_all=all(x['max_abs']==0 for x in comparison.values())))
    write('status.json',{**status,'state':'completed','logged_ticks':len(d['time']),'executed_replays':1})
except BaseException as e:
    write('error.json',dict(type=type(e).__name__,message=str(e)));raise
