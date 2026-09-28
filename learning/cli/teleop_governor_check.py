#!/usr/bin/env python3
"""Preparation and closed-loop timing audit; no neural policies or rewards."""
import argparse
from pathlib import Path
import json,time,pickle,hashlib,platform,subprocess
import jax
import jax.numpy as jp
import numpy as np
from sttw_control.teleop_commands import load_teleop_config
from sttw_control.teleop_env import TeleopEnv
from sttw_control.teleop_budget import Budget

def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def prepare(env,budget,out):
    t=time.perf_counter();s=env.initial();jax.block_until_ready(s)
    load_s=time.perf_counter()-t
    budget.charge(plant=700,reason='shared 3.5s preparation')
    t=time.perf_counter();logs=[]
    for k in range(700):
        s,l=env.step(s,s.raw,True);jax.block_until_ready(s)
        if k==0:first_step_s=time.perf_counter()-t
        logs.append(jax.device_get(l))
        if bool(s.failed):break
    np.savez_compressed(out/'preparation.npz',**{k:np.asarray([l[k] for l in logs]) for k in logs[0]})
    last=logs[-1]
    metrics=dict(ticks=len(logs),phi=float(last['phi']),phi_dot=float(last['phi_dot']),
        speed=float(last['actual_forward_speed']),delta=float(last['actual_delta']),failed=bool(s.failed),
        load_and_initial_forward_s=load_s,first_step_compile_and_execute_s=first_step_s,
        preparation_wall_s=time.perf_counter()-t)
    metrics['passed']=len(logs)==700 and not metrics['failed'] and abs(metrics['phi'])<=.05 and abs(metrics['phi_dot'])<=.10 and abs(metrics['speed']-2.3)<=.10 and abs(metrics['delta'])<=.03
    # Reanchor only once, task starts here; preserve ESO/physical tick/actuator.
    pose=env.helpers.pose(s.data)
    s=s.replace(reference_pose=pose,yaw_unwrapped=pose[2],yaw_wrapped=pose[2])
    with (out/'prepared_snapshot.pkl').open('wb') as f:pickle.dump(jax.device_get(s),f)
    write(out/'preparation.json',metrics)
    return s,metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['contracts'],default='contracts');p.add_argument('--predictor-budget',type=int,default=80000000)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--cases',nargs='*');p.add_argument('--methods',nargs='*')
    p.add_argument('--resume-from',type=Path);p.add_argument('--float64',action='store_true')
    a=p.parse_args();c=load_teleop_config(a.config)
    if a.dry_run:print(json.dumps({'valid':True,'physical_ticks':0,'phase':'contracts'}));return
    if a.float64:jax.config.update('jax_enable_x64',True)
    out=Path(a.output);root=a.resume_from if a.resume_from else out.parent
    if not out.resolve().is_relative_to(root.resolve()):raise ValueError('attempt must remain within the same run')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'physical_contracts.json').exists():raise RuntimeError('existing contract evidence; refusing overwrite')
    budget=Budget(root,a.predictor_budget)
    t=time.perf_counter();env=TeleopEnv(c);build_s=time.perf_counter()-t
    manifest=dict(base_commit=c['base']['commit'],commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),xml=env.bundle.identity,
        jax=jax.__version__,devices=[str(x) for x in jax.devices()],platform=platform.platform(),model_load_s=build_s,
        jax_enable_x64=jax.config.x64_enabled,model_dtype=str(env.helpers.mjx_model.body_mass.dtype))
    if a.resume_from:
        if json.loads((root/'frozen_config.json').read_text())!=c:raise ValueError('frozen config mismatch')
        source=root/'contracts/prepared_snapshot.pkl'
        with source.open('rb') as f:s=pickle.load(f)
        if a.float64:
            from sttw_control.teleop_precision import promote_snapshot
            s=promote_snapshot(s)
        manifest.update(source_snapshot=str(source),source_snapshot_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            migration='complete state values preserved; float leaves64/contact geom indices64' if a.float64 else 'none')
        with (out/'prepared_snapshot.pkl').open('wb') as f:pickle.dump(jax.device_get(s),f)
        metrics=json.loads((root/'contracts/preparation.json').read_text())
    else:
        write(root/'frozen_config.json',c);s,metrics=prepare(env,budget,out)
    manifest['snapshot_dtype']=str(s.data.qpos.dtype);write(out/'source_manifest.json',manifest)
    if metrics['passed']:
        from sttw_control.teleop_contract_checks import production_batch_contracts
        checks=production_batch_contracts(env,s,budget,out)
        write(out/'status.json',dict(status='prepanel_contracts_passed' if checks['passed'] else 'timing_or_state_mismatch',contracts_complete=checks['passed']))
        if checks['passed']:
            write(root/'contracts/current_contracts.json',dict(result=str((out/'physical_contracts.json').relative_to(root)),
                snapshot=str((out/'prepared_snapshot.pkl').relative_to(root)),source_manifest=str((out/'source_manifest.json').relative_to(root)),float64=a.float64))
        print(json.dumps(checks,indent=2),flush=True)
    else:write(out/'status.json',dict(status='baseline_not_ready',contracts_complete=False))

if __name__=='__main__':main()
