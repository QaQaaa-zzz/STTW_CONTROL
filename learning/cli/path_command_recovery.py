#!/usr/bin/env python3
"""Read-only pending by default. Capture and resume require --execute.

Request JSON: config (resolved design), source (integration manifest), and for
resume snapshots plus arms control0/candidate0/control1/candidate1. Each arm
specifies completed run directory and independent endpoint alpha. Its immutable
best_safety_task_model receipt and frozen run manifest are verified, never last.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('pending','capture','resume'),default='pending')
    parser.add_argument('--request',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.mode=='pending' or not args.execute:
        print(json.dumps(dict(state='pending_not_run',mode=args.mode,training_steps=0,physical_ticks=0,
            reason='capture/resume require explicit mode, request, output and --execute')))
        return
    if args.request is None or args.output is None:parser.error('explicit execution requires --request and --output')
    import jax
    import jax.numpy as j
    from sttw_control.path_command_env import PathCommandEnv
    from sttw_control.path_command_evaluation import BatchedPanelEvaluator
    from sttw_control.path_command_policy import PathCommandActor
    from sttw_control.frozen_local_lower import FrozenLocalLower
    from sttw_control.path_command_recovery import make_identity,capture_natural_snapshots,evaluate_pairs,validate_recovery_arm
    from sttw_control.preference_best import atomic_json
    from sttw_control.path_command_policy import config_key
    from sttw_control.path_command_fork import physical_config_identity
    from sttw_control.runtime import configure_compilation_cache
    request=json.loads(args.request.read_text());cfg=json.loads(Path(request['config']).read_text())
    source=json.loads(Path(request['source']).read_text());candidate=source['candidate']
    if candidate['source_update']!=300 or candidate['actor_sha256']!='d408a94e886fa7f6309aa2c7635eb62e273c197405293ac9c2f1859014f25bcc':raise ValueError('frozen local300 identity mismatch')
    with Path(candidate['actor']).open('rb') as f:lower_params=jax.tree.map(j.asarray,pickle.load(f))
    spec=json.loads((Path(source['upper_run'])/'alpha0/frozen_config_stage52.json').read_text())
    with Path(source['prepared_bank']).open('rb') as f:bank=pickle.load(f)
    snapshot=jax.tree.map(lambda x:j.asarray(x[source['bank_index']]),bank)
    env=PathCommandEnv(cfg,spec,FrozenLocalLower(lower_params),terminal_contract='terminal_contract_v2')
    identity=make_identity(env,source);ticks=0
    def account(n):
        nonlocal ticks
        ticks+=n
        if ticks>48000:raise RuntimeError('recovery prefix plus ten tails exceeds 48000 lower ticks')
    evaluator=BatchedPanelEvaluator(env,snapshot,cfg,account=account)
    configure_compilation_cache(Path('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache'))
    zero=PathCommandActor().init(jax.random.PRNGKey(0),j.zeros(351))
    arms=None;prefix_charged=0
    if args.mode=='resume':
        capture=json.loads((Path(request['snapshots'])/'capture.json').read_text())
        if not capture.get('completed') or capture.get('identity')!=identity:raise ValueError('completed compatible capture receipt required')
        prefix_charged=capture['active_prefix_lower_ticks']
        if not isinstance(prefix_charged,int) or not 0<=prefix_charged<=8000:raise ValueError('capture budget receipt mismatch')
        ticks=prefix_charged
        if set(request['arms'])!={'control0','candidate0','control1','candidate1'}:raise ValueError('four final selected arms required')
        arms={'B0':dict(params=zero,actor_sha256=None,alpha=0)}
        arm_runners={config_key(cfg):evaluator}
        for name,specification in request['arms'].items():
            root=Path(specification['run']);alpha=specification['alpha']
            run_status=json.loads((root/'status.json').read_text())
            training_status=json.loads((root/f'alpha{alpha}'/'training_status.json').read_text())
            manifest=json.loads((root/'manifest.json').read_text())
            final_source=json.loads((root/f'alpha{alpha}'/'final_best/source.json').read_text())
            selected=json.loads((root/f'alpha{alpha}'/'best_safety_task_model.json').read_text())
            validate_recovery_arm(name=name,alpha=alpha,manifest=manifest,run_status=run_status,training_status=training_status,
                selected=selected,final_source=final_source,source=source,lower_sha256=identity['lower'])
            armcfg=manifest['config']
            if physical_config_identity(armcfg)!=identity['physical_config']:raise ValueError('selected arm physical identity mismatch')
            if selected.get('config_key')!=config_key(armcfg):raise ValueError('selected arm config identity mismatch')
            cfg_identity=config_key(armcfg)
            if cfg_identity not in arm_runners:
                arm_env=PathCommandEnv(armcfg,env.plant_spec,FrozenLocalLower(lower_params),terminal_contract='terminal_contract_v2')
                if make_identity(arm_env,source)!=identity:raise ValueError('arm lower/model/controller/actuator identity mismatch')
                runner=BatchedPanelEvaluator(arm_env,snapshot,armcfg,account=account)
                runner.compiled=evaluator.compiled
                arm_runners[cfg_identity]=runner
            with Path(selected['actor']).open('rb') as f:params=jax.tree.map(j.asarray,pickle.load(f))
            arms[name]=dict(params=params,alpha=alpha,actor_sha256=selected['actor_sha256'],state='completed',selected=True,
                evaluator=arm_runners[cfg_identity])
    if args.output.exists():raise FileExistsError('immutable execution output exists')
    try:
        if args.mode=='capture':result=capture_natural_snapshots(evaluator,zero,args.output,identity)
        else:
            from sttw_control.geometric_path import build_path
            pose=env.physics.helpers.pose(snapshot.data)
            template=env.reset(snapshot,build_path('left90_R2',pose,cfg),j.array(2.6),j.array(0.))
            result=evaluate_pairs(evaluator,request['snapshots'],template,identity,arms,args.output)
        atomic_json(args.output/'status.json',dict(state='completed',mode=args.mode,active_lower_ticks=ticks-prefix_charged,cumulative_prefix_and_tail_ticks=ticks,training_steps=0))
        print(json.dumps(dict(state='completed',mode=args.mode,active_lower_ticks=ticks,output=str(args.output))))
    except BaseException as exc:
        if args.output.exists():atomic_json(args.output/'status.json',dict(state='error',mode=args.mode,active_lower_ticks=ticks-prefix_charged,cumulative_prefix_and_tail_ticks=ticks,training_steps=0,error=str(exc)))
        raise

if __name__=='__main__':main()
