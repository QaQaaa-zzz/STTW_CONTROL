"""Real frozen-model shape check only; never executes a forecast or physics."""
import argparse
import json
import pickle
import time
import traceback
from pathlib import Path
from sttw_control.preference_best import atomic_json,digest


def main(source,out,execute_preview=False,binding=None):
    source=source.resolve();out=out.resolve()
    if execute_preview:
        if binding is None:raise ValueError('explicit predeclared numerical binding required')
        declared=json.loads(binding.read_text())
        if declared['source_root']!=str(source) or declared['output']!=str(out) or declared['max_new_lower_ticks']!=1000:raise ValueError('numerical contract differs')
        for file,sha in declared['file_hashes'].items():
            if digest(file)!=sha:raise ValueError('changed numerical input/source '+file)
    out.mkdir(exist_ok=False)
    atomic_json(out/'contract.json',dict(stage=('Five candidate held-proposal numerical preview' if execute_preview else 'Real complete-state / actual five-lane abstract shape only'),
        preview_upper_steps=50,initial_physics_steps=0,max_new_lower_ticks=(1000 if execute_preview else 0),optimizer_updates=0,
        no_event_query=True,no_governor_adoption=True,no_forecast_accuracy_claim=True,
        source_scope='First existing prefix only; loaded source geometry/actor/lower and complete carry',
        future='Zero external force prediction, not known cessation of real force'))
    atomic_json(out/'status.json',dict(phase='preparing',physics_steps=0,optimizer_updates=0))
    started=time.monotonic()
    try:
        import jax
        import jax.numpy as j
        import numpy as np
        from sttw_control.contact_only_environment import load_environment_contact_only
        from sttw_control.goal_preview_baseline import GoalPreviewEnv
        from sttw_control.path_command_policy import make_path_actor
        from sttw_control.geometric_path import Path as GeometricPath
        from sttw_control.path_reference_governor import candidates,make_path_preview,forecast_rows,choose
        from sttw_control.finite_value_evaluation_run import authenticate_models
        plan=json.loads((source/'plan.json').read_text())
        if digest(source/'plan.json')!=json.loads((source/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('changed original plan')
        prefixes=json.loads((source/'prefix_inventory.json').read_text())['cases'];case=prefixes[0]
        context=plan['contexts'][case['context']]
        for key in ['initial','path']:
            if digest(context[key])!=context[key+'_sha256']:raise ValueError('changed original task input')
        cfg=json.loads((source/'common_environment_config.json').read_text());decl=json.loads((source/'source_declaration.json').read_text())
        authenticate_models(cfg,dict(free_GAE=plan['model']))
        env,_=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2');env=GoalPreviewEnv(env)
        file=source/'physical/common_fork_carry.pkl'
        with file.open('rb') as f:bank=pickle.load(f)
        carry=jax.tree.map(lambda x:j.asarray(x[0]),bank)
        if int(carry.state.tick)!=case['fork_task_tick'] or bool(carry.state.physical.failed|carry.state.fault):raise ValueError('invalid saved engineering origin')
        before_clock=(int(carry.state.tick),int(carry.state.physical.physical_tick))
        with np.load(context['path']) as z:path=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
        speed=j.float32(case['requested_speed_m_s']);env.set_log_template(carry.state,path,speed)
        with Path(plan['model']['actor']).open('rb') as f:native=jax.tree.map(j.asarray,pickle.load(f))
        obs,_,_=env.observation(carry.state,path,speed,10.)
        proposed=make_path_actor(cfg).apply(native,obs)
        # Initialization is explicit, not a certified safe backup or old governor.
        grid=candidates(np.zeros(2,np.float32),np.asarray(proposed),cfg)
        one=make_path_preview(env,feedback_spec=plan['feedback_spec'],collect_trace=execute_preview)
        shapes=jax.eval_shape(jax.vmap(lambda z:one(carry,z,path,speed)[1]),j.asarray(grid.latents))
        if any(x.shape!=(5,) for x in shapes.values()):raise ValueError('actual five-lane output shape differs')
        if before_clock!=(int(carry.state.tick),int(carry.state.physical.physical_tick)):raise ValueError('shape check changed real carry')
        inputs={str(source/name):digest(source/name) for name in ['plan.json','common_environment_config.json','source_declaration.json','prefix_inventory.json','physical/common_fork_carry.pkl']}
        numerical={}
        if execute_preview:
            if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('actual GPU required for numerical cost check')
            batch=jax.jit(jax.vmap(lambda z:one(carry,z,path,speed)[1:]))
            atomic_json(out/'status.json',dict(phase='compiling',physics_steps=0,allocated_lower_slots=0,optimizer_updates=0))
            compile_start=time.monotonic();executable=batch.lower(j.asarray(grid.latents)).compile();compile_s=time.monotonic()-compile_start
            atomic_json(out/'status.json',dict(phase='executing_preview',physics_steps=None,allocated_lower_slots=1000,optimizer_updates=0))
            run_start=time.monotonic();summary,trace=executable(j.asarray(grid.latents));jax.block_until_ready(summary);execution_s=time.monotonic()-run_start
            summary=jax.device_get(summary);trace=jax.device_get(trace);forecasts=forecast_rows(summary);decision=choose(forecasts)
            observed=int(np.asarray(summary['observed_lower_ticks']).sum())
            if observed!=int(np.asarray(trace['active_tick']).sum()) or observed>1000:raise ValueError('preview cost count mismatch')
            np.savez_compressed(out/'preview_traces.npz',**trace)
            atomic_json(out/'forecasts.json',forecasts);atomic_json(out/'decision.json',vars(decision))
            numerical=dict(actual_lower_ticks=observed,allocated_lower_slots=1000,compile_s=compile_s,execution_s=execution_s,
                forecast_status=[r['status'] for r in forecasts],decision=vars(decision),governor_control_episodes=0,
                trace_source='Simulated candidate forecasts, not executed closed-loop governor trajectories')
            atomic_json(out/'numerical.json',numerical)
            atomic_json(out/'status.json',dict(phase='numerical_returned',physics_steps=observed,allocated_lower_slots=1000,optimizer_updates=0))
            if decision.status=='prediction_error':raise FloatingPointError('unknown engineering forecast; no negative control relabel')
        atomic_json(out/'receipt.json',dict(phase='completed',physics_steps=numerical.get('actual_lower_ticks',0),optimizer_updates=0,preview_results_exist=execute_preview,
            source_prefix_index=0,source_case_id=case['case_id'],original_task_and_physical_ticks=before_clock,
            shapes={k:dict(shape=x.shape,dtype=str(x.dtype)) for k,x in shapes.items()},
            real_candidate_lanes=5,padding=0,nominal_future_force='zero; no real event supplied',
            input_hashes=inputs,preview_module_sha256=digest(Path('learning/src/sttw_control/path_reference_governor.py')),
            lower_actor_sha256=decl['source']['candidate']['actor_sha256'],middle_actor_sha256=plan['model']['actor_sha256'],
            elapsed_s=time.monotonic()-started,execution=('GPU numerical preview only; not online governor validation' if execute_preview else 'CPU real-model abstract trace; not GPU integration/performance validation'),numerical=numerical))
        atomic_json(out/'status.json',dict(phase='completed',physics_steps=numerical.get('actual_lower_ticks',0),allocated_lower_slots=numerical.get('allocated_lower_slots',0),optimizer_updates=0))
        print(json.dumps(dict(phase='completed',physics_steps=numerical.get('actual_lower_ticks',0),shape={k:x.shape for k,x in shapes.items()})))
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());prior=json.loads((out/'status.json').read_text());atomic_json(out/'error.json',dict(error=repr(exc),last_status=prior));atomic_json(out/'status.json',dict(prior,phase='error',error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--execute-preview',action='store_true');p.add_argument('--binding',type=Path);args=p.parse_args()
    try:main(args.source,args.output,args.execute_preview,args.binding)
    except BaseException:
        if args.output.exists():
            from engineer_reference_bias import notify
            notify(args.output,'STTW预测基线工程检查报错','原错误和计费记录已保留；不自动重试，不把工程错误记为控制失败。')
        raise
    else:
        if args.execute_preview:
            from engineer_reference_bias import notify
            notify(args.output,'STTW五候选预测工程检查完成','固定五候选前向计算结束，实际步数和耗时已记录。不是在线治理控制结果，不扩展场景。')
