"""Trace the actual frozen Actor/environment and bias step; integrate zero ticks."""
import argparse
import json
import os
import pickle
from pathlib import Path

import jax
import jax.numpy as j
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def main(root):
    root=root.resolve();out=root/'shape';out.mkdir(exist_ok=False)
    from prepare_short_recovery_campaign import register
    register(root/'ACTIVE_SHAPE.json','STTW参考偏置真实模型零积分shape',out/'status.json')
    atomic_json(out/'status.json',dict(phase='running',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    try:
        p=json.loads((root/'plan.json').read_text())
        if digest(root/'plan.json')!=json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('changed plan')
        for file,sha in {**p['sealed_files'],**p['source_sha256']}.items():
            if digest(file)!=sha:raise ValueError('changed sealed input/source '+file)
        from sttw_control.contact_only_environment import load_environment_contact_only
        from sttw_control.goal_preview_baseline import GoalPreviewEnv
        from sttw_control.reference_bias_optimization import make_bias_step
        from sttw_control.path_command_policy import make_path_actor
        from sttw_control.geometric_path import Path as GeometricPath
        from sttw_control.finite_value_evaluation import rebind_path_context
        from sttw_control.causal_reference_baseline import initial_gate
        from sttw_control.causal_recovery_relay import CausalRelayState,initial_gate as initial_relay
        cfg=json.loads((root/'common_environment_config.json').read_text());decl=json.loads((root/'source_declaration.json').read_text())
        env,_=load_environment_contact_only(cfg,decl['source'],'terminal_contract_v2');env=GoalPreviewEnv(env)
        name=next(iter(p['contexts']));ctx=p['contexts'][name]
        with Path(ctx['initial']).open('rb') as f:bank=pickle.load(f).state
        original=jax.tree.map(lambda x:j.asarray(x[0]),bank)
        if int(original.tick)!=0 or int(original.physical.physical_tick)!=ctx['initial_physical_tick']:raise ValueError('original complete clock required')
        with np.load(ctx['path']) as z:path=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
        state=rebind_path_context(original,path,j.float32(ctx['speed_m_s']),env).replace(alpha=j.float32(.5))
        carry=CausalRelayState(state,initial_gate(),j.int32(0),initial_relay())
        with Path(p['model']['actor']).open('rb') as f:native=jax.tree.map(j.asarray,pickle.load(f))
        parameters=dict(actor=dict(native=native,keep_channels=j.array([True,True])),feedback_gain_rad=j.float32(.1),
                        bias_positions=j.asarray(ctx['bias_positions_m']),bias_targets=j.zeros((5,2),j.float32))
        speed=j.float32(ctx['speed_m_s']);env.set_log_template(state,path,speed)
        one=make_bias_step(env,make_path_actor(cfg),horizon_s=10.,spec=p['feedback_spec'])
        repeat=lambda tree:jax.tree.map(lambda x:j.repeat(x[None],16,axis=0),tree)
        shaped=jax.eval_shape(jax.vmap(one,in_axes=(0,0,None,0,0,0)),repeat(carry),repeat(parameters),path,
                              j.full(16,speed),j.zeros((16,5),j.float32),j.ones(16,j.int32))
        logs=shaped[1]
        for field in ['original_actor_latent','coordinated_actor_latent','reference_bias_target']:
            if logs[field].shape!=(16,4,2):raise ValueError('unexpected bias log shape')
        atomic_json(out/'receipt.json',dict(physics_steps=0,optimizer_updates=0,batch=16,context=name,alpha=.5,
            source_plan_sha256=digest(root/'plan.json'),shape_script_sha256=digest(__file__),
            scope='Actual current frozen Actor/complete source state/environment JAX trace only, one context layout. Not compilation/timing/control improvement, all-context rollout or replay equivalence.',
            logs={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in logs.items()}))
        atomic_json(out/'status.json',dict(phase='completed',pid=os.getpid(),physics_steps=0,optimizer_updates=0))
    except BaseException as exc:
        import traceback
        (out/'traceback.txt').write_text(traceback.format_exc());atomic_json(out/'status.json',dict(phase='error',pid=os.getpid(),error=repr(exc)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
