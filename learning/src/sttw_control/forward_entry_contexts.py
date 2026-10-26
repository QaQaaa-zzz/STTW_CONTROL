"""Real forward prefixes for entry-history transfer, not independent roots."""
import copy
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np


def prefix_inventory():
    return [dict(name=f'v{speed:g}_yaw{angle:+g}', speed_m_s=speed,
                 turn_angle_deg=angle, lower_ticks=320)
            for speed in (1.7, 2.1) for angle in (-12., 0., 12.)]


def prefix_config(config, spec):
    result=copy.deepcopy(config)
    result['path'].update(straight_prefix_m=.3, turn_radius_m=4.,
                          turn_angle_rad=float(np.deg2rad(abs(spec['turn_angle_deg']))),
                          curvature_transition_length_m=.2, straight_tail_m=75.)
    # build_path's straight is80m. Give every curved prefix the same grid
    # length without resampling curvature or changing the relevant geometry.
    c=result['path']
    plateau=c['turn_angle_rad']*c['turn_radius_m']-c['curvature_transition_length_m']
    end=c['straight_prefix_m']+2*c['curvature_transition_length_m']+plateau
    c['straight_tail_m']=80.-end
    return result


def task_relative_state(state):
    """Change task time origin only; preserve every dynamic carry by identity.

    Raw saved prefix states retain their original task clock. This helper is
    for a later, explicitly declared new task, not a physical/lower reset.
    """
    import jax.numpy as j
    return state.replace(tick=j.int32(0))


def collect(root, source, *, shape_only=False):
    import jax
    import jax.numpy as j
    from .contact_only_environment import load_environment_contact_only
    from .geometric_path import build_path
    from .preference_best import atomic_json, digest
    from .direct_command_training import flatten_logs
    from prepare_short_recovery_campaign import register

    root=Path(root).resolve(); source=Path(source).resolve()
    protocol=json.loads((root/'protocol.json').read_text())
    if protocol['prefixes']!=prefix_inventory() or protocol['source_run']!=str(source):
        raise ValueError('sealed six-prefix contract differs')
    if digest(source/'plan.json')!=protocol['source_plan_sha256']:
        raise ValueError('source plan changed')
    for name, expected in protocol['sealed_inputs'].items():
        if digest(name)!=expected:raise ValueError('sealed collection input changed: '+name)
    plan=json.loads((source/'plan.json').read_text())
    if plan['experiment']!='finite_goal_preview_baseline_v1':
        raise ValueError('frozen goal comparison source required')
    cfg=json.loads((source/'common_environment_config.json').read_text())
    declaration=json.loads((source/'source_declaration.json').read_text())
    if declaration['source']['bank_index']!=3:
        raise ValueError('original central source index3 required')
    out=root/('shape' if shape_only else 'collection');out.mkdir(exist_ok=False)
    status=dict(phase='running',pid=os.getpid(),actual_lower_ticks=0,
                allocated_lower_slots=0,optimizer_updates=0)
    def update(**kw):
        status.update(kw,checked_unix=time.time());atomic_json(out/'status.json',status)
    update();register(out/'ACTIVE_RUN.json','STTW真实前向进入历史采集',out/'status.json')
    try:
        env,snapshot=load_environment_contact_only(cfg,declaration['source'],'terminal_contract_v2')
        if int(snapshot.physical_tick)!=700 or bool(snapshot.failed):
            raise ValueError('original real prepared source required')
        pose=env.physics.helpers.pose(snapshot.data)
        states=[];paths=[]
        for spec in protocol['prefixes']:
            local_cfg=prefix_config(cfg,spec)
            kind='straight' if spec['turn_angle_deg']==0 else ('left90_R2' if spec['turn_angle_deg']>0 else 'right90_R2')
            path=build_path(kind,np.asarray(pose),local_cfg)
            paths.append(jax.tree.map(j.asarray,path))
            states.append(env.reset(snapshot,paths[-1],spec['speed_m_s'],alpha=0.))
        stack=lambda xs:jax.tree.map(lambda *xs:j.stack(xs),*xs)
        states=stack(states);paths=stack(paths)
        speeds=j.asarray([s['speed_m_s'] for s in protocol['prefixes']])
        env.set_log_template(jax.tree.map(lambda a:a[0],states),jax.tree.map(lambda a:a[0],paths),speeds[0])
        def advance(s,path,speed):
            end,logs=env.policy_step(s,j.zeros(2),path,speed,1.6)
            return end,logs
        @jax.jit
        def chunk(states):
            return jax.lax.scan(lambda st,_:jax.vmap(advance)(st,paths,speeds),states,None,length=20)
        if shape_only:
            result=jax.eval_shape(chunk,states)
            if result[1]['active_tick'].shape!=(20,6,4):raise ValueError('unexpected prefix layout')
            atomic_json(out/'receipt.json',dict(shape=[20,6,4],physics_steps=0,optimizer_updates=0))
            update(phase='completed');return
        if json.loads((root/'shape/status.json').read_text())['phase']!='completed':
            raise ValueError('completed GPU shape required before collection')
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('GPU required')
        with (out/'initial_states.pkl').open('wb') as f:pickle.dump(jax.device_get(states),f)
        arrays=[];started=time.monotonic()
        for block in range(4):
            states,logs=chunk(states);jax.block_until_ready(states)
            flat=flatten_logs(jax.device_get(logs));arrays.append(flat)
            status['actual_lower_ticks']+=int(flat['active_tick'].sum())
            status['allocated_lower_slots']+=480
            update(stage='collecting',completed_blocks=block+1,elapsed_s=time.monotonic()-started)
        host=jax.device_get(states)
        with (out/'full_states.pkl').open('wb') as f:pickle.dump(host,f)
        merged={k:np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
        np.savez_compressed(out/'prefix_logs.npz',**merged)
        records=[]
        for index,spec in enumerate(protocol['prefixes']):
            s=jax.tree.map(lambda a:a[index],states)
            measured,pose,speed=env.physics.observe(s.physical.data)
            valid=not bool(s.physical.failed|s.fault|s.domain_exit) and int(s.tick)==320 and int(s.physical.physical_tick)==1020
            records.append(dict(spec,valid=valid,task_tick=int(s.tick),physical_tick=int(s.physical.physical_tick),
                pose=np.asarray(pose).tolist(),actual_speed_m_s=float(speed),roll_rad=float(measured[0]),
                domain_exit=bool(s.domain_exit),fault=bool(s.fault),physical_failed=bool(s.physical.failed)))
        atomic_json(out/'receipt.json',dict(contexts=records,source_run=str(source),
            protocol_sha256=digest(root/'protocol.json'),full_states_sha256=digest(out/'full_states.pkl'),
            actual_lower_ticks=status['actual_lower_ticks'],allocated_lower_slots=status['allocated_lower_slots'],
            elapsed_s=time.monotonic()-started,optimizer_updates=0,
            scope='Six correlated real forward histories from shared bank3 ancestor; no state injection, replacement, independent-root or TEST claim. Full PathState saved before any new-task rebind/reset.'))
        update(phase='completed',stage='all_prefixes_retained',valid_contexts=sum(r['valid'] for r in records))
    except BaseException as exc:
        import traceback
        (out/'traceback.txt').write_text(traceback.format_exc())
        update(phase='error',error=repr(exc));raise
