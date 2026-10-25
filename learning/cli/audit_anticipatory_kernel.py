"""Actual-state tracing and a single bounded synchronized GPU comparison."""
import argparse
import json
import os
import pickle
from pathlib import Path
import time
import traceback
import numpy as np
from sttw_control.preference_best import atomic_json,digest


def run(root,phase,source):
    from prepare_short_recovery_campaign import register
    if phase=='shape':root.mkdir(parents=True,exist_ok=False)
    target=root/('interface_status.json' if phase=='shape' else 'status.json')
    if phase=='benchmark':
        if target.exists():raise ValueError('benchmark already attempted; preserve evidence')
        if json.loads((root/'interface_status.json').read_text())['phase']!='completed':raise ValueError('actual interface trace required')
    status=dict(phase='running',stage='load_original_frozen_objects',pid=os.getpid(),
                active_lower_ticks=0,optimizer_updates=0,maximum_lower_ticks=960,checked_unix=time.time())
    def update(**kw):status.update(kw,checked_unix=time.time());atomic_json(target,status)
    update();register(root/('INTERFACE_ACTIVE_RUN.json' if phase=='shape' else 'ACTIVE_RUN.json'),
        'STTW_CONTROL 提前参考共用内核'+('零积分核对' if phase=='shape' else '960步同步GPU基准'),target)
    try:
        import jax
        import jax.numpy as j
        from sttw_control.contact_only_environment import load_environment_contact_only
        from sttw_control.geometric_path import Path as GeometricPath
        from sttw_control.path_command_policy import make_path_actor
        from sttw_control.causal_reference_baseline import initial_gate,validate_causal_contract
        from sttw_control.anticipatory_reference import ReferenceEvaluationState,make_reference_step
        from sttw_control.survival_endpoint_models import authenticate_endpoint
        from sttw_control.survival_endpoint_collection import make_endpoint_collection_step
        from sttw_control.runtime import configure_compilation_cache
        files=[Path(__file__),Path('learning/src/sttw_control/anticipatory_reference.py'),
               Path('learning/src/sttw_control/live_path_continuation.py')]
        if phase=='benchmark':
            plan=json.loads((root/'plan.json').read_text())
            for name,sha in plan['source_sha256'].items():
                if digest(name)!=sha:raise ValueError('changed audited source '+name)
        source_plan=json.loads((source/'plan.json').read_text())
        authenticated=authenticate_endpoint(source_plan['training_run'])
        config=json.loads((source/'common_environment_config.json').read_text())
        if config!=authenticated['config']:raise ValueError('common config drift')
        declaration=json.loads((source/'source_declaration.json').read_text())
        env,_=load_environment_contact_only(config,declaration['source'],'terminal_contract_v2')
        validate_causal_contract(config,env.plant_spec['network']['frame_fields'])
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        group=json.loads((source/'groups.json').read_text())[0]
        assert group['inference_mode']=='mean' and group['speed_reference_floor_m_s'] is None
        cases=json.loads((source/'cases.json').read_text());cases=[cases[i] for i in group['case_indices']]
        count=len(cases);assert count==12
        statefile=source/group['group_id']/'initial.pkl'
        with statefile.open('rb') as f:old=jax.tree.map(j.asarray,pickle.load(f))
        base=old.state
        first=jax.tree.map(lambda a:a[0],base)
        routes={}
        for name in {r['route'] for r in cases}:
            with np.load(source/'reference_paths'/f'{name}.npz') as z:
                routes[name]=GeometricPath(*(j.asarray(z[k]) for k in ['s','xy','heading','curvature','turn_end','goal']))
        env.set_log_template(first,routes[cases[0]['route']],j.float32(cases[0]['speed']))
        stack=lambda items:jax.tree.map(lambda *x:j.stack(x),*items)
        paths=stack([routes[c['route']] for c in cases]);speed=j.asarray([c['speed'] for c in cases],j.float32)
        weights={}
        for role in {c['role'] for c in cases}:
            with Path(authenticated['models'][role]['actor']).open('rb') as f:weights[role]=jax.tree.map(j.asarray,pickle.load(f))
        parameters=stack([weights[c['role']] for c in cases])
        start=float(np.asarray(base.physical.physical_tick)[0])*.005
        pulse=j.asarray([[start+c['pulse_onset_task_tick']*.005,start+c['pulse_onset_task_tick']*.005+c['pulse_duration_s'],
            *(c['force_N']*np.asarray(c['world_direction']))] for c in cases],j.float32)
        gate=jax.tree.map(lambda a:j.broadcast_to(a,(count,)+a.shape),initial_gate())
        new=ReferenceEvaluationState(base,gate)
        actor=make_path_actor(config)
        old_step=make_endpoint_collection_step(env,actor,horizon_s=10.,mode='mean')
        new_step=make_reference_step(env,actor,horizon_s=10.)
        @jax.jit
        def old_chunk(c,parameters,paths,speed,pulse):
            return jax.lax.scan(lambda s,_:jax.vmap(old_step)(s,parameters,j.zeros((count,2)),paths,speed,pulse),c,None,length=5)
        @jax.jit
        def new_chunk(c,parameters,paths,speed,pulse):
            return jax.lax.scan(lambda s,_:jax.vmap(new_step)(s,parameters,paths,speed,pulse,
                j.zeros(count,j.int32),j.ones(count),j.zeros(count,bool)),c,None,length=5)
        if phase=='shape':
            shape=jax.eval_shape(new_chunk,new,parameters,paths,speed,pulse)
            assert shape[1]['active_tick'].shape==(5,12,4)
            assert shape[1]['effective_costs'].shape==(5,12,4,10)
            atomic_json(root/'plan.json',dict(project='STTW_CONTROL',purpose=__doc__,
                source_run=str(source),source_initial_state=str(statefile),source_initial_state_sha256=digest(statefile),
                source_sha256={str(p.resolve()):digest(p) for p in files},
                config_sha256=digest(source/'common_environment_config.json'),model_metadata=authenticated['metadata'],
                batch=12,upper_steps_per_call=5,calls=dict(old_warm=1,old_measure=1,new_warm=1,new_measure=1),
                maximum_active_lower_ticks=960,optimizer_updates=0,
                scope='Domain-valid first0.1s only; all profiles/base zero, no force reaches this interval. No post-domain semantic equivalence or end-to-end campaign speed claim.'))
            atomic_json(root/'interface_receipt.json',dict(physics_steps=0,optimizer_updates=0,
                active_log_shape=list(shape[1]['active_tick'].shape),cost_log_shape=list(shape[1]['effective_costs'].shape)))
            update(phase='completed',stage='actual_N12_common_kernel_traced_no_integration')
            return
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('GPU benchmark requires actual GPU backend')
        receipt=dict(device=[dict(platform=d.platform,kind=d.device_kind) for d in jax.devices()],
                     jax_version=jax.__version__,numpy_version=np.__version__,active_lower_ticks=0)
        outputs={}
        for name,fn,initial in [('old',old_chunk,old),('new',new_chunk,new)]:
            update(stage='compile_and_warm_'+name)
            t=time.perf_counter();warm=fn(initial,parameters,paths,speed,pulse);jax.block_until_ready(warm);receipt[name+'_cold_s']=time.perf_counter()-t
            logs=warm[1][1] if name=='old' else warm[1]
            used=int(np.asarray(logs['active_tick']).sum());assert used==240
            receipt['active_lower_ticks']+=used;update(active_lower_ticks=receipt['active_lower_ticks'])
            update(stage='synchronized_measure_'+name)
            t=time.perf_counter();out=fn(initial,parameters,paths,speed,pulse);jax.block_until_ready(out);receipt[name+'_warm_s']=time.perf_counter()-t
            logs=out[1][1] if name=='old' else out[1]
            used=int(np.asarray(logs['active_tick']).sum());assert used==240
            receipt['active_lower_ticks']+=used;update(active_lower_ticks=receipt['active_lower_ticks'])
            outputs[name]=jax.device_get(out)
        a,b=outputs['old'][0].state,outputs['new'][0].state
        differences={}
        for name,x,y in [('qpos',a.physical.data.qpos,b.physical.data.qpos),
                         ('qvel',a.physical.data.qvel,b.physical.data.qvel),
                         ('reference',a.physical.governor.current_reference,b.physical.governor.current_reference),
                         ('upper_history',a.history.frames,b.history.frames),
                         ('offset',a.offsets,b.offsets)]:
            differences[name]=float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
            if not np.isfinite(differences[name]):raise ValueError('nonfinite benchmark comparison')
        assert receipt['active_lower_ticks']==960
        receipt.update(differences=differences,warm_speedup=receipt['old_warm_s']/receipt['new_warm_s'],
            physical_steps_each_measured_path=240,optimizer_updates=0,
            old_domain_exit=bool(np.any(a.domain_exit)),new_domain_exit=bool(np.any(b.domain_exit)),
            old_contact_or_fault=bool(np.any(a.physical.failed|a.fault)),new_contact_or_fault=bool(np.any(b.physical.failed|b.fault)),
            scope='One synchronized warm sample each on identical real N12 initial data; includes output readiness, excludes saving outputs. Not a multi-repeat performance or post-domain equivalence claim.')
        atomic_json(root/'receipt.json',receipt)
        update(phase='completed',stage='bounded_GPU_benchmark_reported')
        print(json.dumps(receipt,indent=2))
    except BaseException as exc:
        (root/(phase+'_traceback.txt')).write_text(traceback.format_exc())
        update(phase='error',error=repr(exc));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=['shape','benchmark'],required=True)
    p.add_argument('--source',type=Path,default=Path('runs/live_path_rejoin_pilot_v3'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output.resolve(),a.phase,a.source.resolve())
