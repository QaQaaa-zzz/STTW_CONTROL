"""Restored-state performance harness; all algorithms live in production modules."""
import copy
import hashlib
import pickle
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def measure_repeated(operation, restore, synchronize, warmups=2, repeats=3, observe=None, on_sample=None):
    if warmups < 2 or repeats < 3:
        raise ValueError('at least two warmups and three measurements required')
    result = {'warmup_seconds': [], 'samples_seconds': [], 'intervals': []}
    for index in range(warmups + repeats):
        restore()
        synchronize()
        epoch=time.time()
        start = time.perf_counter()
        operation()
        synchronize()
        result['warmup_seconds' if index < warmups else 'samples_seconds'].append(time.perf_counter()-start)
        result['intervals'].append(dict(kind='warmup' if index<warmups else 'measurement',start_epoch=epoch,end_epoch=time.time()))
        if observe is not None: result.setdefault('outcomes', []).append(observe())
        if on_sample is not None:on_sample(result)
    result['median_seconds'] = statistics.median(result['samples_seconds'])
    return result


def hardware():
    def command(args):
        p = subprocess.run(args, capture_output=True, text=True)
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    versions = {}
    for name in ('jax', 'jaxlib', 'torch', 'mujoco', 'mujoco-mjx', 'rsl-rl-lib', 'flax', 'tensordict'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'not installed'
    return dict(versions=versions, gpu=command(['nvidia-smi']),
                processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv']),
                revision=command(['git','rev-parse','HEAD']), dirty=command(['git','diff','--stat']),
                environment={k:v for k,v in os.environ.items() if k.startswith(('JAX_', 'XLA_', 'CUDA_'))})


def reference_ppo_factory(revision,root):
    """Load immutable reference only for A/B; never copy or select a trainer."""
    import importlib.util
    source=subprocess.check_output(['git','show',revision+':learning/src/sttw_control/direct_command_ppo.py'],text=True)
    path=root/'reference_ppo.py';path.write_text(source)
    spec=importlib.util.spec_from_file_location('sttw_control._reference_ppo',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.make_algorithm


def run(args):
    import jax
    import jax.numpy as jp
    import torch
    from .smooth_command_training import SmoothCampaign, capture_rng, restore_rng
    from .direct_command_training import write
    root = Path(args.output).resolve()
    if root.exists(): raise FileExistsError('benchmark output must be new')
    root.mkdir(parents=True)
    write(root/'arguments.json',vars(args))
    (root/'source.patch').write_text(subprocess.check_output(['git','diff','HEAD','--','learning'],text=True))
    snapshot=root/'source';snapshot.mkdir()
    for name in ('smooth_performance','smooth_command_training','smooth_command_config','smooth_device_statistics','direct_command_training','direct_command_ppo','direct_command_env','smooth_evaluation'):
        source=Path(__file__).with_name(name+'.py')
        if source.exists():(snapshot/source.name).write_text(source.read_text())
    if args.mode=='ppo' and args.rollout_source:
        return run_frozen_ppo(args,root)
    write(root/'hardware_before.json', hardware())
    telemetry = (root/'gpu.csv').open('w')
    monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,pstate,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem','--format=csv','-lms','200'], stdout=telemetry, stderr=subprocess.DEVNULL)
    process_log=(root/'gpu_processes.csv').open('w')
    process_monitor=subprocess.Popen(['nvidia-smi','--query-compute-apps=timestamp,pid,process_name,used_gpu_memory','--format=csv','-lms','1000'],stdout=process_log,stderr=subprocess.DEVNULL)
    campaign = None
    try:
        start = time.perf_counter()
        kwargs = dict(fresh=True, preference_v51=True, smoke=args.smoke,log_mode=args.log_mode,reset_guard=args.reset_guard)
        if args.num_envs is not None or args.rollout_steps is not None: kwargs.update(num_envs=args.num_envs, rollout_steps=args.rollout_steps)
        write(root/'benchmark_status.json',dict(state='initializing',pid=os.getpid(),mode=args.mode,last_update_epoch=time.time()))
        campaign = SmoothCampaign(root/'run', **kwargs)
        campaign.initialize()
        from . import direct_command_ppo as ppo_module
        original_factory=ppo_module.make_algorithm
        try:
            if args.reference_revision:ppo_module.make_algorithm=reference_ppo_factory(args.reference_revision,root)
            for alpha in (0,1):campaign.create_endpoint(alpha)
        finally:ppo_module.make_algorithm=original_factory
        if args.evaluation_parent:
            sources={}
            parent=Path(args.evaluation_parent)
            expected=json.loads((parent/'manifest.json').read_text())
            actual=json.loads((campaign.out/'manifest.json').read_text())
            for key in ('prepared_bank_sha256','lower'):
                if expected[key]!=actual[key]:raise ValueError('evaluation identity mismatch: '+key)
            for alpha in (0,1):
                path=parent/f'alpha{alpha}/checkpoints/update_{args.evaluation_update:04d}.pt'
                saved=torch.load(path,map_location='cuda',weights_only=False)
                for key in ('reward','action','network','plant','limits','reference','lower_controller','lower_reference_centered','actor_temporal_regularizer'):
                    if saved['config'].get(key)!=campaign.resolve(alpha).get(key):raise ValueError('evaluation contract mismatch: '+key)
                campaign.endpoints[alpha]['algo'].policy.load_state_dict(saved['policy'])
                campaign.endpoints[alpha]['update']=args.evaluation_update
                sources[str(alpha)]=dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            write(root/'evaluation_sources.json',sources)
        def sync():
            jax.effects_barrier()
            torch.cuda.synchronize()
        sync()
        preparation = time.perf_counter()-start
        (root/'advance_optimized_hlo.txt').write_text(campaign.advance.as_text())
        original_out = campaign.out
        initial = {}
        for alpha, ep in campaign.endpoints.items():
            initial[alpha] = dict(learner=ep['algo']._snapshot(), storage=copy.deepcopy(ep['algo'].storage),
                                 fields={k:copy.deepcopy(v) for k,v in ep.items() if k not in ('algo','writer','state','obs')},
                                 state=ep['state'], obs=ep['obs'].clone(), accepted=ep['algo'].accepted_policy_updates)
        with (root/'initial_states.pkl').open('wb') as f:
            pickle.dump(jax.device_get({a:s['state'] for a,s in initial.items()}),f)
        torch.save({a:{k:s[k] for k in ('learner','storage','accepted')} for a,s in initial.items()},root/'initial_learners.pt')
        seen = campaign.seen_cases.copy()
        trial = [0]
        def restore():
            trial[0] += 1
            write(root/'benchmark_status.json',dict(state='running',pid=os.getpid(),mode=args.mode,iteration=trial[0],warmups=args.warmups,repeats=args.repeats,last_update_epoch=time.time()))
            campaign.out = root/f'trial_{trial[0]:03d}'
            for alpha, snap in initial.items():
                ep = campaign.endpoints[alpha]; algo, writer = ep['algo'], ep['writer']
                ep.clear(); ep.update(copy.deepcopy(snap['fields']))
                ep.update(algo=algo, writer=writer, state=snap['state'], obs=snap['obs'].clone())
                algo._restore(snap['learner']); algo.storage=copy.deepcopy(snap['storage'])
                algo.accepted_policy_updates=snap['accepted']; algo.hard_kl_stop=False; algo.nonfinite_stop=False
                algo.transition.clear()
                write(campaign.out/f'alpha{alpha}/frozen_config.json', campaign.resolve(alpha))
            campaign.seen_cases=seen.copy()
            restore_rng(initial[0]['fields']['rng'])
        active = [];env_outcomes=[]
        fixed_z = jp.asarray(__import__('numpy').random.default_rng(87).normal(size=(campaign.steps,campaign.n,2)).astype('float32'))
        jax.block_until_ready(fixed_z)
        def env_only():
            state=campaign.endpoints[0]['state'];ticks=jp.int32(0);failures=ends=jp.int32(0);fault=jp.bool_(False)
            for z in fixed_z:
                result=campaign.advance(state,z);state=result[0]
                ticks=ticks+result[10]['all_active_ticks']
                failures=failures+result[7].sum();ends=ends+result[5].sum()
                fault=fault|jp.any(result[3]|result[8])
            jax.block_until_ready((state,ticks,failures,ends,fault))
            if bool(fault):raise RuntimeError('env-only benchmark encountered policy/lower fault')
            active.append(int(ticks));env_outcomes.append(dict(active_ticks=int(ticks),physical_failures=int(failures),episode_ends=int(ends),policy_lower_fault=False))
        def rollout():
            algo=campaign.endpoints[0]['algo']; update=algo.update
            try:
                algo.update=lambda:dict(hard_kl_stop=False,nonfinite_stop=False,benchmark_update_skipped=True)
                if not campaign.batch(0): raise RuntimeError('rollout stopped')
            finally: algo.update=update
        def end_to_end():
            for alpha in (0,1):
                if not campaign.batch(alpha): raise RuntimeError('engineering batch stopped')
                campaign.save(alpha,campaign.endpoints[alpha]['update'])
        operation={'env':env_only,'rollout':rollout,'end-to-end':end_to_end,
                   'evaluation':lambda:campaign.evaluate_preference(args.evaluation_update,batched=args.evaluation_batched)}.get(args.mode)
        if args.mode=='ppo':
            algo=campaign.endpoints[0]['algo']
            if args.rollout_source:
                payload=torch.load(args.rollout_source,map_location='cuda',weights_only=False)
                if payload['config']!=campaign.resolve(0):raise ValueError('frozen rollout configuration mismatch')
                initial[0]['learner']=payload['learner'];initial[0]['accepted']=payload['accepted']
                frozen=payload['storage'];rng={k:payload['learner'][k] for k in ('torch_rng','cuda_rng','numpy_rng','python_rng')}
            else:
                restore(); rollout(); sync()
                frozen=copy.deepcopy(algo.storage);rng=capture_rng()
            torch.save(dict(storage=frozen,learner=algo._snapshot(),config=campaign.resolve(0),accepted=algo.accepted_policy_updates),root/'frozen_rollout.pt')
            restore_initial=restore
            def restore():
                restore_initial();algo.storage=copy.deepcopy(frozen);restore_rng(rng)
            operation=algo.update
        completed=[0]
        def progress():
            if args.mode in ('rollout','end-to-end'):
                ticks=0
                for alpha in ((0,) if args.mode=='rollout' else (0,1)):
                    row=json.loads((campaign.out/f'alpha{alpha}/metrics.jsonl').read_text().splitlines()[-1])
                    ticks+=row['groups']['all']['count']
                active.append(ticks)
            elif args.mode=='evaluation':
                import numpy as np
                active.append(sum(int(np.load(p)['active_tick'].sum()) for p in campaign.out.glob(f'evaluation{args.evaluation_update}/*/*.npz')))
            completed[0]+=1
            write(root/'benchmark_status.json',dict(state='measuring',mode=args.mode,completed_iterations=completed[0],total_iterations=args.warmups+args.repeats,last_update_epoch=time.time()))
        result=measure_repeated(operation,restore,sync,args.warmups,args.repeats,observe=progress,on_sample=lambda row:write(root/'timing_partial.json',row))
        scheduled=(3*(66 if args.evaluation_update==100 else 26)*200 if args.mode=='evaluation'
                   else campaign.n*campaign.steps*4*(2 if args.mode=='end-to-end' else 1))
        result.update(mode=args.mode, reference_ppo_revision=args.reference_revision, preparation_compile_seconds=preparation,
                      scheduled_control_ticks_per_sample=scheduled,physics_substeps_upper_bound_per_sample=scheduled*25,
                      evaluation_physical_concurrency=(18 if args.evaluation_update==100 else 6) if args.evaluation_batched else 3,
                      num_envs=campaign.n,rollout_steps=campaign.steps,seed=87,
                      scheduled_ticks_per_rollout=campaign.n*campaign.steps*4,active_ticks=active,env_outcomes=env_outcomes,
                      jax_compilation_cache_dir=jax.config.jax_compilation_cache_dir,
                      jax_compilation_cache_enabled=jax.config.jax_enable_compilation_cache,
                      jax_device_memory_stats=jax.devices()[0].memory_stats(),
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_cuda=torch.version.cuda, caveats=['GPU peak in gpu.csv is sampled; Torch peak excludes JAX',
                      'Frozen evaluation checkpoints loaded' if args.evaluation_parent else 'Fresh endpoint policies; not control qualification',
                      'End-to-end is two engineering batches plus checkpoint, evaluation is separately measured'])
        write(root/'timing.json',result)
        # A separate instrumented repetition; synchronized scopes are diagnostic,
        # excluded from the uninstrumented throughput samples above.
        phases={}; originals=[]
        def wrap(obj,name,label):
            fn=getattr(obj,name)
            def timed(*a,**kw):
                sync();t=time.perf_counter()
                out=fn(*a,**kw)
                if label in ('environment','device_statistics'):jax.block_until_ready(out)
                sync();phases[label]=phases.get(label,0.)+time.perf_counter()-t
                return out
            originals.append((obj,name,fn));setattr(obj,name,timed)
        restore()
        for name,label in [('advance','environment'),('td','framework_handoff'),('record_episode_statistics','episode_statistics'),('case_manifest','reset_event_output'),('save','checkpoint')]:wrap(campaign,name,label)
        if hasattr(campaign,'accumulate_statistics'):wrap(campaign,'accumulate_statistics','device_statistics')
        if hasattr(campaign,'write_reset_events'):wrap(campaign,'write_reset_events','reset_event_output')
        for ep in campaign.endpoints.values():
            wrap(ep['writer'],'flush','tensorboard_output')
            for name,label in [('act','policy_inference'),('process_env_step','storage'),('compute_returns','GAE'),('update','PPO')]:wrap(ep['algo'],name,label)
        try:
            # Evaluation already records every phase inside each operation.
            # A sixth full panel would add no diagnostic scope.
            if args.mode!='evaluation':operation();sync()
            if args.mode=='rollout':
                algo=campaign.endpoints[0]['algo']
                torch.save(dict(storage=algo.storage,learner=algo._snapshot(),config=campaign.resolve(0),accepted=algo.accepted_policy_updates),root/'frozen_rollout.pt')
            if args.trace:
                restore()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    state=campaign.endpoints[0]['state']
                    for z in fixed_z[:4]:
                        result=campaign.advance(state,z);state=result[0]
                    jax.block_until_ready(state);sync()
                prof.export_chrome_trace(str(root/'torch_trace.json'))
                write(root/'trace_scope.json',dict(mode='fixed-action environment',policy_steps=min(4,campaign.steps),num_envs=campaign.n,excluded_from_throughput=True))
        finally:
            for obj,name,fn in reversed(originals):setattr(obj,name,fn)
        write(root/'diagnostic_phases.json',dict(seconds=phases,evaluation_uses_per_operation_timings=args.mode=='evaluation',overhead='synchronized diagnostic repetition, not additive throughput timings',
            evaluation_timing_files=[str(p.relative_to(root)) for p in root.glob('trial_*/evaluation*/**/*timings.json')],
            unmeasured=['general_file_output'] if args.mode=='evaluation' else ['reward_audit','plotting','general_file_output']))
        if args.verify_log_modes:
            verify_log_modes(campaign, root)
        campaign.out=original_out;campaign._status(state='complete',stage='benchmark',benchmark_only=True)
        write(root/'benchmark_status.json',dict(state='complete',mode=args.mode,completed_iterations=args.warmups+args.repeats,total_iterations=args.warmups+args.repeats,last_update_epoch=time.time()))
    except BaseException as exc:
        write(root/'benchmark_status.json',dict(state='error',reason=repr(exc),last_update_epoch=time.time()))
        if campaign:
            campaign.out=root/'run';campaign._status(state='error',stage='benchmark',reason=repr(exc))
        raise
    finally:
        if campaign:
            for ep in campaign.endpoints.values(): ep['writer'].close()
        monitor.terminate();monitor.wait();telemetry.close()
        process_monitor.terminate();process_monitor.wait();process_log.close()
        write(root/'hardware_after.json',hardware())


def verify_log_modes(campaign, root):
    """Full pytree fixed-action regression, including forced horizon resets."""
    import jax
    import jax.numpy as jp
    import numpy as np
    from .direct_command_training import write
    from .direct_command_env import task_horizon_steps
    _, full, _ = campaign.setup_batch(campaign.n,log_mode='evaluation_full')
    _, summary, _ = campaign.setup_batch(campaign.n,log_mode='training_summary',reset_guard=campaign.reset_guard)
    results=[];violations=[]
    # Keep the original tolerance. A/A diagnoses nondeterminism; it does not
    # automatically excuse an A/B failure or widen acceptance thresholds.
    for count in (0,1,campaign.n//2,campaign.n):
        base=campaign.states.replace(tick=jp.where(jp.arange(campaign.n)<count,task_horizon_steps(campaign.spec)*4-4,0))
        a=b=aa=base;maximum={'AA':0.,'AB':0.}
        for step in range(min(campaign.steps,16)):
            z=jp.asarray(np.random.default_rng(87+step).normal(size=(campaign.n,2)),jp.float32)
            x=full(a,z);repeat=full(aa,z);y=summary(b,z)
            for kind,other in [('AA',repeat),('AB',y)]:
                for index in list(range(10))+[11]:
                    lefts,_=jax.tree_util.tree_flatten_with_path(x[index]);rights=jax.tree.leaves(other[index])
                    for (path,left),right in zip(lefts,rights):
                        left,right=np.asarray(left),np.asarray(right)
                        if left.dtype.kind in 'biu':ok=np.array_equal(left,right);difference=0. if ok else 1.
                        else:
                            ok=np.allclose(left,right,rtol=2e-4,atol=2e-5,equal_nan=False)
                            difference=float(np.max(np.abs(left-right))) if left.size else 0.
                        maximum[kind]=max(maximum[kind],difference)
                        if not ok:violations.append(dict(kind=kind,forced_done_count=count,step=step,output=index,path=str(path),max_absolute_difference=difference,discrete=left.dtype.kind in 'biu',nonfinite=not(np.isfinite(left).all() and np.isfinite(right).all())))
            a,b,aa=x[0],y[0],repeat[0]
        results.append(dict(forced_done_count=count,steps=min(campaign.steps,16),max_absolute_difference=maximum))
    decisions_equal=not any(v['discrete'] or v['nonfinite'] for v in violations)
    receipt=dict(passed=not violations,strict_numeric_passed=not violations,contract_decisions_equal=decisions_equal,continuous_review_required=bool(violations),rtol=2e-4,atol=2e-5,cases=results,violations=violations,note='Tiny differences do not alone block work; no blanket acceptance of finite continuous-state errors. Review physical magnitudes and independent contracts before adoption.')
    write(root/'full_state_equivalence.json',receipt)
    if not decisions_equal:raise AssertionError('discrete decisions or finiteness differ; see full_state_equivalence.json')


def run_frozen_ppo(args,root):
    """No simulator setup needed to benchmark exactly the same serialized rollout."""
    import torch
    from .direct_command_ppo import make_algorithm
    from .direct_command_training import write
    payload=torch.load(args.rollout_source,map_location='cuda',weights_only=False)
    if args.reference_revision:
        make_algorithm=reference_ppo_factory(args.reference_revision,root)
    torch.set_num_threads(2)
    frozen=payload['storage'];config=payload['config']
    algo=make_algorithm(frozen.observations[0],frozen.num_transitions_per_env,config,'cuda');algo.temporal_spec=config
    def restore():
        algo._restore(payload['learner']);algo.storage=copy.deepcopy(frozen)
        algo.accepted_policy_updates=payload['accepted'];algo.hard_kl_stop=False;algo.nonfinite_stop=False
    outcomes=[]
    def operation():outcomes.append(algo.update())
    write(root/'hardware_before.json',hardware())
    telemetry=(root/'gpu.csv').open('w')
    monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,pstate,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem','--format=csv','-lms','200'],stdout=telemetry,stderr=subprocess.DEVNULL)
    try:
        result=measure_repeated(operation,restore,torch.cuda.synchronize,args.warmups,args.repeats,on_sample=lambda row:write(root/'timing_partial.json',row))
        result.update(mode='ppo',frozen_rollout=str(Path(args.rollout_source).resolve()),
            frozen_rollout_sha256=hashlib.sha256(Path(args.rollout_source).read_bytes()).hexdigest(),
            reference_revision=args.reference_revision,metrics=outcomes,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write(root/'timing.json',result)
        torch.save(dict(snapshot=algo._snapshot(),metrics=outcomes[-1]),root/'result.pt')
        if args.trace:
            restore()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                operation();torch.cuda.synchronize()
            prof.export_chrome_trace(str(root/'torch_trace.json'))
    finally:
        monitor.terminate();monitor.wait();telemetry.close()
        write(root/'hardware_after.json',hardware())
