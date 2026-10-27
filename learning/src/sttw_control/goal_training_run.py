"""One fixed ordinary goal-learning baseline, not a coordination algorithm."""
import copy
import json
import os
import pickle
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json, digest


def training_cases(reference, iteration, seed):
    """Frozen inventory, fresh action noise and hidden onset; no sampled labels."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, iteration, 331]))
    cases = []
    for context in ('left', 'right'):
        base = next(c for c in reference if c['context'] == context and c['profile'] == 7)
        for alpha in (0., .5, 1.):
            for force in (0., -32., 32.):
                for rep in range(2):
                    cases.append(dict(base, alpha=alpha, force_N=force, replicate=rep,
                        pulse_onset_task_tick=int(rng.integers(400, 601)),
                        episode_id=f'collection{iteration:04d}/episode{len(cases):03d}',
                        rng_fold_in=len(cases)))
    return cases


def episodes_from_logs(decisions, physical, cases, goals, collection_id):
    """Retain exact declared inventory, including no-action/fault/failed cases."""
    if decisions['valid'].shape[1] != len(cases) or physical['active_tick'].shape[1] != len(cases):
        raise ValueError('whole declared population required')
    return [dict(episode_id=c['episode_id'], case=copy.deepcopy(c), collection_id=collection_id,
        original_goal_xy=np.asarray(goals[c['context']], float).tolist(),
        decisions={k:v[:,i] for k,v in decisions.items()},
        physical={k:v[:,i] for k,v in physical.items()}) for i,c in enumerate(cases)]


def prepare(root):
    root=Path(root).resolve(); mount=Path('/media/qy/6A7E3DB07E3D75C3')
    stat=os.statvfs('/home/qy/STTW_CONTROL')
    if not os.path.ismount(mount) or not root.is_relative_to(mount) or stat.f_bavail*stat.f_frsize < 50*1024**3:
        raise ValueError('mounted data output and at least50GiB system space required')
    source=Path('runs/live_preference_collector_v1_attempt2').resolve()
    old=json.loads((source/'plan.json').read_text()); cfg=json.loads((source/'effective_config.json').read_text())
    if json.loads((source/'status.json').read_text())['phase']!='completed':raise ValueError('completed source required')
    inputs=[source/n for n in ('plan.json','effective_config.json','source_declaration.json')]
    for key in ('actor','checkpoint'):
        if digest(old['model'][key])!=old['model'][key+'_sha256']:raise ValueError('parent identity changed')
        inputs.append(Path(old['model'][key]))
    for c in old['contexts'].values():
        for key in ('initial','path'):
            if digest(c[key])!=c[key+'_sha256']:raise ValueError('initial/path identity changed')
            inputs.append(Path(c[key]))
    declaration=json.loads((source/'source_declaration.json').read_text())['source']
    inputs += [Path(declaration['prepared_bank']),Path(declaration['upper_run'])/'alpha0/frozen_config_stage52.json']
    inputs += [Path(declaration['candidate'][k]) for k in ('actor','checkpoint','config')]
    cap=sum(float(x) for x in cfg['path_reward']['component_caps'].values())
    if not np.isclose(cap,cfg['path_reward']['failure_cost_rate_bound']):raise ValueError('cost cap bound mismatch')
    root.mkdir(parents=True,exist_ok=False)
    sources=list(Path('learning/src/sttw_control').rglob('*.py'))+[Path('learning/cli/train_goal_baseline.py')]
    hashes={}
    for p in sources:
        absolute=p.resolve(); hashes[str(absolute)]=digest(absolute)
        target=root/'source_at_launch'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    plan=dict(project='STTW_CONTROL',experiment_id='STTW-ORDINARY-GOAL-LEARNING-V1',
        question='Does ordinary goal learning provide task utility beyond frozen parent/zero/speed support?',
        baseline_not_novel_method=True,source=str(source),model=old['model'],contexts=old['contexts'],
        reference_cases=old['cases'],inputs={str(p):digest(p) for p in inputs},sources=hashes,
        initialization='weights_only actual task_only64 Actor/logstd; zero goal branch/new GCS; fresh Adam',
        horizon_s=10.,radius_m=1.,goal_distance_scale_m=10.,episodes=36,collections=64,warmup_collections=2,
        seed=2026102727,torch_threads=4,maximum_allocated_lower_slots=4608000,
        learner=dict(objective_weights=dict(goal=2.,cost=.5,survival=1.),cost_target_scale=cap*cfg['path_reward']['scale']*10.,
            epochs=4,actor_lr=1e-4,value_lr=3e-4,clip=.2,entropy_coef=.001,grad_clip=1.,actor_aggregation='scalarize_then_clip'),
        effective_training_contract='plan.learner/collections/warmup/episodes override legacy training_future_phase_C template fields; G/C/S returns undiscounted, legacy failure replacement excluded from C',
        objective='2G+S-.5*Cpre/B; individual-outcome ordering only, not chance constraints or expected lexicographic priority',
        evaluation=dict(status='frozen_design_not_executed',roles=['fixed_endpoint64','parent_task64','zero','zero_plus_profile7_speed_support'],
            permissions='Actor and parent: profile7/reentry; zero: no residual or support; rule: existing speed_support_latent from zeroNN with same eligibility, not independent-support override',
            original_alpha=[0.,.5,1.],horizon_s=10.,radius_m=1.,
            transfer=dict(path_radius_m=1.25,angle_deg=90.,left_speed_m_s=1.7,right_speed_m_s=1.6,onsets_task_ticks=[360,640],forces_N=[0.,-32.,32.],duration_s=.1),
            training_geometry_check=dict(radius_m=1.1,left_speed_m_s=1.8,right_speed_m_s=1.5,onset_task_tick=500,forces_N=[0.,-32.,32.]),
            declared_physical_cases=dict(transfer=80,training_geometry_check=48,total=128),
            baseline_aliases='zero and zero-plus-support are alpha-independent in control; execute alpha0 physical once, rescore original alpha0/.5/1 costs with existing original reward, retaining aliases rather than independent repeats',
            scope='Joint geometry/speed/time DEV transfer, same correlated origin; not independent TEST or single-factor generalization',
            zeros='One zero-force physical trajectory per distinct policy/alpha/context, not per onset. Shared physical policies may be rescored without extra independent count.',
            adoption='No automatic adoption; paired G/S gains/losses and original-unit alpha costs, no best-checkpoint selection',
            stop_family='If ordinaryA is covered by simple controls, stop this learning route; goal gain alone is not disturbance-tolerance gain'),
        acceleration=['vmap36','scan50 x10','dynamic parameters reuse one JIT','persistent JAX cache'],
        stop='Exactly64 complete fresh collections, no wallclock cap/automatic extension/restart. Preserve errors/unknown and stop; contact is a normal label.',
        first_collection='Full first collection belongs to64 and is value warmup; no extension of historical200tick engineering.',
        code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    atomic_json(root/'effective_config.json',cfg);atomic_json(root/'plan.json',plan)
    atomic_json(root/'seal.json',dict(plan_sha256=digest(root/'plan.json'),config_sha256=digest(root/'effective_config.json')))
    atomic_json(root/'status.json',dict(phase='prepared',collections_completed=0,active_lower_ticks=0,allocated_lower_slots=0,optimizer_steps=[0,0]))
    return plan


def _tensorboard(root):
    import urllib.request
    for port in range(6045,6060):
        with socket.socket() as s:
            try:s.bind(('127.0.0.1',port));break
            except OSError:continue
    else:raise RuntimeError('no free TensorBoard port')
    with (root/'tensorboard.log').open('x') as f:
        process=subprocess.Popen([sys.executable,'-m','tensorboard.main','--logdir',str(root/'tensorboard'),
            '--port',str(port),'--host','127.0.0.1','--reload_interval','5'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    url=f'http://localhost:{port}'
    for _ in range(30):
        if process.poll() is not None:raise RuntimeError('TensorBoard exited')
        try:
            with urllib.request.urlopen(url,timeout=2) as response:
                if response.status==200:break
        except OSError:pass
        time.sleep(1)
    else:raise RuntimeError('TensorBoard unavailable')
    atomic_json(root/'tensorboard_server.json',dict(url=url,pid=process.pid,port=port))
    return url


def execute(root):
    root=Path(root).resolve();status=json.loads((root/'status.json').read_text())
    if status['phase']!='prepared':raise ValueError('fresh prepared run only; no implicit restart')
    def update(**kwargs):
        status.update(kwargs,checked_unix=time.time());atomic_json(root/'status.json',status)
    update(phase='running',stage='loading',pid=os.getpid());learner=None;writer=None
    try:
        import torch
        import jax
        import jax.numpy as j
        from torch.utils.tensorboard import SummaryWriter
        from .runtime import configure_compilation_cache
        from .goal_networks import GoalNetworkBundle,make_goal_collection_evaluator
        from .goal_learner import GoalLearner
        from .goal_live_collection import GoalLiveCollectionState,make_goal_live_collection_step
        from .contact_only_environment import load_environment_contact_only
        from .path_command_policy import make_torch_path_network,export_torch_actor
        from .geometric_path import Path as GeometricPath
        from .causal_reference_baseline import initial_gate
        from .anticipatory_run import flatten_ticks
        plan=json.loads((root/'plan.json').read_text());seal=json.loads((root/'seal.json').read_text())
        if digest(root/'plan.json')!=seal['plan_sha256'] or digest(root/'effective_config.json')!=seal['config_sha256']:raise ValueError('sealed contract changed')
        for p,sha in dict(plan['inputs'],**plan['sources']).items():
            if digest(p)!=sha:raise ValueError('sealed source/input changed: '+p)
        cfg=json.loads((root/'effective_config.json').read_text())
        declaration=json.loads((Path(plan['source'])/'source_declaration.json').read_text())['source']
        torch.set_num_threads(plan['torch_threads']);torch.manual_seed(plan['seed'])
        parent=make_torch_path_network(351,2,cfg)
        saved=torch.load(plan['model']['checkpoint'],map_location='cpu',weights_only=False)
        parent.load_state_dict(saved['actor'],strict=True)
        with Path(plan['model']['actor']).open('rb') as f:original_actor=pickle.load(f)
        exported=export_torch_actor(parent,cfg)
        if jax.tree.structure(exported)!=jax.tree.structure(original_actor):raise ValueError('parent Actor structure mismatch')
        for a,b in zip(jax.tree.leaves(exported),jax.tree.leaves(original_actor)):
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        np.testing.assert_array_equal(saved['log_std'].detach().numpy(),np.asarray(plan['model']['log_std'],np.float32))
        net=GoalNetworkBundle(parent,cfg,saved['log_std'])
        learner=GoalLearner(net,**plan['learner'])
        checkpoint_dir=root/'checkpoints';checkpoint_dir.mkdir();torch.save(learner.checkpoint(),checkpoint_dir/'learner_0000.pt')
        configure_compilation_cache('/home/qy/STTW_CONTROL/runs/.jax_compilation_cache')
        if not any(d.platform=='gpu' for d in jax.devices()):raise ValueError('real GPU required')
        env,_=load_environment_contact_only(cfg,declaration,'terminal_contract_v2');paths={};origins={};goals={}
        for context,c in plan['contexts'].items():
            with Path(c['initial']).open('rb') as f:bank=pickle.load(f).state
            origins[context]=jax.tree.map(lambda x:j.asarray(x[0]),bank)
            s=origins[context]
            if int(s.tick)!=0 or int(s.physical.physical_tick)!=700 or bool(s.fault|s.domain_exit|s.physical.failed):raise ValueError('original complete task0 state required')
            with np.load(c['path']) as z:
                paths[context]=GeometricPath(*(j.asarray(z[k]) for k in ('s','xy','heading','curvature','turn_end','goal')))
                goals[context]=np.array([np.interp(float(z['goal']),z['s'],z['xy'][:,i]) for i in (0,1)])
        stack=lambda xs:jax.tree.map(lambda *x:j.stack(x),*xs)
        cases0=training_cases(plan['reference_cases'],0,plan['seed'])
        if len(cases0)!=plan['episodes']:raise ValueError('fixed episode count mismatch')
        path_batch=stack([paths[c['context']] for c in cases0]);speed=j.asarray([c['speed'] for c in cases0],j.float32)
        profiles=j.full((plan['episodes'],),7,j.int32);goal_batch=j.asarray([goals[c['context']] for c in cases0],j.float32)
        env.set_log_template(origins['left'],paths['left'],speed[0])
        one=make_goal_live_collection_step(env,make_goal_collection_evaluator(cfg),horizon_s=plan['horizon_s'],radius_m=plan['radius_m'],distance_scale_m=plan['goal_distance_scale_m'])
        @jax.jit
        def chunk(carry,parameters,pulses):
            return jax.lax.scan(lambda st,_:jax.vmap(one)(st,parameters,parameters['log_std'],path_batch,speed,pulses,profiles,goal_batch),carry,None,length=50)
        url=_tensorboard(root);writer=SummaryWriter(str(root/'tensorboard'))
        update(tensorboard_url=url,stage='starting_collections')
        started=time.monotonic()
        for iteration in range(plan['collections']):
            stat=os.statvfs('/home/qy/STTW_CONTROL')
            if stat.f_bavail*stat.f_frsize<50*1024**3:raise RuntimeError('system free space below50GiB')
            directory=root/'collections'/f'{iteration+1:04d}';directory.mkdir(parents=True,exist_ok=False)
            cases=training_cases(plan['reference_cases'],iteration,plan['seed'])
            manifest=[dict(episode_id=c['episode_id'],case=c,original_goal_xy=goals[c['context']].tolist()) for c in cases]
            frozen=learner.freeze_collection(manifest,horizon_s=plan['horizon_s'],radius_m=plan['radius_m'])
            atomic_json(directory/'sampling_identity.json',{k:v for k,v in frozen.items() if k!='parameters'})
            atomic_json(directory/'cases.json',cases)
            key=jax.random.fold_in(jax.random.PRNGKey(plan['seed']),iteration)
            carry=stack([GoalLiveCollectionState(origins[c['context']].replace(alpha=j.float32(c['alpha'])),
                jax.random.fold_in(key,c['rng_fold_in']),initial_gate(),j.int32(0),j.bool_(False)) for c in cases])
            parameters=stack([frozen['parameters'] for _ in cases])
            pulses=j.asarray([[(700+c['pulse_onset_task_tick'])*.005,(700+c['pulse_onset_task_tick'])*.005+.1,
                *[c['force_N']*d for d in c['world_direction']]] for c in cases],j.float32)
            if iteration==0:
                update(stage='real_plant_shape',collection=1)
                shaped=jax.eval_shape(chunk,carry,parameters,pulses)
                if shaped[1][0]['goal_context'].shape!=(50,36,5) or shaped[1][1]['active_tick'].shape!=(50,36,4):
                    raise ValueError('actual plant collection shape mismatch')
                atomic_json(root/'shape_receipt.json',dict(goal_context=[50,36,5],physical_slots=[50,36,4],physics_integrations=0,optimizer_steps=0))
            with (directory/'initial_carry.pkl').open('wb') as f:pickle.dump(jax.device_get(carry),f)
            decisions=[];physical=[];collection_started=time.monotonic()
            for block in range(10):
                update(stage='collecting',collection=iteration+1,block=block,inflight_slots=plan['episodes']*200,
                    inflight_actual_cost='unknown_until_device_return')
                carry,(d,p)=chunk(carry,parameters,pulses);jax.block_until_ready(carry)
                host,d,p=jax.device_get((carry,d,p));p=flatten_ticks(p)
                # Count known executed work before filesystem writes can fail.
                update(stage='saving_raw',active_lower_ticks=status['active_lower_ticks']+int(p['active_tick'].sum()),
                    allocated_lower_slots=status['allocated_lower_slots']+int(p['active_tick'].size),inflight_slots=0,
                    inflight_actual_cost='known_from_returned_active_mask')
                np.savez_compressed(directory/f'chunk_{block:02d}.npz',**{'decision_'+k:v for k,v in d.items()},**{'lower_'+k:v for k,v in p.items()})
                decisions.append(d);physical.append(p)
                if np.any(host.state.fault) or np.any(p['active_tick']&(p['nonfinite']|p['lower_fault']|p['policy_fault'])):raise ValueError('engineering fault; raw preserved')
            with (directory/'final_carry.pkl').open('wb') as f:pickle.dump(host,f)
            d={k:np.concatenate([x[k] for x in decisions]) for k in decisions[0]}
            p={k:np.concatenate([x[k] for x in physical]) for k in physical[0]}
            episodes=episodes_from_logs(d,p,cases,goals,frozen['collection_id'])
            ready=learner.prepare_population(episodes)
            atomic_json(directory/'population_receipt.json',ready)
            rows=ready['summaries'];warmup=iteration<plan['warmup_collections']
            diagnostics=[]
            for i,(case,row) in enumerate(zip(cases,rows)):
                impulse=p['applied_force_impulse'][:,i][p['active_tick'][:,i]].sum(axis=0,dtype=np.float64)
                expected=case['force_N']*.1*np.asarray(case['world_direction'])
                diagnostics.append(dict(**row,case=case,impulse_Ns=impulse.tolist(),
                    full_pulse_exposure=bool(np.allclose(impulse,expected,rtol=.01,atol=1e-5)),
                    exposure_scope='Diagnostic only; early-contact and partial exposure episodes remain in original population'))
            atomic_json(directory/'case_diagnostics.json',diagnostics)
            update(stage='updating',collection=iteration+1,training_phase='value_warmup' if warmup else 'ordinary_PPO')
            result=learner.update(warmup=warmup);atomic_json(directory/'learner_result.json',result)
            target=checkpoint_dir/f'learner_{iteration+1:04d}.pt';torch.save(learner.checkpoint(),target)
            atomic_json(root/'latest_checkpoint.json',dict(path=str(target),sha256=digest(target),collections_completed=iteration+1))
            valid_reward=p['active_tick'] & p['task_reward_valid']
            running_return=np.where(valid_reward,p['tick_reward'],0.).sum(axis=0,dtype=np.float64)
            writer.add_scalar('rollout/reward',float(running_return.mean()),iteration+1)
            writer.add_scalar('rollout/goal',np.mean([r['goal']=='arrived' for r in rows]),iteration+1)
            writer.add_scalar('rollout/survival',np.mean([r['survival']=='survived' for r in rows]),iteration+1)
            writer.add_scalar('rollout/joint',np.mean([r['goal']=='arrived' and r['survival']=='survived' for r in rows]),iteration+1)
            writer.add_scalar('train/actor_loss',result['history'][-1]['actor'],iteration+1)
            writer.add_scalar('train/value_loss',result['history'][-1]['value'],iteration+1)
            writer.add_scalar('time/collection_s',time.monotonic()-collection_started,iteration+1);writer.flush()
            atomic_json(directory/'completion.json',dict(collection=iteration+1,checkpoint_sha256=digest(target),warmup=warmup,
                active_ticks=int(p['active_tick'].sum()),allocated_slots=int(p['active_tick'].size),elapsed_s=time.monotonic()-collection_started))
            if status['allocated_lower_slots']>plan['maximum_allocated_lower_slots']:raise ValueError('declared slot count exceeded')
            update(collections_completed=iteration+1,optimizer_steps=learner.optimizer_steps,elapsed_s=time.monotonic()-started)
        update(phase='completed',stage='fixed_endpoint_ready',evaluation_status='not_started',adoption='pending')
    except BaseException as exc:
        (root/'traceback.txt').write_text(traceback.format_exc())
        if learner is not None:
            import torch
            try:torch.save(learner.checkpoint(),root/'error_forensic.pt')
            except Exception as save_error:status['forensic_save_error']=repr(save_error)
        update(phase='error',error=repr(exc),restart='not automatic; preserve partial optimizer/physical state')
        raise
    finally:
        if writer is not None:writer.close()
