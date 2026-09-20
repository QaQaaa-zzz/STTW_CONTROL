#!/usr/bin/env python3
"""Execute every declared comparison arm sequentially; preserve failed artifacts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from sttw_control.deferred_training import verify_files


def run(plan_path):
    plan=json.loads(Path(plan_path).read_text())
    from sttw_control.env import load_config
    from sttw_control.training import TrainingConfig
    for arm in plan['arms']:
        load_config(arm['task']);TrainingConfig(**json.loads(Path(arm['training']).read_text()))
        panel=json.loads(Path(arm['panel']).read_text())
        for key in ('evaluation_seeds','start_seconds','duration_seconds','path_tolerance_m','extra_tolerance_m','heading_tolerance_rad','hold_seconds'): 
            if key not in panel:raise ValueError('panel missing '+key)
    root=Path(plan['output']);root.mkdir(exist_ok=False)
    completed=[]
    def status(phase,**extra):
        temp=root/'status.tmp';temp.write_text(json.dumps(dict(phase=phase,completed=completed,**extra),indent=2)+'\n');temp.replace(root/'status.json')
    env=dict(os.environ,PYTHONPATH=str(Path(plan['repository'])/'learning/src'),MUJOCO_GL='egl',XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1');env.pop('JAX_PLATFORMS',None)
    def invoke(args,log,cpu=False):
        verify_files(plan['input_sha256']); e=dict(env)
        if cpu:e['JAX_PLATFORMS']='cpu'
        with log.open('x') as f:subprocess.run([sys.executable,*map(str,args)],cwd=plan['repository'],env=e,stdout=f,stderr=subprocess.STDOUT,check=True)
    try:
        for arm in plan['arms']:
            verify_files(plan['input_sha256']); out=root/arm['name'];out.mkdir();frozen=out/'frozen';frozen.mkdir()
            for key in ('task','training','panel'):(frozen/f'{key}.json').write_bytes(Path(arm[key]).read_bytes())
            (out/'declaration.json').write_text(json.dumps(dict(priority_alphas=arm['alphas'],arm=arm,source_sha256=plan['input_sha256'],role='development comparison; not holdout'),indent=2)+'\n')
            status('training',arm=arm['name']);invoke(['learning/cli/train.py','--task',frozen/'task.json','--config',frozen/'training.json','--output',out/'training'],out/'training.log')
            st=json.loads((out/'training/status.json').read_text());assert st['complete']
            checkpoint=(st['best_checkpoint'] or st['last_checkpoint']) if arm['endpoint']=='best_or_last' else st['last_checkpoint']
            panel=json.loads((frozen/'panel.json').read_text())
            for i,a in enumerate(arm['alphas']):
                for seed in panel['evaluation_seeds']:
                    status('evaluation',arm=arm['name'],alpha=a,seed=seed)
                    args=['learning/cli/disturbance.py','--task',frozen/'task.json','--panel',frozen/'panel.json','--training-run',out/'training','--checkpoint',checkpoint,'--output',out/'evaluation'/f'alpha_{i}'/f'seed_{seed}','--seed',seed]
                    if a is not None:args+=['--priority-alpha',a]
                    invoke(args,out/f'evaluation_{i}_{seed}.log',True)
            status('diagnostics',arm=arm['name']);invoke(['learning/cli/reward_breakdown.py','--run',out],out/'diagnostics.log',True)
            completed.append(dict(arm=arm['name'],checkpoint=checkpoint,training_steps=st['control_transitions']))
        status('analysis')
        from sttw_control.campaign_analysis import summarize
        summarize(plan_path)
        status('complete')
        (root/'INDEX.md').write_text('# Comparison campaign\n\n'+ '\n'.join(f"- [{x['arm']}]({x['arm']}/analysis/reward_breakdown/INDEX.md)" for x in completed)+'\n\nCompare physical metrics within matched tasks; returns across reward definitions are not comparable.\n')
    except Exception as exc:
        status('error',error=repr(exc));raise



def mode_training_documents(source_task, source_training_config, source_run, checkpoint,
                            *, updates=80, engineering=False):
    """Three isolated optimizers; same reward/dynamics and equal per-mode sample budget."""
    import copy
    if type(updates) is not int or not 1<=updates<=250:
        raise ValueError('independent-mode stage is bounded to 1..250 updates/mode')
    if source_task.get('tracking',{}).get('objective')!='soft_budget_v1':
        raise ValueError('mode separation must start from the frozen V1 task')
    source_run=str(Path(source_run).resolve());checkpoint=str(Path(checkpoint).resolve())
    arms=[]
    for alpha in (0.,.5,1.):
        task=copy.deepcopy(source_task);training=copy.deepcopy(source_training_config)
        task['priority'].update(randomize_alpha=False,fixed_alpha=alpha,training_alphas=[alpha],validation_alphas=[0.,.5,1.])
        training.update(resume_checkpoint=None,actor_init_checkpoint=checkpoint,actor_init_training=source_run,
            updates=2 if engineering else updates,training_reward_selection=True,training_reward_best_enabled=True,
            evaluation_reward_best=False,best_model_every_update=False,selection_incumbent=None,
            command_selection=False,command_patience=0,validation_updates=None)
        if engineering:
            training.update(num_envs=12,rollout_steps=8,epochs=1,minibatch_size=32,
                phase_spread_initialization=False,plot_interval=0,checkpoint_interval=2,
                warmup_steps=0,warmup_pool_size=0)
        # Keep formal optimizer/batch geometry unchanged. Do not silently triple a run.
        if not engineering and (training['num_envs'],training['rollout_steps'])!=(1024,128):
            raise ValueError('formal bounded mode experiment requires the inspected 1024x128 geometry')
        arms.append({'alpha':alpha,'name':f'alpha_{alpha:g}','task':task,'training':training})
    budget=sum(a['training']['num_envs']*a['training']['rollout_steps']*a['training']['updates'] for a in arms)
    if budget>98304000:raise ValueError('mode experiment exceeds authorized three-mode 250-update budget')
    return arms,budget


def pareto_indices(rows, *, require_working=False):
    """Non-dominated candidate indices, retaining all rejected candidates elsewhere."""
    import math
    valid=[]
    for i,r in enumerate(rows):
        finite=all(r.get(k) is not None and math.isfinite(r[k]) for k in ('speed_rmse','path_rmse'))
        admissible=r['final_hold'] and not r['failed'] and not r['deadline_missed']
        if require_working:admissible=admissible and r['working_guard_passed']
        if finite and admissible:valid.append(i)
    return [i for i in valid if not any(
        rows[j]['speed_rmse']<=rows[i]['speed_rmse'] and rows[j]['path_rmse']<=rows[i]['path_rmse']
        and (rows[j]['speed_rmse']<rows[i]['speed_rmse'] or rows[j]['path_rmse']<rows[i]['path_rmse'])
        for j in valid if j!=i)]


def probe_statistics(data, *, start_seconds, alpha, bias):
    """Post-command metrics from actually simulated full traces, not guessed profiles."""
    import numpy as np
    trace=data['trace'];c=data['config'];dt=c['controller']['dt']
    mask=np.rint(trace['time'][:-1]/dt).astype(int)>=round(start_seconds/dt)
    if not mask.any():
        # An early physical failure is evidence, never silently discarded/retried.
        return {'alpha':float(alpha),'bias':list(map(float,bias)),
            'speed_rmse':None,'path_rmse':None,'overspeed_peak':None,'roll_peak':None,
            'roll_excess_seconds':None,'overspeed_seconds':None,
            'failed':bool(trace['terminated'][-1]),'deadline_missed':bool(trace['return_state'][-1,7]),
            'final_hold':False,'working_guard_passed':False,
            'actual_transitions':len(trace['time'])-1,'comparison_window_observed':False,
            'role':'candidate ended before command window; retained as unsuccessful evidence'}
    ev=trace['true_forward_speed'][1:]-trace['reference_command'][:-1,0]
    ey=trace['path_features'][1:,0];roll=np.abs(trace['measurement'][1:,0]);tc=c['tracking']
    summary=data['summary']
    finite=bool(np.all(np.isfinite(ev)) and np.all(np.isfinite(ey)) and np.all(np.isfinite(roll)))
    full=trace['time'][-1]>=c['horizon_seconds']-dt*.1
    return {'alpha':float(alpha),'bias':list(map(float,bias)),
        'speed_rmse':float(np.sqrt(np.mean(ev[mask]**2))),
        'path_rmse':float(np.sqrt(np.mean(ey[mask]**2))),
        'overspeed_peak':float(np.maximum(ev[mask],0).max()),
        'roll_peak':float(roll[mask].max()),'roll_excess_seconds':float(dt*np.sum(roll[mask]>tc['roll_working_limit'])),
        'overspeed_seconds':float(dt*np.sum(ev[mask]>tc['overspeed_band'])),
        'failed':bool(trace['terminated'][-1]) or not finite,
        'deadline_missed':bool(trace['return_state'][-1,7]),
        'final_hold':bool(full and summary['terminal_tracking_hold']),
        'working_guard_passed':bool(finite and full and np.max(roll[mask])<=tc['roll_working_limit']
                                    and np.max(ev[mask])<=tc['overspeed_band']),
        'actual_transitions':len(trace['time'])-1,'comparison_window_observed':True,
        'role':'local open-loop residual offset then frozen feedback; NOT a deployable expert proof'}


def run_local_mode_probe(source_run, checkpoint, output, *, seed=49001, engineering=False):
    """Finite intervention family with exact EnvState preparation reuse.

    Reference, ECBC/ESO, physical actuators and residual authority are unchanged.
    Both final task gate and the STRICT 0.30/0.05 working gate are reported. Empty
    frontier means this local search did not find one, not global infeasibility.
    """
    import hashlib,importlib.util
    os.environ['JAX_PLATFORMS']='cpu'
    import numpy as np
    from dataclasses import asdict
    from sttw_control.env import RecoveryEnv,config_from_dict
    from sttw_control.network import make_policy_identity,load_policy
    from sttw_control.evaluation import evaluate
    from sttw_control.tracking_diagnostics import audit_trace,rescore_trace
    source_run=Path(source_run).resolve();checkpoint=Path(checkpoint).resolve();root=Path(output)
    if not checkpoint.is_relative_to(source_run/'checkpoints'):raise ValueError('probe checkpoint outside source training')
    if root.exists():raise ValueError('probe output exists; do not overwrite evidence')
    source=json.loads((source_run/'declaration.json').read_text());cfg=config_from_dict(source['task'])
    spec=importlib.util.spec_from_file_location('sttw_five_scene_config',Path('learning/cli/five_scene_review.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _,configs,warmups=module.review_configurations(cfg,seed,[0.,.5,1.])
    env=RecoveryEnv(configs['synthetic_turn'],backend='cpu');dt=env.config.controller.dt
    identity=make_policy_identity(env.bundle.identity,asdict(cfg),cfg.observation.history_steps)
    policy=load_policy(checkpoint,expected=identity)
    if not json.loads((source_run/'status.json').read_text()).get('complete'):raise ValueError('source stage incomplete')
    biases=(0.,) if engineering else (-.5,-.25,0.,.25,.5)
    pairs=[(x,y) for x in biases for y in biases]
    start=warmups['synthetic_turn'];duration=.5
    root.mkdir(parents=True);rows=[];cross=[]
    manifest={'checkpoint':str(checkpoint),'source':str(source_run),'seed':seed,'alphas':[0.,.5,1.],
        'bias_values':list(biases),'intervention_start':start,'intervention_duration':duration,
        'profile':'sin(pi*(time-start)/duration)^2, zero outside; clip ONLY final normalized residual to [-1,1]',
        'full_episode_control_budget':len(pairs)*3*env.horizon,
        'preparation_control_budget':round(env.config.preparation_seconds/dt),
        'source_actor_sha256':hashlib.sha256((checkpoint/'actor.msgpack').read_bytes()).hexdigest(),
        'role':'offline finite feasibility/ranking diagnostic, no training; original reference retained',
        'engineering':engineering,'complete':False}
    (root/'declaration.json').write_text(json.dumps(manifest,indent=2)+'\n')
    prepared=env.reset(seed)
    reference_qpos=np.asarray(prepared.data.qpos).copy()
    # env.step copies MjData and immutable JAX controller/history state; do NOT reset
    # just qpos/qvel or erase the prepared ESO/history between candidate rollouts.
    try:
        for alpha in (0.,.5,1.):
            for j,bias in enumerate(pairs):
                path=root/f'alpha_{alpha:g}'/f'candidate_{j:02d}'
                step_number=[0]
                def actor(obs):
                    t=step_number[0]*dt;step_number[0]+=1
                    if t<start:return np.zeros(2)
                    base=np.asarray(policy(obs))
                    pulse=np.sin(np.pi*(t-start)/duration)**2 if start<=t<start+duration else 0.
                    return np.clip(base+np.asarray(bias)*pulse,-1.,1.)
                identity_info={**identity,'checkpoint':str(checkpoint),'probe_bias':list(bias),
                    'scope':'frozen policy plus explicit temporary residual intervention'}
                evaluate(env,path,seed=seed,priority_alpha=alpha,policy=actor,policy_identity=identity_info,initial_state=prepared)
                np.testing.assert_array_equal(prepared.data.qpos,reference_qpos)
                data=audit_trace(path);record=probe_statistics(data,start_seconds=start,alpha=alpha,bias=bias)
                record['path']=str(path.relative_to(root));rows.append(record)
                for scoring_alpha in (0.,.5,1.):
                    scored=rescore_trace(data,scoring_alpha)
                    cross.append({'candidate':record['path'],'scoring_alpha':scoring_alpha,
                        'return':float(scored['trace']['reward'][1:].sum())})
                (root/'progress.json').write_text(json.dumps({'finished_candidates':len(rows),'expected':3*len(pairs),'last':record},indent=2)+'\n')
                print('probe',len(rows),'/',3*len(pairs),record['path'],'gate',record['working_guard_passed'],flush=True)
        final_set=pareto_indices(rows);strict_set=pareto_indices(rows,require_working=True)
        result={**manifest,'complete':True,'candidates':rows,'cross_scores':cross,
            'actual_control_transitions':sum(r['actual_transitions'] for r in rows),
            'final_gate_frontier':final_set,'strict_working_frontier':strict_set,
            'strict_gate_scope':'same 0.30rad and 0.05m/s working requirement for every alpha; physical failure threshold remains 0.70rad',
            'interpretation':'No candidate is an online expert certificate. Empty local frontier is not a proof of global infeasibility.'}
        (root/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        (root/'status.json').write_text(json.dumps({'complete':True,'phase':'complete','strict_frontier_count':len(strict_set),'final_frontier_count':len(final_set)},indent=2)+'\n')
        # Small evidence package; full candidate traces stay on the training computer.
        import zipfile
        with zipfile.ZipFile(root.with_name(root.name+'_summary.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for n in ('declaration.json','results.json','status.json'):z.write(root/n,n)
        return result
    except Exception as exc:
        (root/'status.json').write_text(json.dumps({'complete':False,'phase':'error','error':repr(exc)},indent=2)+'\n');raise


def run_independent_modes(source_run, checkpoint, output, *, updates=80, engineering=False):
    """Run three separate stock RSL trainers; no shared tensors/optimizer moments."""
    import hashlib
    from dataclasses import asdict
    from sttw_control.env import config_from_dict
    from sttw_control.training import TrainingConfig
    source_run=Path(source_run).resolve();checkpoint=Path(checkpoint).resolve();root=Path(output).resolve()
    source=json.loads((source_run/'declaration.json').read_text())
    if not json.loads((source_run/'status.json').read_text()).get('complete'):raise ValueError('source run incomplete')
    if not checkpoint.is_relative_to(source_run/'checkpoints'):raise ValueError('checkpoint outside declared source run')
    arms,budget=mode_training_documents(source['task'],source['training'],source_run,checkpoint,updates=updates,engineering=engineering)
    for arm in arms:config_from_dict(arm['task']);TrainingConfig(**arm['training'])
    repository=Path.cwd().resolve();root.mkdir(parents=True,exist_ok=False);(root/'frozen').mkdir()
    source_hash=hashlib.sha256((checkpoint/'actor.msgpack').read_bytes()).hexdigest()
    manifest={'schema':'sttw_independent_modes_v1','complete':False,'source_training':str(source_run),
        'source_checkpoint':str(checkpoint),'source_actor_sha256':source_hash,'formal_control_budget':budget,
        'same_initial_actor':True,'reward_changed':False,'mode_values':[0.,.5,1.],
        'selection':'predeclared final endpoint, initial source reported separately; no pooled reward best',
        'engineering':engineering,'modes':{},'scope':'three fully independent Actors/Critics/Adam states, sequential jobs; alpha remains input'}
    def publish():
        p=root/'modes.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n');q.replace(p)
    publish()
    env=dict(os.environ,PYTHONPATH=str(repository/'learning/src'),XLA_PYTHON_CLIENT_PREALLOCATE='false',PYTHONUNBUFFERED='1')
    if engineering:env['JAX_PLATFORMS']='cpu'
    else:
        env.pop('JAX_PLATFORMS',None)
    try:
        if not engineering:
            subprocess.run([sys.executable,'-c',
                'import jax,torch; assert any(d.platform=="gpu" for d in jax.devices()) and torch.cuda.is_available(), "formal mode training needs working CUDA JAX and Torch"'],
                check=True,env=env)
        for arm in arms:
            if hashlib.sha256((checkpoint/'actor.msgpack').read_bytes()).hexdigest()!=source_hash:raise ValueError('source Actor changed')
            name=arm['name'];task=root/'frozen'/f'{name}_task.json';training=root/'frozen'/f'{name}_ppo.json'
            task.write_text(json.dumps(arm['task'],indent=2)+'\n');training.write_text(json.dumps(arm['training'],indent=2)+'\n')
            out=root/name/'training';log=root/f'{name}.log'
            (root/'status.json').write_text(json.dumps({'phase':'training','alpha':arm['alpha'],'complete':False,'total_budget':budget},indent=2)+'\n')
            print('starting',name,'budget',arm['training']['num_envs']*arm['training']['rollout_steps']*arm['training']['updates'],'log',log,flush=True)
            args=[sys.executable,'learning/cli/train.py','--task',str(task),'--config',str(training),'--output',str(out)]
            with log.open('x') as f:subprocess.run(args,cwd=repository,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            st=json.loads((out/'status.json').read_text())
            if not st.get('complete'):raise RuntimeError('trainer exited without complete status')
            selected=Path(st['last_checkpoint']).resolve()
            expected=arm['training']['num_envs']*arm['training']['rollout_steps']*arm['training']['updates']
            if st['control_transitions']!=expected:raise ValueError('independent mode consumed a different training budget')
            init=json.loads((out/'actor_initialization.json').read_text())
            if init['actor_payload_sha256']!=source_hash:raise ValueError('mode did not start from the same frozen Actor')
            manifest['modes'][str(arm['alpha'])]={'alpha':arm['alpha'],'training':str(out),
                'checkpoint':str(selected),'update':arm['training']['updates'],
                'actor_payload_sha256':hashlib.sha256((selected/'actor.msgpack').read_bytes()).hexdigest(),
                'declaration_sha256':hashlib.sha256((out/'declaration.json').read_bytes()).hexdigest(),
                'control_transitions':st['control_transitions']}
            publish()
        manifest['complete']=True;publish()
        (root/'status.json').write_text(json.dumps({'phase':'complete','complete':True,'total_training_transitions':budget},indent=2)+'\n')
        return manifest
    except Exception as exc:
        (root/'status.json').write_text(json.dumps({'phase':'error','complete':False,'error':repr(exc)},indent=2)+'\n');raise


def run_mode_recovery(args):
    """Explicit stages, not an endless experiment queue. No code generation by Codex."""
    import hashlib
    root=args.output.resolve();source=args.source_training.resolve()
    if args.source_checkpoint is None:
        cp=Path(json.loads((source/'best_model.json').read_text())['checkpoint']).resolve()
    else:cp=args.source_checkpoint.resolve()
    if not cp.is_relative_to(source/'checkpoints'):raise ValueError('source checkpoint is outside training run')
    stages=['probe','specialize','review'] if args.stage=='all' else [args.stage]
    if args.stage=='all':
        # Probe/review initialize CPU JAX; training initializes CUDA. Never execute
        # them in one interpreter whose backend has already been chosen.
        for stage in stages:
            child=[sys.executable,str(Path(__file__).resolve()),'--mode-recovery',
                '--source-training',str(source),'--source-checkpoint',str(cp),
                '--output',str(root),'--stage',stage,'--updates-per-mode',str(args.updates_per_mode)]
            if args.engineering:child.append('--engineering')
            child_env=dict(os.environ,PYTHONPATH=str(Path.cwd()/'learning/src'))
            if stage in ('probe','review') or args.engineering:child_env['JAX_PLATFORMS']='cpu'
            else:child_env.pop('JAX_PLATFORMS',None)
            subprocess.run(child,check=True,env=child_env)
        return {'stages_executed':stages,'output':str(root),'scope':'separate CPU/CUDA processes; finite declared budget'}
    root.mkdir(parents=True,exist_ok=True)
    declaration={'source_training':str(source),'source_checkpoint':str(cp),
        'source_actor_sha256':hashlib.sha256((cp/'actor.msgpack').read_bytes()).hexdigest(),
        'updates_per_mode':args.updates_per_mode,'engineering':args.engineering,
        'interpretation':'feasibility probe and independent-mode isolation; neither promises task success'}
    d=root/'campaign.json'
    if d.exists() and json.loads(d.read_text())!=declaration:raise ValueError('existing campaign inputs differ')
    d.write_text(json.dumps(declaration,indent=2)+'\n')
    for stage in stages:
        if stage=='probe':run_local_mode_probe(source,cp,root/'probe',engineering=args.engineering)
        elif stage=='specialize':run_independent_modes(source,cp,root/'specialization',updates=args.updates_per_mode,engineering=args.engineering)
        else:
            bundle=root/'specialization/modes.json'
            if not bundle.exists() or not json.loads(bundle.read_text()).get('complete'):raise ValueError('finish specialization before review')
            if args.engineering:
                print('Engineering specialization finished; no formal scene evaluation requested.',flush=True);continue
            # Source and final composite are explicitly separate comparisons, same scenes/seeds.
            for kind,extra in [('source',[]),('modes',['--mode-bundle',str(bundle)])]:
                cmd=[sys.executable,'learning/cli/five_scene_review.py','--training',str(source),'--checkpoint',str(cp),
                    '--output',str(root/f'{kind}_review'),'--scenes','core_left','core_right','synthetic_turn','--compact',*extra]
                subprocess.run(cmd,check=True,env={**os.environ,'JAX_PLATFORMS':'cpu','PYTHONPATH':str(Path.cwd()/'learning/src')})
    return {'stages_executed':stages,'output':str(root),'scope':'explicitly requested finite stages'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--plan',type=Path)
    group.add_argument('--mode-recovery',action='store_true')
    p.add_argument('--source-training',type=Path);p.add_argument('--source-checkpoint',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--stage',choices=('probe','specialize','review','all'),default='probe')
    p.add_argument('--updates-per-mode',type=int,default=80)
    p.add_argument('--engineering',action='store_true')
    args=p.parse_args()
    if args.plan:run(args.plan)
    else:
        if args.source_training is None or args.output is None:p.error('--mode-recovery needs --source-training and --output')
        print(json.dumps(run_mode_recovery(args),indent=2))
