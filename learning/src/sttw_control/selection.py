"""Development-only candidate acceptance and recovery-first ordering."""
import numpy as np


def rank_candidate(candidate,baseline,*,speed_slack=.01,nominal_slack=.01):
    arrays=[np.asarray(candidate[k],dtype=float) for k in ('radial_rmse','speed_rmse','nominal_radial_rmse','post_event_peak','settling_seconds')]
    if not all(np.isfinite(x).all() for x in arrays):return None,'nonfinite validation'
    reference=[np.asarray(baseline[k],dtype=float) for k in ('speed_rmse','nominal_radial_rmse')]
    if not all(np.isfinite(x).all() for x in reference):return None,'nonfinite baseline reference'
    if any(candidate['failed']) or any(candidate['nominal_failed']):return None,'physical failure'
    if np.any(np.asarray(candidate['speed_rmse'])>np.minimum(.2,np.asarray(baseline['speed_rmse'])+speed_slack)):
        return None,'speed regression against paired baseline'
    if np.any(np.asarray(candidate['nominal_radial_rmse'])>np.asarray(baseline['nominal_radial_rmse'])+nominal_slack):
        return None,'undisturbed tracking regression'
    event=np.asarray(candidate['event_present'],bool)
    if not event.any():return (0.,0.,float(np.mean(candidate['radial_rmse']))),'eligible'
    complete=np.asarray(candidate['post_event_hold_complete'],bool)[event]
    # Missing recovery is right-censored at the observation budget, never zero.
    return (float(np.mean(~complete)),float(np.mean(np.asarray(candidate['settling_seconds'])[event])),
            float(np.mean(np.asarray(candidate['post_event_peak'])[event]))),'eligible'


def rank_command_candidate(candidate, baseline, *, speed_slack=.05, yaw_slack=.05, scope="initial"):
    """Development rank, not a safety certificate; failed-only banks remain labelled."""
    if scope=='full_episode':
        keys=('speed_rmse','yaw_rmse','episode_return','failed','terminal_tracking_hold')
        if not all(np.isfinite(np.asarray(source[k],float)).all() for source in (candidate,baseline) for k in keys):
            return None,'nonfinite command validation'
        nominal=~np.asarray(baseline['failed'],bool)
        speed=np.asarray(candidate['speed_rmse'])>np.asarray(baseline['speed_rmse'])+speed_slack
        yaw=np.asarray(candidate['yaw_rmse'])>np.asarray(baseline['yaw_rmse'])+yaw_slack
        delta=np.asarray(candidate['episode_return'])-np.asarray(baseline['episode_return'])
        return (float(np.sum(candidate['failed'])),float(np.sum(nominal&(speed|yaw))),float(np.sum(~np.asarray(candidate['terminal_tracking_hold'],bool))),-float(np.mean(delta))),'failure count, full-episode regression on surviving baselines, terminal tracking misses, negative paired return gain; candidate rank is not task acceptance'
    if scope!='initial':raise ValueError('invalid command ranking scope')
    keys=('initial_speed_rmse','initial_yaw_rmse','episode_return')
    if not all(np.isfinite(np.asarray(candidate[k],float)).all() for k in keys):
        return None,'nonfinite command validation'
    speed=np.asarray(candidate['initial_speed_rmse'])>np.asarray(baseline['initial_speed_rmse'])+speed_slack
    yaw=np.asarray(candidate['initial_yaw_rmse'])>np.asarray(baseline['initial_yaw_rmse'])+yaw_slack
    return (float(np.sum(candidate['failed'])),float(np.sum(speed|yaw)), -float(np.mean(candidate['episode_return']))),'failure count, initial tracking regression count, negative mean episode return'


def command_improved(score,best,min_delta):
    if score is None:return False
    if best is None:return True
    return score[:-1]<best[:-1] or (score[:-1]==best[:-1] and score[-1]<best[-1]-min_delta)


def command_should_stop(update,stale,min_updates,patience):
    return patience>0 and update>=min_updates and stale>=patience


def refresh_best_reward_model(training_dir):
    """Maintain a best-return alias without changing stopping or acceptance rules.

    Saved fixed-development validations and a compatible resume checkpoint compete.
    Unevaluated checkpoints never inherit their rollout's training reward.
    """
    import hashlib,json,os
    from pathlib import Path
    root=Path(training_dir).resolve()
    declaration=root/'declaration.json'
    declared=json.loads(declaration.read_text()) if declaration.exists() else {}
    config=declared.get('training',{})
    metadata_paths=sorted((root/'checkpoints').glob('update_*/training.json'))
    resume=config.get('resume_checkpoint')
    if resume:
        checkpoint=Path(resume)
        parent=json.loads((checkpoint.parents[1]/'declaration.json').read_text())
        fields=('validation_seeds','command_validation_schedules')
        same_panel=all(parent['training'].get(k)==config.get(k) for k in fields)
        same_policy=parent.get('policy_identity') is not None and parent['policy_identity']==declared.get('policy_identity')
        if same_panel and same_policy:metadata_paths.append(checkpoint/'training.json')
    candidates=[]
    for metadata in metadata_paths:
        raw=metadata.read_bytes();entry=json.loads(raw);v=entry.get('validation')
        if not v:continue
        returns=np.asarray(v.get('episode_return',[]),float)
        if not returns.size or not np.isfinite(returns).all():continue
        candidates.append(dict(checkpoint=str(metadata.parent),update=entry['update'],mean_episode_return=float(returns.mean()),case_count=int(returns.size),failures=sum(v.get('failed',[])),metadata_sha256=hashlib.sha256(raw).hexdigest()))
    if not candidates:return None
    if len({c['case_count'] for c in candidates})!=1:raise ValueError('development panel sizes differ')
    chosen=max(candidates,key=lambda x:(x['mean_episode_return'],-x['update']))
    validation=json.loads((Path(chosen['checkpoint'])/'training.json').read_text())['validation']
    baseline=json.loads((root/'baseline_validation.json').read_text())
    geometric=declared.get('task',{}).get('tracking') is not None
    if geometric:
        rank,reason=rank_tracking_candidate(validation,baseline,speed_slack=config.get('selection_speed_slack',.05),nominal_slack=config.get('selection_nominal_slack',.03))
    else:
        rank,reason=rank_command_candidate(validation,baseline,scope='full_episode',speed_slack=config.get('command_speed_slack',.05),yaw_slack=config.get('command_yaw_slack',.05))
    result=dict(**chosen,criterion='maximum mean fixed-development episode return',scope='evaluated stage checkpoints plus compatible resume checkpoint; earlier update wins exact ties; no extra physics',development_gates_passed=bool(rank is not None and (geometric or all(x==0 for x in rank[:-1]))),acceptance_rank=rank,acceptance_reason=reason,task_success_verified=False,candidates=candidates)
    saved_updates={json.loads(p.read_text())['update'] for p in metadata_paths}
    evaluated_updates={c['update'] for c in candidates}
    result['selection_coverage']={'saved_updates':len(saved_updates),'evaluated_updates':len(evaluated_updates),'unevaluated_updates':sorted(saved_updates-evaluated_updates),'all_saved_updates_evaluated':saved_updates<=evaluated_updates,'scope':'this stage plus compatible resume checkpoint; not unrecorded historical stages or parameter-space optimum'}
    dest=root/'best_model'
    if dest.exists() and not dest.is_symlink():raise ValueError('best_model exists and is not a managed symlink')
    temporary=root/'.best_model.tmp'
    if temporary.is_symlink():temporary.unlink()
    temporary.symlink_to(os.path.relpath(chosen['checkpoint'],root),target_is_directory=True);temporary.replace(dest)
    tmp=root/'best_model.tmp.json';tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(root/'best_model.json')
    return result


def rank_tracking_candidate(candidate, baseline, *, speed_slack=.05, nominal_slack=.03):
    """Hard development acceptance before ranking; not a stability certificate."""
    keys=('radial_rmse','speed_rmse','heading_rmse','nominal_radial_rmse',
          'nominal_speed_rmse','nominal_heading_rmse','episode_return',
          'speed_tolerance_exceed_fraction','path_tolerance_exceed_fraction')
    timed_keys=('longitudinal_rmse','yaw_rate_rmse','nominal_longitudinal_rmse',
                'nominal_yaw_rate_rmse','longitudinal_tolerance_exceed_fraction',
                'yaw_rate_tolerance_exceed_fraction','nominal_longitudinal_tolerance_exceed_fraction',
                'nominal_yaw_rate_tolerance_exceed_fraction')
    timed=any(k in data for data in (candidate,baseline) for k in timed_keys)
    if timed:
        if not all(k in data for data in (candidate,baseline) for k in timed_keys):
            return None,'incomplete timed tracking validation'
        keys+=timed_keys
    if not all(np.isfinite(np.asarray(data[k],float)).all() for data in (candidate,baseline) for k in keys):
        return None,'nonfinite geometric validation'
    if any(candidate['failed']) or any(candidate['nominal_failed']):return None,'physical failure'
    if not all(candidate['terminal_tracking_hold']) or not all(candidate['nominal_terminal_tracking_hold']):
        return None,'final common tracking hold incomplete'
    if any(candidate['return_deadline_missed']) or any(candidate['nominal_return_deadline_missed']):
        return None,'declared return deadline missed'
    # Nominal paired baseline must itself survive; failure cannot become a cheap reference.
    if any(baseline['nominal_failed']):return None,'nominal baseline panel is not valid for non-regression'
    nominal_checks=(('nominal_speed_rmse',speed_slack),('nominal_radial_rmse',nominal_slack),
                    ('nominal_heading_rmse',nominal_slack))
    if timed:
        nominal_checks+=(('nominal_longitudinal_rmse',nominal_slack),
                         ('nominal_yaw_rate_rmse',nominal_slack))
    for key,slack in nominal_checks:
        if np.any(np.asarray(candidate[key])>np.asarray(baseline[key])+slack):
            return None,'undisturbed regression: '+key
    exceed_keys=('speed_tolerance_exceed_fraction','path_tolerance_exceed_fraction')
    if timed:exceed_keys+=timed_keys[4:]
    if any(np.any(np.asarray(candidate[k])>.1) for k in exceed_keys):
        return None,'relaxed tolerance exceeded for more than 10% of evaluated mature steps'
    delta=np.asarray(candidate['episode_return'])-np.asarray(baseline['episode_return'])
    return (-float(np.mean(delta)),float(np.mean(candidate['radial_rmse']))),'development gates passed; independent evaluation still required'


def record_training_reward_best(training_dir, checkpoint, policy_update, sampled_during_update, score):
    """Rank actual sampling policies by noisy training-batch mean step reward.

    A batch is generated BEFORE optimization. Never attach its reward to the
    post-update policy. This is not fixed-development or episode-return ranking.
    """
    import json, os
    from pathlib import Path
    root=Path(training_dir).resolve()
    checkpoint=Path(checkpoint).resolve()
    if not np.isfinite(score):raise ValueError('training reward must be finite')
    if not checkpoint.is_dir():raise ValueError('sampling checkpoint is missing')
    path=root/'best_model.json'
    prior=json.loads(path.read_text()) if path.exists() else None
    criterion='maximum sampled training mean step reward'
    if prior is not None and prior.get('criterion')!=criterion:
        raise ValueError('cannot mix development and training reward rankings')
    if prior is not None and score<=prior['mean_step_reward']:return prior
    result=dict(checkpoint=str(checkpoint),update=int(policy_update),
        sampled_during_update=int(sampled_during_update),mean_step_reward=float(score),
        criterion=criterion,scope='sampling policies in this continuation only; first observed wins ties; final post-update policy unscored',
        limitations='random short training batches, stochastic actions and inherited closed-loop state; not a fixed evaluation or complete-episode ranking',
        development_gates_passed=None,task_success_verified=False)
    dest=root/'best_model'
    if dest.exists() and not dest.is_symlink():raise ValueError('best_model is not a managed symlink')
    temporary=root/'.best_model.tmp'
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(os.path.relpath(checkpoint,root),target_is_directory=True)
    temporary.replace(dest)
    tmp=root/'best_model.tmp.json'
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    return result


def comparison_updates(saved, sampled_best=None, requested=None):
    """Declare the candidate subset before evaluation; include initial and LAST.

    It is legitimate for the first/zero model to win. The comparison never
    forces a later winner, silently excludes the last, or claims a global best.
    """
    saved=sorted(set(saved))
    if not saved:raise ValueError('no saved checkpoints')
    if requested is not None:
        wanted=sorted(set(requested))
        if not wanted or not set(wanted)<=set(saved):raise ValueError('requested checkpoint is not saved')
        return wanted
    wanted={saved[0],saved[-1]}
    if len(saved)>1:wanted.add(saved[1])
    for fraction in (1/3,2/3):wanted.add(saved[round((len(saved)-1)*fraction)])
    if sampled_best in saved:wanted.add(sampled_best)
    return sorted(wanted)


def fixed_comparison_rank(candidate,baseline,*,speed_slack=.05,nominal_slack=.03):
    """Eligible models first; diagnostic fallback does not become qualified."""
    rank,reason=rank_tracking_candidate(candidate,baseline,speed_slack=speed_slack,nominal_slack=nominal_slack)
    required=('failed','nominal_failed','terminal_tracking_hold','nominal_terminal_tracking_hold',
              'return_deadline_missed','nominal_return_deadline_missed','episode_return',
              'nominal_speed_rmse','nominal_radial_rmse','nominal_heading_rmse')
    lengths={len(candidate[k]) for k in required}|{len(baseline[k]) for k in required}
    if len(lengths)!=1 or not next(iter(lengths)):raise ValueError('fixed panel mismatch or empty evidence')
    if not all(np.isfinite(np.asarray(d[k],float)).all() for d in (candidate,baseline) for k in required):
        return None,False,'nonfinite fixed comparison'
    if rank is not None:return (0.,*rank),True,reason
    count=lambda k:float(np.sum(candidate[k]))
    misses=sum(len(candidate[k])-count(k) for k in ('terminal_tracking_hold','nominal_terminal_tracking_hold'))
    regress=np.zeros(len(candidate['failed']),bool)
    for key,slack in (('nominal_speed_rmse',speed_slack),('nominal_radial_rmse',nominal_slack),('nominal_heading_rmse',nominal_slack)):
        regress|=np.asarray(candidate[key])>np.asarray(baseline[key])+slack
    gain=float(np.mean(np.asarray(candidate['episode_return'])-np.asarray(baseline['episode_return'])))
    return (1.,count('failed')+count('nominal_failed'),float(regress.sum()),misses,
            count('return_deadline_missed')+count('nominal_return_deadline_missed'),-gain),False,reason


def compare_fixed_checkpoints(training_dir,output,*,updates=None,seeds=(51001,),event=None):
    """Explicit POST-training MJX evaluation, no optimizer or stochastic action.

    Shared reset seeds, complete state, alpha, reference and physical event are
    identical for every candidate. One force case plus paired nominal, three
    alphas per seed: six episodes/candidate, and one six-episode baseline batch.
    Training logs/aliases are NEVER overwritten. Standard panel stays separate.
    """
    import json,hashlib,time
    from pathlib import Path
    from dataclasses import replace,asdict
    import jax
    import jax.numpy as jp
    from flax import serialization
    from .env import RecoveryEnv,config_from_dict
    from .network import ResidualActor,make_policy_identity,load_policy
    from .training import TrainingConfig,normalization
    from .validation import make_validator
    from .tensorboard_logging import TrainingEvents
    root=Path(training_dir).resolve();out=Path(output).resolve()
    declared_path=root/'declaration.json';declared=json.loads(declared_path.read_text())
    if not json.loads((root/'status.json').read_text()).get('complete'):
        raise ValueError('fixed checkpoint comparison requires a complete training stage')
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int for s in seeds):raise ValueError('invalid evaluation seeds')
    cfg=config_from_dict(declared['task'])
    if cfg.tracking is None or cfg.timed_reference is None or cfg.timed_reference.mode!='geometry':
        raise ValueError('fixed comparison currently supports the explicit geometric task')
    saved={int(p.name.split('_')[-1]):p for p in (root/'checkpoints').glob('update_*') if (p/'actor.msgpack').exists()}
    bestpath=root/'best_model.json';sampled_best=json.loads(bestpath.read_text()).get('update') if bestpath.exists() else None
    chosen=comparison_updates(saved,sampled_best,updates)
    train=TrainingConfig(**declared['training'])
    event=dict(event) if event is not None else {'start':4.,'duration':1.,'force':-2.}
    if event['start']+event['duration']+cfg.tracking.return_seconds>cfg.horizon_seconds:
        raise ValueError('fixed comparison needs the full declared post-event window')
    train=replace(train,training_reward_selection=False,validation_seeds=tuple(seeds),validation_events=(event,))
    cfg=replace(cfg,priority=replace(cfg.priority,validation_alphas=(0.,.5,1.)))
    env=RecoveryEnv(cfg,backend='mjx')
    # The policy remains bound to its original task, not evaluation overrides.
    expected=make_policy_identity(env.bundle.identity,declared['task'],cfg.observation.history_steps)
    if expected!=declared['policy_identity']:raise ValueError('runtime/model identity changed; do not evaluate incompatible checkpoints')
    actor=ResidualActor(hidden_sizes=train.hidden_sizes,activation=train.activation)
    mean,std=normalization(cfg);validator=make_validator(env,actor,jp.asarray(std),train)
    def parameters(path):
        load_policy(path,expected=expected)  # verifies payload, dimensions and identity
        side=json.loads((path/'identity.json').read_text())
        if not np.array_equal(side['mean'],mean) or not np.array_equal(side['std'],std):raise ValueError('normalization changed')
        return jax.tree.map(jp.asarray,serialization.msgpack_restore((path/'actor.msgpack').read_bytes()))
    out.mkdir(parents=True,exist_ok=False)
    manifest={'schema':'sttw_fixed_checkpoint_comparison_v1','complete':False,
        'training_run':str(root),'training_declaration_sha256':hashlib.sha256(declared_path.read_bytes()).hexdigest(),
        'policy_identity':expected,'candidate_updates':chosen,'seeds':list(seeds),'alphas':[0.,.5,1.],
        'event':event,'episodes_per_candidate':6*len(seeds),'baseline_episodes':6*len(seeds),
        'maximum_control_transitions':(len(chosen)+1)*6*len(seeds)*env.horizon,
        'scope':'declared checkpoint subset on fixed random development cases; not all updates or independent held-out success',
        'selection_order':'eligible first; otherwise failures, nominal regressions, final hold misses, deadlines, negative paired return gain',
        'candidates':[]}
    def write():
        p=out/'selection.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n');temp.replace(p)
    def host(raw):return {k:np.where(np.isfinite(np.asarray(v)),np.asarray(v),None).tolist() for k,v in raw.items()}
    write();started=time.monotonic()
    try:
        first=parameters(saved[chosen[0]])
        baseline=host(validator({'actor':first},True));manifest['baseline']=baseline;write()
        selected=None
        with TrainingEvents(out/'tensorboard',profile='core',register=False) as events:
            for update in chosen:
                p=saved[update];value=host(validator({'actor':parameters(p)},False))
                rank,qualified,reason=fixed_comparison_rank(value,baseline,speed_slack=train.selection_speed_slack,nominal_slack=train.selection_nominal_slack)
                entry={'update':update,'checkpoint':str(p),'rank':rank,'development_gates_passed':qualified,
                       'reason':reason,'validation':value,'actor_sha256':hashlib.sha256((p/'actor.msgpack').read_bytes()).hexdigest()}
                manifest['candidates'].append(entry)
                if rank is not None and (selected is None or (rank,update)<(selected['rank'],selected['update'])):selected=entry
                events.write({'update':update,'validation':value,'selection':{'development_gates_passed':qualified},
                    'paired_episode_return':{'baseline':baseline['episode_return'],'candidate':value['episode_return'],
                     'delta':(np.asarray(value['episode_return'])-np.asarray(baseline['episode_return'])).tolist()}})
                print(json.dumps({'checkpoint':p.name,'qualified':qualified,'rank':rank,'reason':reason}),flush=True);write()
        if selected is None:raise RuntimeError('no finite comparison candidate')
        manifest.update(complete=True,selected_checkpoint=selected['checkpoint'],selected_update=selected['update'],
                        development_gates_passed=selected['development_gates_passed'],task_success_verified=False,
                        elapsed_seconds=time.monotonic()-started)
        write();return manifest
    except Exception as exc:
        manifest.update(error=repr(exc),complete=False);write();raise
