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
