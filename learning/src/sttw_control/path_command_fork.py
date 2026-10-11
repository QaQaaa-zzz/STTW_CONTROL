"""Explicit distribution-only fork and fail-closed bounded extension decisions.

Normal exact-identity resume remains untouched. A fork proves the old identity
against the parent manifest before restoring the entire old learner state.
"""
from copy import deepcopy
import json
import pickle
from pathlib import Path
import numpy as np
from .preference_best import config_hash, digest
from .path_command_policy import SCHEMA

IDENTITY_KEYS=('schema','alpha','lower_actor_sha256','config_sha256','sampler_sha256','best_sha256','seed')
PLAN_ALLOWED={'design_config','updates','sampler_version','sampler_resolution','fork','conditional_extension','coverage_monitor'}


def _equal(actual, expected, label):
    if config_hash(actual)!=config_hash(expected):
        raise ValueError('fork frozen identity mismatch: '+label)


def validate_fork_contract(plan, cfg, source):
    """Resolve immutable best receipts and prove the narrow declared allowlist.

    Paths in ``plan['fork']`` must already be absolute (load_plan resolves them).
    File hashes and complete saved learner contents are audited on CPU; this
    function neither changes RNG nor creates an algorithm or simulator.
    """
    fork=plan['fork']
    if not all(Path(fork[k]).is_absolute() for k in ('parent_run','identities')):
        raise ValueError('fork paths must be resolved by load_plan')
    parent=Path(fork['parent_run']);manifest_path=parent/'manifest.json'
    manifest=json.loads(manifest_path.read_text());old_cfg=manifest['config'];old_plan=manifest['plan']
    _equal(source,manifest['source'],'complete source manifest')
    frozen=deepcopy(cfg)
    sampler=frozen.pop('training_sampler',None)
    if sampler is None or plan.get('sampler_version')!='demand_exposure_v2':
        raise ValueError('explicit demand_exposure_v2 sampler required')
    if frozen['training_future_phase_C'].get('validation_updates')!=[50,100,150,200]:
        raise ValueError('fork validation schedule must be 50/100/150/200')
    if 'validation_updates' in old_cfg['training_future_phase_C']:
        frozen['training_future_phase_C']['validation_updates']=old_cfg['training_future_phase_C']['validation_updates']
    else:
        frozen['training_future_phase_C'].pop('validation_updates',None)
    _equal(frozen,old_cfg,'complete config except sampler and validation schedule')
    _equal({k:v for k,v in plan.items() if k not in PLAN_ALLOWED},
           {k:v for k,v in old_plan.items() if k not in PLAN_ALLOWED},'plan outside allowlist')
    if plan['updates']!=100:
        raise ValueError('fork initial budget must be 100 new updates per endpoint')
    resolution=deepcopy(plan['sampler_resolution'])
    _equal(resolution.pop('training_sampler',None),sampler,'sampler resolution')
    _equal(resolution,old_plan['sampler_resolution'],'legacy sampler resolution')
    _equal(source['candidate']['actor_sha256'],fork['lower_actor_sha256'],'frozen lower')
    paired=json.loads(Path(fork['identities']).read_text())
    identities={};original={}
    for alpha in (0,1):
        name=f'alpha{alpha}';record=paired[name]['best_task']
        expected=fork['expected_parents'][name]
        for key in ('source_update','checkpoint_sha256','actor_sha256'):
            _equal(record[key],expected[key],name+' '+key)
        local=json.loads((parent/name/'best_task_model.json').read_text())
        # Full receipt equality prevents a copied actor hash hiding different
        # evaluation/config identity or an unrelated parent generation.
        _equal(record,local,name+' local immutable best receipt')
        identity=dict(schema=SCHEMA,alpha=alpha,lower_actor_sha256=source['candidate']['actor_sha256'],
                      config_sha256=config_hash(old_cfg),sampler_sha256=config_hash(old_plan['sampler_resolution']),
                      best_sha256=config_hash(old_plan['best_resolution']),seed=old_plan['seed'])
        _equal({k:record[k] for k in IDENTITY_KEYS},identity,name+' original full identity')
        _audit_saved_parent(record,identity)
        identities[name]=record;original[name]=identity
    return dict(parent_run=str(parent.resolve()),parent_manifest=manifest,
                parent_manifest_sha256=digest(manifest_path),identities=identities,
                original_identities=original,physical_continuation='fresh episodes from unchanged complete prepared bank; not frame-continuous')


def _audit_saved_parent(record, identity):
    import torch
    for key in ('checkpoint','actor'):
        if digest(record[key])!=record[key+'_sha256']:
            raise ValueError('fork parent '+key+' SHA mismatch')
    saved=torch.load(record['checkpoint'],map_location='cpu',weights_only=False)
    required={'policy','optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','update','identity','schema','terminal_contract'}
    if not required.issubset(saved):raise ValueError('fork parent lacks complete learner or RNG state')
    _equal(saved['identity'],identity,'serialized original identity')
    if saved['schema']!=SCHEMA or saved['terminal_contract']!='terminal_contract_v2' or saved['update']!=record['source_update']:
        raise ValueError('fork parent schema/update/terminal mismatch')
    state=saved['policy'];optimizer=saved['optimizer']
    if 'log_std' not in state or not any(k.startswith('critic.') for k in state):
        raise ValueError('fork parent lacks critic or log_std')
    if not optimizer.get('state') or not optimizer.get('param_groups'):
        raise ValueError('fork parent lacks trained matching optimizer')
    if not all({'step','exp_avg','exp_avg_sq'}.issubset(v) for v in optimizer['state'].values()):
        raise ValueError('fork parent has incomplete Adam moments')
    with Path(record['actor']).open('rb') as stream:actor=pickle.load(stream)['params']
    weights=[(k,v) for k,v in state.items() if k.startswith('actor.') and k.endswith('.weight')]
    weights.sort(key=lambda item:int(item[0].split('.')[1]))
    if len(weights)!=len(actor):raise ValueError('fork actor layer mismatch')
    for i,(key,value) in enumerate(weights):
        expected=actor[f'Dense_{i}']
        if not np.array_equal(value.cpu().numpy().T,np.asarray(expected['kernel'])) or not np.array_equal(state[key.replace('.weight','.bias')].cpu().numpy(),np.asarray(expected['bias'])):
            raise ValueError('fork actor and full learner are not the same generation')
    return saved


def restore_fork(algo, receipt, alpha):
    """Restore using ORIGINAL full identity, including optimizer and all RNGs."""
    from .path_command_training import restore_checkpoint
    name=f'alpha{alpha}';record=receipt['identities'][name];identity=receipt['original_identities'][name]
    _audit_saved_parent(record,identity)
    saved=restore_checkpoint(record['checkpoint'],algo,identity)
    for key,expected in saved['policy'].items():
        if not np.array_equal(algo.policy.state_dict()[key].detach().cpu().numpy(),expected.detach().cpu().numpy()):
            raise ValueError('restored learner differs from original parent')
    return saved


def _finite(value, label):
    result=float(value)
    if not np.isfinite(result):raise ValueError('nonfinite '+label)
    return result


def _matched_high_speed_error(parent, current, alpha, cfg):
    """Same metric/window definition for both models, retaining arrival times.

    Prefer the task's chi>0 normalized primary RMSE on both traces. If either
    trace has no chi>0 samples, use abs(path_curvature)>1e-4 for both, never a
    full-episode substitute. This fallback is explicitly recorded per case.
    """
    from .path_command_selection import cases
    names=[name for name,route,speed in cases(cfg) if route!='straight' and speed==cfg['stage_C_evaluation']['conflict_speed_m_s']]
    if not names:raise ValueError('no matched high speed turn cases')
    metrics={}
    for name in names:
        pair=[parent[name],current[name]]
        masks=[np.asarray(d['chi'])>0 for d in pair];window='chi>0'
        if not all(mask.any() for mask in masks):
            masks=[np.abs(np.asarray(d['path_curvature']))>1e-4 for d in pair];window='abs(path_curvature)>1e-4 for both traces'
        values=[]
        for d,mask in zip(pair,masks):
            if not mask.any():raise ValueError('empty high speed turn window '+name)
            error=np.asarray(d['path_cross_track'],float)/.10 if alpha==0 else (np.asarray(d['actual_forward_speed'],float)-np.asarray(d['v_user'],float))/.05
            if error.shape!=mask.shape or not np.isfinite(error).all():raise ValueError('invalid matched primary error '+name)
            values.append(_finite(np.sqrt(np.mean(error[mask]**2)),name))
        metrics[name]=dict(window=window,parent=values[0],current=values[1],parent_samples=int(masks[0].sum()),current_samples=int(masks[1].sum()))
    return dict(metric='normalized RMSE: alpha0 ey/.10; alpha1 (v-v_user)/.05',cases=metrics,
                parent_worst=max(v['parent'] for v in metrics.values()),current_worst=max(v['current'] for v in metrics.values()))


def _safety(traces,cfg):
    threshold=cfg['stage_B_baseline']['working_roll_rad'];dt=cfg['timing']['lower_dt_s'];peak=0.;seconds=0.;integral=0.
    for d in traces.values():
        roll=np.asarray(d['peak_roll'],float)
        if not roll.size or not np.isfinite(roll).all():raise ValueError('invalid safety trace')
        peak=max(peak,float(roll.max()));seconds+=float(np.sum(roll>threshold))*dt
        integral+=float(np.maximum(roll-threshold,0).sum())*dt
    return dict(peak_roll_rad=peak,exceedance_seconds_total=seconds,exceedance_integral_rad_s=integral)


def extension_decision(parent_summary,current_summary,parent_traces,current_traces,statistics_previous,statistics_latest,coverage_fraction,alpha,cfg):
    """JSON decision only. Caller enforces 100/150 boundaries and hard cap 200.

    Statistics must represent two complete consecutive 25-update windows; rate
    fields use the same active-transition denominator in both. Missing evidence
    rejects extension, including absent strong-demand cost/error observations.
    """
    checks={};evidence={}
    try:
        if alpha not in (0,1):raise ValueError('alpha must be 0 or 1')
        from .path_command_selection import cases
        names={name for name,_,_ in cases(cfg)}
        for summary,traces in ((parent_summary,parent_traces),(current_summary,current_traces)):
            if len(names)!=6 or set(summary['cases'])!=names or set(traces)!=names or not summary['all_declared_traces_complete']:
                raise ValueError('complete matched six-case DEV required')
        checks['coverage_at_least_five_percent']=.05<=_finite(coverage_fraction,'coverage')<=1
        checks['all_six_goal_and_final_hold']=all(x['goal'] and x['final_hold'] for x in current_summary['cases'].values()) and current_summary['goal_hold_failures']==0
        checks['dev_task_failures_not_worse']=current_summary['task_failures']<=parent_summary['task_failures']
        for s in (statistics_previous,statistics_latest):
            if s['window_updates']!=25:raise ValueError('two complete 25-update statistics windows required')
            if s['windows']['strong_demand']['samples']<=0:raise ValueError('no strong-demand observations')
        for field in ('engineering_failure_rate','domain_failure_rate'):
            previous=_finite(statistics_previous[field],field);latest=_finite(statistics_latest[field],field)
            if min(previous,latest)<0:raise ValueError('negative failure rate')
            checks[field+'_not_worse']=latest<=previous
        metric=_matched_high_speed_error(parent_traces,current_traces,alpha,cfg);evidence['high_speed_main_error']=metric
        old_safety=_safety(parent_traces,cfg);new_safety=_safety(current_traces,cfg)
        evidence['safety']=dict(parent=old_safety,current=new_safety)
        checks['safety_peak_and_cumulative_exceedance_not_worse']=all(new_safety[k]<=old_safety[k]+1e-12 for k in old_safety)
        work_better=current_summary['working_failures']<parent_summary['working_failures']
        main_better=metric['parent_worst']>0 and metric['current_worst']<=.95*metric['parent_worst']
        checks['dev_working_failures_or_five_percent_main_error_improvement']=work_better or (current_summary['working_failures']<=parent_summary['working_failures'] and main_better)
        field='lateral_rmse_m' if alpha==0 else 'speed_rmse_m_s'
        old=statistics_previous['windows']['strong_demand'];new=statistics_latest['windows']['strong_demand']
        old_error=_finite(old[field],field);new_error=_finite(new[field],field)
        trend=new_error<old_error
        if 'costs' in old and 'costs' in new:
            old_cost=_finite(old['costs']['primary']['effective_mean'],'strong primary cost')
            new_cost=_finite(new['costs']['primary']['effective_mean'],'strong primary cost')
            trend=trend or new_cost<old_cost
            evidence['strong_primary_cost']=dict(previous=old_cost,latest=new_cost)
        evidence['strong_main_error']=dict(metric=field,previous=old_error,latest=new_error)
        checks['recent_strong_main_error_or_cost_improves']=trend
    except (KeyError,ValueError,TypeError,IndexError) as exc:
        return dict(allow=False,reasons=['missing_or_invalid_evidence: '+str(exc)],checks=checks,evidence=evidence)
    return dict(allow=all(checks.values()),reasons=[key for key,value in checks.items() if not value],checks=checks,evidence=evidence)
