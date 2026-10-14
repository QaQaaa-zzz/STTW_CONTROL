"""Fixed-path final comparison: actual geometry first, all commands unfiltered."""
from pathlib import Path
import json
import numpy as np

def _interval_context(d, config):
    """Rebuild rates using four-tick endpoints, never copied logged costs/rates.

    New traces repeat raw interval boundaries on every row, including partial
    terminal intervals. Legacy traces must be a contiguous prefix from reset;
    reset offsets are zero by the environment contract. A sliced live block
    without its endpoints is insufficient evidence.
    """
    dt=config['timing']['lower_dt_s'];upper_dt=config['timing']['upper_dt_s']
    times=np.asarray(d['time'],float);n=len(times)
    if not n or not np.isfinite(times).all():raise ValueError('empty or nonfinite trace time')
    ticks=np.rint(times/dt).astype(np.int64)
    if (ticks<0).any() or not np.allclose(times/dt,ticks,rtol=0,atol=.002) or not np.all(np.diff(ticks)==1):
        raise ValueError('trace must contain contiguous actual lower ticks')
    if not np.isclose(upper_dt,4*dt,rtol=0,atol=1e-12):raise ValueError('four-tick interval timing mismatch')
    offsets=np.asarray(d['offsets'],float)
    if offsets.shape!=(n,2) or not np.isfinite(offsets).all():raise ValueError('invalid raw offset measurements')
    failure=np.broadcast_to(np.asarray(d.get('physical_failure',False),bool),(n,)) | np.broadcast_to(np.asarray(d.get('tracking_domain_failure',d.get('domain_exit',False)),bool),(n,))
    starts=ticks//4*4;rate=np.empty((n,2));failed=np.zeros(n,bool);ends=np.empty(n,np.int64)
    fields=('interval_offset_before','interval_offset_end','interval_start_tick','interval_active_ticks','interval_failed')
    supplied=[k in d for k in fields]
    if any(supplied) and not all(supplied):raise ValueError('incomplete interval boundary context')
    if all(supplied):
        before=np.asarray(d[fields[0]],float);after=np.asarray(d[fields[1]],float)
        logged_starts=np.asarray(d[fields[2]]);counts=np.asarray(d[fields[3]]);terminal=np.asarray(d[fields[4]],bool)
        if before.shape!=(n,2) or after.shape!=(n,2) or any(x.shape!=(n,) for x in (logged_starts,counts,terminal)):
            raise ValueError('invalid interval boundary context shape')
        if not np.isfinite(before).all() or not np.isfinite(after).all() or not np.array_equal(logged_starts,starts):
            raise ValueError('invalid interval boundary context')
        if not np.isfinite(counts).all() or not np.all((counts>=1)&(counts<=4)&(counts==np.rint(counts))):
            raise ValueError('invalid interval active tick count')
        for start in np.unique(starts):
            ix=np.flatnonzero(starts==start);first=ix[0]
            for value in (before,after,counts,terminal):
                if not np.all(value[ix]==value[first]):raise ValueError('inconsistent interval boundary context')
            last_tick=int(start+counts[first]-1)
            if (ticks[ix]>last_tick).any():raise ValueError('trace extends past interval endpoint')
            endpoint=np.flatnonzero(ticks==last_tick)
            if endpoint.size and not np.allclose(offsets[endpoint[0]],after[first],rtol=1e-5,atol=1e-7):
                raise ValueError('raw offset endpoint disagrees with interval context')
            predecessor=np.flatnonzero(ticks==start-1)
            if predecessor.size and not np.allclose(offsets[predecessor[0]],before[first],rtol=1e-5,atol=1e-7):
                raise ValueError('raw offset before interval disagrees with context')
            # Explicit scoring context also preserves v1 domain engineering faults:
            # only v2 treats a finite domain exit as a replaced failure interval.
            if np.any(np.broadcast_to(np.asarray(d.get('physical_failure',False),bool),(n,))[ix]) and not terminal[first]:
                raise ValueError('physical failure disagrees with interval context')
            rate[ix]=(after[first]-before[first])/upper_dt
            failed[ix]=terminal[first];ends[ix]=last_tick
    else:
        if ticks[0]!=0:raise ValueError('slice is missing interval boundary context before its first tick')
        for start in np.unique(starts):
            ix=np.flatnonzero(starts==start)
            if len(ix)<4 and not failure[ix[-1]]:
                raise ValueError('partial live interval is missing endpoint context')
            before=np.zeros(2) if start==0 else offsets[ix[0]-1]
            rate[ix]=(offsets[ix[-1]]-before)/upper_dt
            failed[ix]=np.any(failure[ix]);ends[ix]=ticks[ix[-1]]
    return dict(offset_rate=rate,interval_start_tick=starts,interval_failed=failed,
                interval_end_tick=ends,ticks=ticks)


def reconstruct_costs(d,alpha,config):
    """Independent NumPy reconstruction of all ten costs from raw measurements."""
    c=config['path_reward'];version=c.get('reward_version')
    if 'reward_version' in c and version not in ('path_safety_preference_v1','path_safety_preference_v2'):
        raise ValueError(f'unknown path reward version: {version!r}')
    if version in ('path_safety_preference_v1','path_safety_preference_v2') and 'phi' not in d:
        raise ValueError('signed roll phi is required for path_safety_preference_v1')
    multiplier=1.
    if version=='path_safety_preference_v2':
        try:multiplier=float(c['primary_speed_multiplier'])
        except (KeyError,TypeError,ValueError) as e:raise ValueError('v2 requires finite positive primary speed multiplier') from e
        if not np.isfinite(multiplier) or multiplier<=0:raise ValueError('v2 requires finite positive primary speed multiplier')
    interval=_interval_context(d,config);n=len(interval['ticks'])
    def measurement(key):
        value=np.asarray(d[key],float)
        if value.shape!=(n,) or not np.isfinite(value).all():raise ValueError(f'invalid raw measurement {key}')
        return value
    chi=measurement('chi');speed=measurement('actual_forward_speed')
    ev=speed-measurement('v_user');ey=measurement('path_cross_track');eh=measurement('path_heading_error')
    peak=measurement('peak_roll');roll_rate=measurement('phi_dot');offsets=np.asarray(d['offsets'],float)
    h=lambda x:np.where(np.abs(x)<=1,x*x,2*np.abs(x)-1)
    pos=lambda x:np.maximum(x,0)
    raw=dict(speed=(10-9*chi*(1-alpha))*h(ev/c['speed_scale_m_s']),
        path=(10-9*chi*alpha)*(h(ey/c['cross_track_scale_m'])+c['path_heading_factor']*h(eh/c['path_heading_scale_rad'])),
        primary=c['primary_weight']*chi*((1-alpha)*h(pos(np.abs(ey)-c['primary_path_band_m'])/c['primary_path_scale_m'])+multiplier*alpha*h(pos(np.abs(ev)-c['primary_speed_band_m_s'])/c['primary_speed_scale_m_s'])),
        roll=c['roll_weight']*h(pos(peak-c['roll_start_rad'])/c['roll_scale_rad']),
        working_roll=c['working_roll_weight']*h(pos(peak-c['working_roll_start_rad'])/c['working_roll_scale_rad']),
        roll_rate=c['roll_rate_weight']*h(pos(np.abs(roll_rate)-c['roll_rate_start_rad_s'])/c['roll_rate_scale_rad_s']),
        overspeed=c['overspeed_weight']*h(pos(ev-c['overspeed_band_m_s'])/c['overspeed_scale_m_s']),
        low_speed=c['low_speed_weight']*h(pos(c['low_speed_start_m_s']-speed)/c['low_speed_scale_m_s']),
        offset_magnitude=c['offset_magnitude_weight']*np.sum((offsets/np.asarray(c['offset_magnitude_scales']))**2,axis=1),
        offset_rate=c['offset_rate_weight']*np.sum((interval['offset_rate']/np.asarray(c['offset_rate_scales']))**2,axis=1))
    diagnostic={k:np.zeros(n) for k in ('predictive_roll','working_exposure','corridor')}
    if version in ('path_safety_preference_v1','path_safety_preference_v2'):
        phi=measurement('phi');rho=np.maximum(np.abs(phi),np.abs(phi+c['risk_prediction_s']*roll_rate))
        diagnostic=dict(predictive_roll=c['predictive_roll_weight']*h(pos(rho-c['predictive_roll_start_rad'])/c['predictive_roll_scale_rad']),
            working_exposure=c['working_exposure_weight']*np.clip((peak-.30)/c['working_exposure_width_rad'],0,1),
            corridor=c['corridor_weight']*h(pos(np.abs(ey)-c['corridor_band_m'])/c['corridor_scale_m']))
        raw['roll']+=diagnostic['predictive_roll'];raw['working_roll']+=diagnostic['working_exposure'];raw['path']+=diagnostic['corridor']
    effective={k:np.minimum(v,c['component_caps'][k]) for k,v in raw.items()}
    return dict(raw=raw,effective=effective,capped={k:v>c['component_caps'][k] for k,v in raw.items()},
                diagnostic_subcosts=diagnostic,**interval)


def reward_series(d,alpha,config,horizon=None):
    """Same-alpha reweighting from raw physics; one original failure tail."""
    rebuilt=reconstruct_costs(d,alpha,config);c=config['path_reward'];dt=config['timing']['lower_dt_s']
    components={k:-c['scale']*dt*v for k,v in rebuilt['effective'].items()}
    components['failure_tail']=np.zeros(len(rebuilt['ticks']))
    failed=rebuilt['interval_failed']
    for value in components.values():value[failed]=0
    terminal=failed & (rebuilt['ticks']==rebuilt['interval_end_tick'])
    h=config['training_future_phase_C']['episode_s'] if horizon is None else horizon
    gamma=config['training_future_phase_C']['gamma']
    remaining=np.ceil(h/config['timing']['upper_dt_s'])-rebuilt['interval_start_tick'][terminal]//4
    components['failure_tail'][terminal]=-c['failure_extra']-c['scale']*config['timing']['upper_dt_s']*c['failure_cost_rate_bound']*(1-gamma**remaining)/(1-gamma)
    return sum(components.values()),components

def _rms(x):
    x=np.asarray(x,float)
    return float(np.sqrt(np.mean(x*x))) if x.size else None


def paired_path_metrics(data,path,config,bin_width_m=.5):
    """Descriptive fixed-world metrics. Different PP nominal streams are expected."""
    dt=config['timing']['lower_dt_s'];b=config['stage_B_baseline'];h=config['training_future_phase_C']['episode_s']
    turn_end=float(path['turn_end']);curved=np.flatnonzero(np.abs(path['curvature'])>1e-8)
    turn_start=float(path['s'][curved[0]]) if len(curved) else None
    result={'window_definition':'spatial turn from fixed-path nonzero curvature extent; strong demand chi>=.5; no time shifts',
            'bin_width_m':bin_width_m,'methods':{},'pairing':{}}
    for method,d in data.items():
        n=len(d['time']);progress=np.asarray(d['path_progress']);speed=np.asarray(d['actual_forward_speed']);ev=speed-np.asarray(d['v_user']);ey=np.asarray(d['path_cross_track']);eh=np.asarray(d['path_heading_error']);roll=np.asarray(d['peak_roll']);chi=np.asarray(d['chi']);t=np.asarray(d['time'])+dt
        if not n or not all(np.isfinite(x).all() for x in [progress,speed,ev,ey,eh,roll]):raise ValueError('nonfinite or empty paired evidence')
        physical=bool(np.any(d['physical_failure']));domain=bool(np.any(d.get('tracking_domain_failure',d.get('domain_exit',False))))
        if np.any(d.get('policy_fault',False)) or np.any(d.get('lower_fault',False)):raise ValueError('engineering fault is invalid paired evidence')
        if not (physical or domain) and n!=round(h/dt):raise ValueError('nonterminal truncated paired evidence')
        turn=np.zeros(n,bool) if turn_start is None else (progress>=turn_start)&(progress<=turn_end)
        entry=np.zeros(n,bool) if turn_start is None else (progress>=max(0,turn_start-1))&(progress<turn_start)
        entry_speed=float(np.mean(speed[entry])) if entry.any() else None
        masks={'whole':np.ones(n,bool),'spatial_turn':turn,'strong_demand':chi>=.5,'exit':(progress>turn_end) if turn_start is not None else np.zeros(n,bool)}
        def window(mask):
            if not mask.any():return {'samples':0,'status':'N/A'}
            return dict(samples=int(mask.sum()),seconds=float(mask.sum()*dt),speed_rmse_m_s=_rms(ev[mask]),path_rmse_m=_rms(ey[mask]),heading_rmse_rad=_rms(eh[mask]),mean_underspeed_m_s=float(np.maximum(-ev[mask],0).mean()),minimum_actual_speed_m_s=float(speed[mask].min()),mean_actual_speed_m_s=float(speed[mask].mean()),mean_speed_change_from_pre_turn_m_s=None if entry_speed is None else float(speed[mask].mean()-entry_speed),minimum_speed_change_from_pre_turn_m_s=None if entry_speed is None else float(speed[mask].min()-entry_speed),peak_roll_rad=float(roll[mask].max()),working_exceedance_s=float((roll[mask]>b['working_roll_rad']).sum()*dt))
        tail=t>h-b['hold_s']+1e-5
        hold=bool(tail.sum()==round(b['hold_s']/dt) and np.all(abs(ey[tail])<=b['final_y_error_m']) and np.all(abs(eh[tail])<=b['final_path_heading_error_rad']) and np.all(abs(ev[tail])<=b['final_speed_error_m_s']) and not(physical or domain))
        goal=bool(turn_start is None or np.any((np.asarray(d['goal_section_signed_distance'])>=0)&(progress>=np.asarray(d['goal_progress']))))
        governed=np.asarray(d['governed']);nominal=np.asarray(d['nominal']);actual=np.column_stack([speed,d['actual_delta']]);lower=actual-governed
        bins=[]
        for lo in np.arange(0,max(progress.max(),bin_width_m),bin_width_m):
            bins.append(dict(s_start_m=float(lo),s_end_m=float(lo+bin_width_m),**window((progress>=lo)&(progress<lo+bin_width_m))))
        stable=[];width=round(.5/dt)
        # Conservative descriptive constancy definition, not a new controller limit.
        # A qualifying plateau precedes a further >.5s persistent mismatch.
        for start in range(0,max(0,n-2*width),width):
            end=start+2*width+1;segment=governed[start:end]
            if np.all(np.ptp(segment,axis=0)<=np.array([.02,.01])):
                a=lower[start+width:end];bad=[bool(np.all(abs(a[:,0])>.15)),bool(np.all(abs(a[:,1])>.04))]
                stable.append(dict(start_s=float(t[start]),end_s=float(t[end-1]),speed_mismatch=bad[0],steer_mismatch=bad[1]))
        mismatches=[x for x in stable if x['speed_mismatch'] or x['steer_mismatch']]
        outcome_failure=physical or domain or not(goal and hold) or bool(np.any(roll>b['working_roll_rad']))
        lower_evidence=dict(status='N/A' if not stable else ('candidate_association_only' if mismatches and outcome_failure else 'no_blocking_evidence'),stable_definition='governed range <=.02m/s and <=.01rad for >1s, first .5s settling; thresholds descriptive',stable_windows=len(stable),persistent_mismatch_windows=mismatches,causal_blocking_proved=False,reason='No stable governed window' if not stable else 'Concurrent mismatch and task failure require temporal/actuator inspection; imperfect RMSE alone is not a blocker')
        command_stats={key:dict(mean=np.asarray(d[key]).mean(axis=0).tolist(),minimum=np.asarray(d[key]).min(axis=0).tolist(),maximum=np.asarray(d[key]).max(axis=0).tolist()) for key in ['target_offset','filtered_offset','offsets','governed','nominal']}
        result['methods'][method]=dict(windows={k:window(v) for k,v in masks.items()},pre_turn_speed_m_s=entry_speed,physical_failure=physical,tracking_domain_failure=domain,observed_seconds=float(n*dt),goal_crossed=goal,final_hold=hold,final_progress_m=float(progress[-1]),peak_roll_rad=float(roll.max()),working_exceedance_s=float((roll>b['working_roll_rad']).sum()*dt),actual_minus_governed_rmse=[_rms(lower[:,i]) for i in (0,1)],governed_minus_nominal_rmse=[_rms((governed-nominal)[:,i]) for i in (0,1)],commands=command_stats,arclength_bins=bins,lower_blocking=lower_evidence)
    for window in ['whole','spatial_turn','strong_demand','exit']:
        a=result['methods']['alpha0']['windows'][window];b1=result['methods']['alpha1']['windows'][window]
        if not a['samples'] or not b1['samples']:result['pairing'][window]={'status':'N/A'};continue
        dv=a['speed_rmse_m_s']-b1['speed_rmse_m_s'];dy=b1['path_rmse_m']-a['path_rmse_m']
        result['pairing'][window]=dict(D_v_m_s=dv,D_y_m=dy,direction_correct=bool(dv>0 and dy>0),discernible_reference=bool(dv>=.03 and dy>=.02),scope='descriptive different occupied-time weights; compare fixed arc bins too')
    # Each method contributes equally per commonly occupied fixed spatial bin.
    bins0=result['methods']['alpha0']['arclength_bins'];bins1=result['methods']['alpha1']['arclength_bins'];common=[(a,b1) for a,b1 in zip(bins0,bins1) if a['samples'] and b1['samples']]
    result['pairing']['common_arclength_bins']=dict(count=len(common),D_v_m_s=None if not common else float(np.mean([a['speed_rmse_m_s']-b1['speed_rmse_m_s'] for a,b1 in common])),D_y_m=None if not common else float(np.mean([b1['path_rmse_m']-a['path_rmse_m'] for a,b1 in common])),scope='equal .5m bin weight; shared occupied bins only; missing/failed progress retained above')
    return result


def convergence_decision(validation_history,training_trend=None,paired_qualified=False):
    """Evidence-gated recommendation only; never authorizes more training."""
    history=sorted(validation_history,key=lambda x:x['update']);training_trend=training_trend or {}
    if not history or history[-1]['update']<200:return dict(action='continue_authorized_200',reason='200-update endpoint budget incomplete',automatic_extension=False)
    if paired_qualified and history[-1]['summary'].get('qualified_task'):
        return dict(action='freeze_and_propose_up_to_3_new_paths',reason='paired task/preference qualification supplied',automatic_extension=False)
    if len(history)<2:return dict(action='insufficient_late_validation',automatic_extension=False)
    def counts(x):
        s=x['summary'];return tuple(s['score_tuple'][:4])
    def quality(x):return float(x['summary']['score_tuple'][-1])
    previous,last=history[-2:];improvement=(quality(previous)-quality(last))/max(abs(quality(previous)),1e-12)
    train=training_trend.get('matched_condition_relative_improvement')
    if (counts(last)<counts(previous) or improvement>=.02) and train is not None and train>0:
        return dict(action='propose_100_more_requires_authorization',budget_updates=100,late_quality_relative_improvement=improvement,automatic_extension=False)
    if len(history)>=3 and train is not None:
        late=history[-3:];quality_change=(quality(late[0])-min(quality(x) for x in late))/max(abs(quality(late[0])),1e-12)
        if all(counts(x)>=counts(late[0]) for x in late[1:]) and quality_change<.01 and abs(train)<.02:
            return dict(action='plateau_identify_one_main_problem',reason='three late points without count/quality progress and matched training trend flat',automatic_extension=False)
    return dict(action='inspect_coverage_caps_and_task_trends',reason='no evidence sufficient to authorize extension; reward alone is insufficient',late_quality_relative_improvement=improvement,matched_training_trend=train,automatic_extension=False)


def checked_final_sources(root,config=None):
    from .path_command_selection import task_pointer_name
    config={} if config is None else config
    from .preference_best import digest
    root=Path(root);identities={}
    for alpha in (0,1):
        out=root/f'alpha{alpha}';record=json.loads((out/task_pointer_name(config)).read_text())
        for key in ['actor','checkpoint']:
            if digest(record[key])!=record[key+'_sha256']:raise ValueError('best_task file SHA mismatch')
        receipt=json.loads((out/'final_best'/'source.json').read_text())
        if any(receipt[key+'_sha256']!=record[key+'_sha256'] for key in ['actor','checkpoint']):raise ValueError('final physical panel source differs from best_task')
        identities[f'alpha{alpha}']={'best_task':record,'legacy_best':json.loads((out/'best_model.json').read_text()) if (out/'best_model.json').exists() else None,'last':json.loads((out/'last_completed.json').read_text())}
    return identities


def report_coverage_parent_comparison(root,cfg,fork_receipt):
    """Render cached parent/new best evidence; never runs or selects a policy."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .path_command_selection import cases
    from .preference_best import digest
    root=Path(root);parent=Path(fork_receipt['parent_run'])
    identities=checked_final_sources(root)
    out=root/'report'/'coverage_parent_comparison';out.mkdir(parents=True,exist_ok=True)
    result=dict(schema='path_coverage_parent_comparison_v1',parent_run=str(parent),
        current_run=str(root.resolve()),sources={},training={},scenarios={},
        evidence_scope='Original six fixed DEV cases, not independent generalization; cached physics only. No reward ranking.',
        window_semantics='old bend = chi>0 (demand_positive), not all geometric turns; ordinary = chi<=0 may curve; strong_demand = chi>=.5; geometric_turn = abs(path_curvature)>1e-4; overlapping active-tick windows')
    lines=['# Coverage adaptation: parent best versus new best_task','',result['evidence_scope'],
           result['window_semantics'],'','The first panel axis is fixed-world XY. Times are not shifted; traces stop at their actual terminal frame.',
           'A selected new update 0 is the inherited parent candidate, not a newly trained improvement.','']
    for alpha in (0,1):
        key=f'alpha{alpha}';old=fork_receipt['identities'][key];new=identities[key]['best_task']
        source=json.loads((parent/key/'final_best'/'source.json').read_text())
        for field in ('actor','checkpoint'):
            if source[field+'_sha256']!=old[field+'_sha256'] or digest(old[field])!=old[field+'_sha256']:
                raise ValueError('parent comparison source SHA mismatch: '+key+' '+field)
        if source['source_update']!=old['source_update']:raise ValueError('parent comparison update mismatch')
        result['sources'][key]=dict(parent=old,current=new,new_update=new['source_update'],
            lineage_update=old['source_update']+new['source_update'],inherited_parent_selected=new['source_update']==0)
        lines.extend([f"- {key}: parent update {old['source_update']} → selected new update {new['source_update']} "
                      f"(lineage {old['source_update']+new['source_update']}); inherited parent selected: {new['source_update']==0}.",
                      f"  Parent Actor `{old['actor_sha256']}`; current Actor `{new['actor_sha256']}`."])
        def receipts(folder):
            return [dict(path=str(p.resolve()),receipt=json.loads(p.read_text()))
                    for p in sorted((root/key/folder).glob('update_*.json'))]
        coverage=receipts('coverage');statistics=receipts('statistics');decisions=receipts('decisions')
        complete=[x for x in coverage if x['receipt'].get('window_updates')==10]
        counts=[(x['receipt']['windows']['strong_demand']['samples'],x['receipt']['active_lower_samples']) for x in complete]
        exposure=sum(x[0] for x in counts)/sum(x[1] for x in counts) if counts and sum(x[1] for x in counts) else None
        rows=[]
        for entry in statistics:
            s=entry['receipt'];w=s['windows']['strong_demand']
            rows.append(dict(update=s.get('update'),window_updates=s.get('window_updates'),
                strong_fraction=w['sample_fraction'],speed_rmse_m_s=w['speed_rmse_m_s'],
                path_rmse_m=w['lateral_rmse_m'],heading_rmse_rad=w['heading_rmse_rad'],
                peak_roll_rad=w.get('peak_roll_rad'),working_exceedance_fraction=w.get('working_roll_exceedance_fraction'),
                effective_cost_means={k:v['effective_mean'] for k,v in w.get('costs',{}).items()}))
        full=[r for r in rows if r['window_updates']==25];trend={'status':'UNKNOWN: fewer than two complete 25-update windows'}
        if len(full)>=2:
            before,after=full[-2:];primary='path_rmse_m' if alpha==0 else 'speed_rmse_m_s'
            metric_keys=('speed_rmse_m_s','path_rmse_m','heading_rmse_rad','working_exceedance_fraction')
            delta={k:after[k]-before[k] if before[k] is not None and after[k] is not None else None for k in metric_keys}
            trend=dict(status='descriptive random-rollout comparison; no causal or generalization claim',
                previous_update=before['update'],latest_update=after['update'],primary_metric=primary,
                latest_minus_previous=delta,primary_improves=None if delta[primary] is None else delta[primary]<0,
                effective_cost_latest_minus_previous={k:after['effective_cost_means'][k]-v
                    for k,v in before['effective_cost_means'].items() if k in after['effective_cost_means']})
        passed=sum(x['receipt']['windows']['strong_demand']['sample_fraction'] is not None and
                   x['receipt']['windows']['strong_demand']['sample_fraction']>=.05 for x in complete)
        result['training'][key]=dict(coverage_receipts=coverage,statistics_receipts=statistics,
            decision_receipts=decisions,complete_ten_update_windows=len(complete),windows_at_least_five_percent=passed,
            pooled_strong_active_fraction=exposure,twenty_five_window_rows=rows,latest_twenty_five_trend=trend,
            coverage_diagnosis=json.loads((root/key/'coverage_diagnosis.json').read_text()) if (root/key/'coverage_diagnosis.json').exists() else None,
            final_decision=decisions[-1]['receipt'] if decisions else {'status':'UNKNOWN: no decision receipt'},
            last_completed=identities[key]['last'])
        lines.extend([f"- {key} effective strong exposure: {'UNKNOWN' if exposure is None else f'{exposure:.2%}'}; "
                      f"complete 10-update windows ≥5%: {passed}/{len(complete)}. Reset fractions remain separate in JSON.",
                      f"- {key} last recorded decision: `{json.dumps(result['training'][key]['final_decision'],ensure_ascii=False)}`",
                      f"- {key} last two complete 25-update windows: `{json.dumps(trend,ensure_ascii=False)}`"])
    summaries={key:dict(goal_and_hold=0,working_roll_failures=0,physical_or_domain_failures=0)
               for key in ('zero_upper','parent_alpha0','parent_alpha1','alpha0','alpha1')}
    for name,_,speed in cases(cfg):
        locations={'zero_upper':root/'zero_upper',**{f'parent_alpha{a}':parent/f'alpha{a}'/'final_best' for a in (0,1)},
                   **{f'alpha{a}':root/f'alpha{a}'/'final_best' for a in (0,1)}}
        data={k:dict(np.load(p/f'{name}.npz',allow_pickle=False)) for k,p in locations.items()}
        path=dict(np.load(root/'zero_upper'/f'{name}_path.npz',allow_pickle=False))
        for folder in locations.values():
            other=dict(np.load(folder/f'{name}_path.npz',allow_pickle=False))
            if set(other)!=set(path) or any(not np.array_equal(path[k],other[k]) for k in path):
                raise ValueError('parent/new fixed path differs: '+name)
        metrics=paired_path_metrics(data,path,cfg)
        old_pair=paired_path_metrics({'alpha0':data['parent_alpha0'],'alpha1':data['parent_alpha1']},path,cfg)
        metrics['parent_pairing']=old_pair['pairing'];metrics['current_pairing']=metrics.pop('pairing')
        metrics['current_minus_parent']={}
        for alpha in (0,1):
            old=metrics['methods'][f'parent_alpha{alpha}'];new=metrics['methods'][f'alpha{alpha}']
            metrics['current_minus_parent'][f'alpha{alpha}']=dict(
                peak_roll_rad=new['peak_roll_rad']-old['peak_roll_rad'],
                working_exceedance_s=new['working_exceedance_s']-old['working_exceedance_s'],
                windows={w:{k:(new['windows'][w].get(k)-old['windows'][w].get(k))
                    if new['windows'][w].get(k) is not None and old['windows'][w].get(k) is not None else None
                    for k in ('speed_rmse_m_s','path_rmse_m','heading_rmse_rad')}
                    for w in ('whole','spatial_turn','strong_demand','exit')})
        result['scenarios'][name]=metrics
        fig,axes=plt.subplots(3,2,figsize=(15,13));xy=axes[0,0]
        extent=max(float(np.max(d['path_progress'])) for d in data.values())+2
        xy.plot(*path['xy'][path['s']<=extent].T,'k--',label='original fixed path')
        for key,d in data.items():
            m=metrics['methods'][key];summaries[key]['goal_and_hold']+=int(m['goal_crossed'] and m['final_hold'])
            summaries[key]['working_roll_failures']+=int(m['working_exceedance_s']>0)
            summaries[key]['physical_or_domain_failures']+=int(m['physical_failure'] or m['tracking_domain_failure'])
            isparent=key.startswith('parent');style='--' if isparent else '-'
            color='black' if key=='zero_upper' else ('tab:blue' if key.endswith('0') else 'tab:orange')
            label=key
            if key!='zero_upper':
                a=key[-1];identity=result['sources'][f'alpha{a}'];update=identity['parent' if isparent else 'current']['source_update']
                label=f"{'parent' if isparent else 'new'} a{a} update{update}"
            t=np.asarray(d['time'])+cfg['timing']['lower_dt_s']
            xy.plot(*d['actual_xy'].T,style,color=color,label=label)
            xy.plot(*d['actual_xy'][-1],marker='x' if m['physical_failure'] or m['tracking_domain_failure'] else 'o',color=color)
            for ax,y in ((axes[0,1],d['actual_forward_speed']),(axes[1,0],d['path_cross_track']),
                         (axes[1,1],d['peak_roll']),(axes[2,0],d['path_heading_error'])):
                ax.plot(t,y,style,color=color,label=label)
                ax.plot(t[-1],y[-1],marker='x' if m['physical_failure'] or m['tracking_domain_failure'] else 'o',color=color)
        xy.set_aspect('equal',adjustable='datalim');xy.set_xlabel('World X (m)');xy.set_ylabel('World Y (m)')
        axes[0,1].axhline(speed,color='gray',ls=':',label='user speed');axes[0,1].set_ylabel('Actual forward speed (m/s)')
        axes[1,0].set_ylabel('Signed cross-track error (m)');axes[1,1].set_ylabel('Peak absolute roll per tick (rad)')
        axes[1,1].axhline(cfg['stage_B_baseline']['working_roll_rad'],color='red',ls=':',label='working bound')
        axes[2,0].set_ylabel('Signed path heading error (rad)')
        for ax in (axes[1,0],axes[2,0]):
            bound=cfg['stage_B_baseline']['final_y_error_m' if ax is axes[1,0] else 'final_path_heading_error_rad']
            for sign in (-1,1):ax.axhline(sign*bound,color='gray',ls=':',lw=.7)
        axes[2,1].axis('off')
        table=[[k,str(m['goal_crossed']),str(m['final_hold']),f"{m['working_exceedance_s']:.3f}",f"{m['observed_seconds']:.2f}"] for k,m in metrics['methods'].items()]
        axes[2,1].table(cellText=table,colLabels=['Method','Goal','Final hold','Roll > bound (s)','Observed (s)'],loc='center').auto_set_font_size(False)
        for ax in axes.flat:
            if ax is axes[2,1]:continue
            ax.legend(fontsize=7);ax.grid(alpha=.2)
            if ax is not xy:
                ax.set_xlabel('Actual time (s)');ax.axvspan(cfg['training_future_phase_C']['episode_s']-cfg['stage_B_baseline']['hold_s'],cfg['training_future_phase_C']['episode_s'],color='gray',alpha=.08)
        seeds='/'.join(str(result['sources'][f'alpha{a}']['current'].get('seed','UNKNOWN')) for a in (0,1))
        fig.suptitle(name+f' | cached fixed DEV | local300 | run seeds a0/a1={seeds} | no external disturbance\nParent versus disk-selected new best_task; x=failure, o=actual final frame; shaded=final hold window')
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/f'{name}.png',dpi=130);plt.close(fig)
        lines.extend(['',f'## {name}',f'![Fixed XY first, speed/path/roll and final hold]({name}.png)','',
                      '| Method | Whole speed RMSE m/s | Whole path RMSE m | Turn speed RMSE m/s | Turn path RMSE m | Peak roll rad | Roll exceed s | Goal / hold | Failure physical / domain |',
                      '|---|---:|---:|---:|---:|---:|---:|---|---|'])
        fmt=lambda x:'N/A' if x is None else f'{x:.5f}'
        for key,m in metrics['methods'].items():
            w=m['windows']['whole'];turn=m['windows']['spatial_turn']
            lines.append(f"| {key} | {fmt(w['speed_rmse_m_s'])} | {fmt(w['path_rmse_m'])} | {fmt(turn.get('speed_rmse_m_s'))} | {fmt(turn.get('path_rmse_m'))} | {fmt(m['peak_roll_rad'])} | {fmt(m['working_exceedance_s'])} | {m['goal_crossed']} / {m['final_hold']} | {m['physical_failure']} / {m['tracking_domain_failure']} |")
    result['six_case_summary']=summaries
    next_step=dict(schema='path_coverage_decision_summary_v1',initial_new_updates_per_endpoint=100,
        hard_max_new_updates_per_endpoint=200,automatic_further_training=False,
        report='coverage_parent_comparison/INDEX.md',
        qualification_semantics='qualified_task is same-six-DEV task qualification, separate from proof of alpha preference; no automatic extension beyond saved decisions',
        endpoints={key:dict(last_completed=result['training'][key]['last_completed'],
            selected_new_update=result['sources'][key]['new_update'],
            selected_lineage_update=result['sources'][key]['lineage_update'],
            qualified_task=identities[key]['best_task'].get('qualified_task'),
            recorded_decision=result['training'][key]['final_decision']) for key in ('alpha0','alpha1')})
    for key,endpoint in next_step['endpoints'].items():
        completed=endpoint['last_completed']['update'];decision=endpoint['recorded_decision']
        endpoint['action']='freeze_at_hard_cap' if completed>=200 else (
            'freeze_after_failed_extension_gate' if decision.get('allow') is False else
            'report_saved_boundary_no_automatic_additional_training')
        lines.insert(7,f"{key}: completed {completed} new updates; {endpoint['action']}; selected qualified_task={endpoint['qualified_task']} (separate from preference proof).")
    (root/'report'/'next_step.json').write_text(json.dumps(next_step,indent=2,allow_nan=False)+'\n')
    result['next_step']=next_step
    lines[7:7]=['','Physical outcomes (denominator six DEV cases):','',*[f"- {k}: goal + final hold {v['goal_and_hold']}/6; working-roll failures {v['working_roll_failures']}/6; physical/domain failures {v['physical_or_domain_failures']}/6." for k,v in summaries.items()],'']
    lines.extend(['','[Exact identities, physical metrics, common spatial-bin comparisons, exposure receipts and decisions](comparison.json)',
        'Same path is compared at each method’s actual arrival times. Occupied-time RMSE and common spatial-bin differences are descriptive. Neither survival, return, nor best selection proves preference or safety.'])
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'INDEX.md').write_text('\n'.join(lines)+'\n')
    index=root/'report'/'INDEX.md';link='[Parent best100/150 versus coverage best_task](coverage_parent_comparison/INDEX.md)'
    if index.exists() and link not in index.read_text():
        with index.open('a') as stream:stream.write('\n'+link+'\n')
    return result

def report_final_panel(root,config):
    from .path_command_selection import qualify_preference,SAFETY_SELECTION_SCHEMA
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .path_command_selection import cases
    root=Path(root);identities=checked_final_sources(root,config);out=root/'report';out.mkdir(exist_ok=True);lines=['# Geometric upper fixed-path comparison','', 'Same immutable path and prepared state; nominal PP reference may differ by actual pose. No post-hoc smoothing. Best is fixed-DEV ranking, not independent validation or proof of alpha preference.','']
    all_metrics={}
    for name,route,speed in cases(config):
        data={'zero_upper':dict(np.load(root/'zero_upper'/f'{name}.npz'))}
        for alpha in (0,1):data[f'alpha{alpha}']=dict(np.load(root/f'alpha{alpha}'/'final_best'/f'{name}.npz'))
        path=dict(np.load(root/'zero_upper'/f'{name}_path.npz'))
        for alpha in (0,1):
            other=dict(np.load(root/f'alpha{alpha}'/'final_best'/f'{name}_path.npz'))
            if set(other)!=set(path) or any(not np.array_equal(path[k],other[k]) for k in path):raise ValueError('paired fixed path differs')
        all_metrics[name]=paired_path_metrics(data,path,config)
        fig,axes=plt.subplots(5,2,figsize=(15,17));ax=axes[0,0]
        maxs=max(float(d['path_progress'].max()) for d in data.values());mask=path['s']<=maxs+2
        ax.plot(*path['xy'][mask].T,'k--',label='fixed original path')
        for method,d in data.items():
            t=d['time']+.005;ax.plot(*d['actual_xy'].T,label=method);ax.plot(*d['actual_xy'][-1],marker='x' if (d['physical_failure'].any() or np.any(d.get('tracking_domain_failure',d.get('domain_exit',False)))) else 'o')
            axes[0,1].plot(t,d['path_heading_error'],label=method+' heading rad')
            axes[1,0].plot(t,d['actual_forward_speed']-speed,label=method+' speed error m/s');axes[1,1].plot(t,d['path_cross_track'],label=method+' path error m')
            for ch in (0,1):
                for key,style in [('nominal','--'),('governed',':')]:axes[2,ch].plot(t,d[key][:,ch],style,label=method+' '+key,lw=.8)
                axes[2,ch].plot(t,d['actual_forward_speed'] if ch==0 else d['actual_delta'],label=method+' actual',lw=.8)
                axes[3,ch].plot(t,d['target_offset'][:,ch],'--',label=method+' target',lw=.5);axes[3,ch].plot(t,d['filtered_offset'][:,ch],':',label=method+' filtered',lw=.5);axes[3,ch].plot(t,d['offsets'][:,ch],label=method+' correction');axes[3,ch].plot(t,d['lower_error'][:,ch],':',label=method+' lower error')
        for alpha in (0,1):
            for method,style in [(f'alpha{alpha}','-'),('zero_upper','--')]:
                d=data[method];reward,components=reward_series(d,alpha,config)
                if method!='zero_upper':np.testing.assert_allclose(reward,d['scored_tick_reward'],rtol=2e-4,atol=2e-6)
                step=np.arange(1,len(reward)+1);label=method+f' scored alpha{alpha}'
                axes[4,0].plot(step,reward,style,label=label);axes[4,1].plot(step,np.cumsum(reward),style,label=label)
                np.savez_compressed(out/f'{name}_{method}_scored_alpha{alpha}.npz',reward=reward,**components)
        cf,ca=plt.subplots(2,1,figsize=(12,8))
        for alpha in (0,1):
            for method,style in [(f'alpha{alpha}','-'),('zero_upper','--')]:
                _,components=reward_series(data[method],alpha,config)
                for component,values in components.items():
                    ca[0].plot(values,style,label=f'{method} a{alpha} {component}',lw=.6)
                    ca[1].plot(np.cumsum(values),style,lw=.6)
        ca[0].set_ylabel('Signed per-step reward components');ca[1].set_ylabel('Cumulative components');ca[1].set_xlabel('Actual lower steps');ca[0].legend(fontsize=5,ncol=4)
        cf.tight_layout();cf.savefig(out/f'{name}_reward_components.png',dpi=130);plt.close(cf)
        ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)')
        for row in range(5):
            for col in range(2):
                a=axes[row,col];a.legend(fontsize=6);a.grid(alpha=.2)
                if (row,col)!=(0,0):a.set_xlabel('Time (s)' if row<4 else 'Actual lower steps')
        axes[2,0].set_ylabel('m/s');axes[2,1].set_ylabel('rad');axes[3,0].set_ylabel('m/s');axes[3,1].set_ylabel('rad')
        fig.suptitle(name+' | local300 | endpoint disk-loaded best_task | no external disturbance\nReward weights differ by alpha: do not rank physical quality by cross-alpha return');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/f'{name}.png',dpi=130);plt.close(fig)
        # A separate equal-aspect XY artifact is always available first.
        f,a=plt.subplots(figsize=(8,6));a.plot(*path['xy'][mask].T,'k--',label='fixed path')
        for method,d in data.items():a.plot(*d['actual_xy'].T,label=method);a.plot(*d['actual_xy'][-1],marker='x' if (d['physical_failure'].any() or np.any(d.get('tracking_domain_failure',d.get('domain_exit',False)))) else 'o')
        a.set_aspect('equal',adjustable='datalim');a.set_xlabel('World X (m)');a.set_ylabel('World Y (m)');a.legend();a.set_title(name);f.tight_layout();f.savefig(out/f'{name}_XY.png',dpi=130);plt.close(f)
        lines.extend([f'## {name}',f'![XY]({name}_XY.png)',f'[Full commands/errors/reward]({name}.png)',f'[Signed and cumulative components]({name}_reward_components.png)','', '| Method | Whole speed RMSE m/s | Whole path RMSE m | Turn speed RMSE m/s | Turn path RMSE m | Peak roll rad | Goal / hold | Physical / domain failure |','|---|---:|---:|---:|---:|---:|---|---|'])
        fmt=lambda x:'N/A' if x is None else f'{x:.5f}'
        for method,m in all_metrics[name]['methods'].items():
            w=m['windows']['whole'];turn=m['windows']['spatial_turn']
            lines.append(f"| {method} | {fmt(w['speed_rmse_m_s'])} | {fmt(w['path_rmse_m'])} | {fmt(turn.get('speed_rmse_m_s'))} | {fmt(turn.get('path_rmse_m'))} | {fmt(m['peak_roll_rad'])} | {m['goal_crossed']} / {m['final_hold']} | {m['physical_failure']} / {m['tracking_domain_failure']} |")
        lines.extend(['','| Window | Dv m/s | Dy m | Direction supported | Discernible reference |','|---|---:|---:|---|---|'])
        for window,pair in all_metrics[name]['pairing'].items():
            if window=='common_arclength_bins':continue
            lines.append(f"| {window} | {fmt(pair.get('D_v_m_s'))} | {fmt(pair.get('D_y_m'))} | {pair.get('direction_correct','N/A')} | {pair.get('discernible_reference','N/A')} |")
        lines.append('')
    if config.get('task_selection_schema')==SAFETY_SELECTION_SCHEMA:
        traces0={name:dict(np.load(root/'alpha0'/'final_best'/f'{name}.npz')) for name,_,_ in cases(config)}
        traces1={name:dict(np.load(root/'alpha1'/'final_best'/f'{name}.npz')) for name,_,_ in cases(config)}
        paths={name:dict(np.load(root/'alpha0'/'final_best'/f'{name}_path.npz')) for name,_,_ in cases(config)}
        (out/'preference_qualification.json').write_text(json.dumps(qualify_preference(traces0,traces1,paths,config),indent=2,allow_nan=False))
        lines.extend(['[Predeclared safety and preference qualification](preference_qualification.json)',
            'New-stage acceptance first requires both complete common tasks (roll <=.302rad; cross-track <=.40m; original goal/last1s hold and primary gates). Preference then uses every fixed .5m bin of each complete high-speed left/right turn, sqrt(mean(bin MSE)); Dv >=.03m/s and Dy >=.02m. Missing bins fail; no common-fragment shortcut.'])
    (out/'paired_metrics.json').write_text(json.dumps(all_metrics,indent=2,allow_nan=False))
    lines.extend(['[Same-path paired windows, spatial bins, commands and lower evidence](paired_metrics.json)','Differences: Dv=Ev(alpha0)-Ev(alpha1), Dy=Ey(alpha1)-Ey(alpha0). Positive values support the intended direction; .03m/s and .02m are descriptive discernibility references, not reward terms. Lower causal blocking remains unproved from RMSE alone.'])
    (out/'identities.json').write_text(json.dumps(identities,indent=2));lines.append('[Exact checkpoint identities and fixed-validation metrics](identities.json)');(out/'INDEX.md').write_text('\n'.join(lines)+'\n')


def _conditional_turn_bins(trace,path,config):
    """Full original-turn equal-bin MSE; missing coverage is never discarded."""
    start=float(config['path']['straight_prefix_m']);end=float(path['turn_end'])
    if end<=start:return dict(status='not_a_turn',speed_rmse=None,path_rmse=None,counts=[])
    edges=np.append(np.arange(start,end,.5),end);progress=np.asarray(trace['path_progress'])
    ev=np.asarray(trace['actual_forward_speed'])-trace['v_user'];ey=np.asarray(trace['path_cross_track'])
    counts=[];values=[]
    for index,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        selected=(progress>=lo)&((progress<=hi) if index==len(edges)-2 else (progress<hi))
        counts.append(int(selected.sum()))
        if selected.any():values.append([float(np.mean(ev[selected]**2)),float(np.mean(ey[selected]**2))])
    complete=bool(counts) and all(counts)
    errors=np.sqrt(np.mean(values,axis=0)) if complete else (None,None)
    return dict(status='complete' if complete else 'incomplete_turn',edges_m=edges.tolist(),counts=counts,
                speed_rmse=None if errors[0] is None else float(errors[0]),
                path_rmse=None if errors[1] is None else float(errors[1]))


def report_conditional_panel(root,config):
    """All-alpha evidence for one disk-selected shared model; zero simulator calls."""
    from .path_command_selection import (cases,conditional_grid,alpha_key,summarize_conditional,
                                        CONDITIONAL_SELECTION_SCHEMA)
    from .preference_best import digest,config_hash,atomic_json
    root=Path(root);learner=root/'shared';final=learner/'final_best'
    strong=config.get('preference_selection_schema') is not None
    from .path_command_selection import PREFERENCE_SELECTION_SCHEMA,summarize_preference,preference_score_key
    best=json.loads((learner/('best_preference_model.json' if strong else 'best_conditional_model.json')).read_text())
    if best.get('selection_schema')!=(PREFERENCE_SELECTION_SCHEMA if strong else CONDITIONAL_SELECTION_SCHEMA):raise ValueError('conditional best selection identity mismatch')
    if json.loads((final/'source.json').read_text())!=best:raise ValueError('conditional final source differs from selected model')
    for kind in ('checkpoint','actor'):
        if digest(best[kind])!=best[kind+'_sha256']:raise ValueError('conditional best SHA mismatch')
    values=conditional_grid(config);alphas={alpha_key(a):a for a in values}
    meta=json.loads((final/'panel_identity.json').read_text())
    if not meta.get('completed') or meta.get('alphas')!=alphas or meta.get('config_sha256')!=config_hash(config):
        raise ValueError('conditional final panel identity mismatch')
    declared=cases(config)
    panels={key:{name:dict(np.load(final/key/f'{name}.npz')) for name,_,_ in declared} for key in alphas}
    summary=summarize_preference(panels,config) if strong else summarize_conditional(panels,config)
    if strong:summary['score_tuple']=preference_score_key(summary,best['source_update'])
    if tuple(summary['score_tuple'])!=tuple(best['score_tuple']):
        raise ValueError('conditional final metrics differ from selected model')
    baseline={name:dict(np.load(root/'zero_upper'/f'{name}.npz')) for name,_,_ in declared}
    paths={name:dict(np.load(root/'zero_upper'/f'{name}_path.npz')) for name,_,_ in declared}
    # Validate every physical path and reconstructed score before drawing anything.
    scores={};maximum_error=0.;report_cases={};preference_cases={}
    for name,route,speed in declared:
        path=paths[name];scores[name]={};report_cases[name]={'alphas':{},'baseline_trace':str((root/'zero_upper'/f'{name}.npz').resolve())}
        for key,alpha in alphas.items():
            other=dict(np.load(final/key/f'{name}_path.npz'))
            if set(path)!=set(other) or any(not np.array_equal(path[k],other[k]) for k in path):
                raise ValueError('conditional original path differs across alpha or baseline')
            trace=panels[key][name];reward,parts=reward_series(trace,alpha,config)
            logged=np.asarray(trace['scored_tick_reward'])
            if logged.shape!=reward.shape or not np.allclose(reward,logged,rtol=2e-4,atol=2e-6):
                raise ValueError(f'conditional reward reconstruction mismatch: {key}/{name}')
            maximum_error=max(maximum_error,float(np.max(np.abs(reward-logged))))
            baseline_reward,baseline_parts=reward_series(baseline[name],alpha,config)
            scores[name][key]=(reward,parts,baseline_reward,baseline_parts)
            m=dict(summary['per_alpha'][key]['cases'][name])
            m.update(trace=str((final/key/f'{name}.npz').resolve()),
                     full_turn_equal_bins=_conditional_turn_bins(trace,path,config) if route!='straight' else None,
                     total_reward=float(reward.sum()),baseline_same_alpha_reward=float(baseline_reward.sum()),
                     component_returns={k:float(v.sum()) for k,v in parts.items()})
            report_cases[name]['alphas'][key]=m
        if route!='straight' and speed==config['stage_C_evaluation']['conflict_speed_m_s']:
            bins={key:report_cases[name]['alphas'][key]['full_turn_equal_bins'] for key in alphas}
            complete=all(v['status']=='complete' for v in bins.values())
            dv=bins['alpha_0']['speed_rmse']-bins['alpha_1']['speed_rmse'] if complete else None
            dy=bins['alpha_1']['path_rmse']-bins['alpha_0']['path_rmse'] if complete else None
            preference_cases[name]=dict(all_alpha_complete_turn=complete,bins=bins,D_v_m_s=dv,D_y_m=dy,
                endpoint_direction_observed=bool(complete and dv>0 and dy>0),
                legacy_endpoint_difference_gate=bool(complete and dv>=.03 and dy>=.02),
                adjacent_speed_error_changes=None if not complete else np.diff([bins[k]['speed_rmse'] for k in alphas]).tolist(),
                adjacent_path_error_changes=None if not complete else np.diff([bins[k]['path_rmse'] for k in alphas]).tolist())
    preference=dict(cases=preference_cases,qualified=bool(summary['qualified_task'] and len(preference_cases)==2
        and all(v['legacy_endpoint_difference_gate'] for v in preference_cases.values())),
        role='legacy common-task and endpoint preference diagnostics; interior curves are descriptive, not a monotonicity certificate')
    if strong:
        preference.update(qualified=summary['qualified_preference'],acceptance=summary['preference_cases'],
            role='strong fixed-alpha preference; physical/domain and goal/hold required; roll/corridor diagnostics separate')
    result=dict(schema='path_conditional_report_v1',model=best,alphas=alphas,summary=summary,
                cases=report_cases,preference=preference,
                reward_audit=dict(checked_policy_traces=len(declared)*len(alphas),max_abs_tick_error=maximum_error),
                simulation_ticks=0,scope='fixed historical DEV; one shared Actor at every alpha; no generalization claim')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=root/'report';out.mkdir(exist_ok=True)
    colors={key:plt.get_cmap('viridis')(i/(len(alphas)-1)) for i,key in enumerate(alphas)}
    dt=config['timing']['lower_dt_s'];h=config['training_future_phase_C']['episode_s']
    title=f"shared checkpoint u{best['source_update']} SHA={best['checkpoint_sha256'][:12]} | seed={best.get('seed','UNKNOWN')}"
    lines=['# Shared conditional upper: all-alpha fixed-path report','',
           '![All six original/actual XY panels](ALL_XY.png)','',title,'',
           'One Actor, Critic and checkpoint for every alpha. B0 = frozen local300 + ECBC/ESO. '
           'Original paths and requests are unchanged; no external disturbance. x marks real failure and o the real final sample. '
           'Reward uses each alpha objective, with the matching rescored B0; cross-alpha returns are not physical rankings. '
           'The historical .302 rad/.40 m and task gates below are diagnostics, not reasons to shorten the authorized training budget.','']
    if strong:
        lines += [f"Strong preference qualified: {summary['qualified_preference']}; legacy common task diagnostic: {summary['qualified_task']}.",
                  f"Architecture: {best['architecture']}; reward: {best['reward_version']}; selection: {best['selection_schema']}.",
                  'High-speed endpoint targets and complete-turn differences/adjacent-alpha continuity determine preference acceptance. Roll/corridor are reported diagnostics.','']
    allfig,allaxes=plt.subplots(2,3,figsize=(16,10))
    def finish(fig,filename):
        fig.tight_layout(rect=(0,0,1,.955));fig.savefig(out/filename,dpi=110);plt.close(fig)
    def decorate(axes):
        for ax in np.asarray(axes).flat:
            ax.grid(alpha=.2);ax.legend(fontsize=5,ncol=2)
    for (name,route,speed),allxy in zip(declared,allaxes.flat):
        data={'B0':baseline[name],**{k:panels[k][name] for k in alphas}};path=paths[name]
        max_progress=max(float(d['path_progress'].max()) for d in data.values())
        path_mask=path['s']<=max_progress+2
        physical,pa=plt.subplots(2,3,figsize=(16,9))
        xyfig,xy=plt.subplots(figsize=(9,7))
        for ax in (allxy,xy,pa[0,0]):
            ax.plot(*path['xy'][path_mask].T,'k--',label='original fixed path')
            ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=name)
            ax.set_aspect('equal',adjustable='datalim')
        chain,ch=plt.subplots(4,2,figsize=(16,15))
        for method,d in data.items():
            color='0.55' if method=='B0' else colors[method];label='B0 local300' if method=='B0' else f'alpha={alphas[method]:g}'
            t=np.asarray(d['time'])+dt
            failed=bool(np.any(d['physical_failure']) or np.any(d.get('tracking_domain_failure',d.get('domain_exit',False))))
            marker='x' if failed else 'o'
            for ax in (allxy,xy,pa[0,0]):
                ax.plot(*d['actual_xy'].T,color=color,label=label,lw=1)
                ax.plot(*d['actual_xy'][-1],marker=marker,color=color,ms=4)
            curves=((pa[0,1],d['peak_roll']),(pa[0,2],d['actual_forward_speed']-d['v_user']),
                    (pa[1,0],d['path_cross_track']),(pa[1,1],d['path_heading_error']),
                    (pa[1,2],d['actual_forward_speed']))
            for ax,y in curves:
                ax.plot(t,y,color=color,label=label,lw=.9);ax.plot(t[-1],y[-1],marker=marker,color=color,ms=3)
            for channel in (0,1):
                for key,style in (('nominal','--'),('governed',':')):
                    ch[0,channel].plot(t,d[key][:,channel],style,color=color,lw=.7,label=f'{label} {key}')
                actual=d['actual_forward_speed'] if channel==0 else d['actual_delta']
                ch[0,channel].plot(t,actual,color=color,lw=.8,label=f'{label} actual')
                for key,style in (('target_offset','--'),('filtered_offset',':'),('offsets','-')):
                    ch[1,channel].plot(t,d[key][:,channel],style,color=color,lw=.7,label=f'{label} {key}')
                ch[2,channel].plot(t,d['final_command'][:,channel],color=color,label=label,lw=.8)
            ch[3,0].plot(t,d['actual_forward_speed'],color=color,label=f'{label} body',lw=.8)
            ch[3,0].plot(t,d['wheel_speed_proxy'],'--',color=color,label=f'{label} wheel proxy',lw=.7)
            ch[3,1].plot(t,d['wheel_speed_proxy']-d['actual_forward_speed'],color=color,label=label,lw=.8)
        pa[0,1].axhline(config['stage_B_baseline']['working_roll_rad'],color='k',ls='--',lw=.7)
        for sign in (-1,1):pa[1,0].axhline(sign*.4,color='k',ls='--',lw=.7)
        for ax,label in ((pa[0,1],'Substep roll peak (rad)'),(pa[0,2],'Original speed error (m/s)'),
                         (pa[1,0],'Original signed path error (m)'),(pa[1,1],'Original heading error (rad)'),
                         (pa[1,2],'Measured body speed (m/s)')):
            ax.set(xlabel='Time (s)',ylabel=label);ax.axvspan(h-config['stage_B_baseline']['hold_s'],h,color='0.5',alpha=.07)
        pa[1,2].axhline(speed,color='k',ls=':',label='original speed request')
        ch[0,0].axhline(speed,color='k',ls=':',label='original speed request')
        ch[3,0].axhline(speed,color='k',ls=':',label='original speed request')
        chain_labels=(('Request / corrected / measured speed (m/s)','Nominal / corrected / measured steer (rad)'),
            ('Target / filtered / applied speed correction (m/s)','Target / filtered / applied steer correction (rad)'),
            ('Final front steer-rate command (rad/s)','Final rear axle-rate command (rad/s)'),
            ('Requested / body / wheel proxy speed (m/s)','Wheel proxy minus body speed (m/s)'))
        for row in range(4):
            for col in range(2):ch[row,col].set(xlabel='Time (s)',ylabel=chain_labels[row][col])
        for fig in (physical,xyfig,chain):fig.suptitle(f'{name} | {title}\nOriginal {h:g}s window, real failure endpoints, no external disturbance')
        decorate(pa);decorate(ch);decorate([xy]);allxy.legend(fontsize=5)
        finish(physical,f'{name}_physical.png');finish(xyfig,f'{name}_XY.png');finish(chain,f'{name}_command_chain.png')
        rewardfig,ra=plt.subplots(2,1,figsize=(13,8))
        component_names=list(scores[name]['alpha_0'][1]);compfig,ca=plt.subplots(len(component_names),2,figsize=(16,2.1*len(component_names)))
        for key,alpha in alphas.items():
            reward,parts,b_reward,b_parts=scores[name][key];color=colors[key]
            for y,pieces,style,label in ((reward,parts,'-',f'alpha={alpha:g} shared'),(b_reward,b_parts,'--',f'alpha={alpha:g} B0 scored')):
                steps=np.arange(1,len(y)+1)
                ra[0].plot(steps,y,style,color=color,label=label,lw=.8)
                ra[1].plot(steps,np.cumsum(y),style,color=color,label=label,lw=.8)
                for index,component in enumerate(component_names):
                    v=pieces[component];ca[index,0].plot(steps,v,style,color=color,label=label,lw=.7)
                    ca[index,1].plot(steps,np.cumsum(v),style,color=color,label=label,lw=.7)
        ra[0].set(ylabel='Signed reward per actual lower tick');ra[1].set(ylabel='Cumulative signed reward',xlabel='Actual lower steps')
        for index,component in enumerate(component_names):
            ca[index,0].set(ylabel=component+' per tick',xlabel='Actual lower steps')
            ca[index,1].set(ylabel=component+' cumulative',xlabel='Actual lower steps')
        for fig in (rewardfig,compfig):fig.suptitle(f'{name} | {title}\nSame-alpha B0 rescored independently; cumulative rewards include the true terminal tail')
        decorate(ra);decorate(ca);finish(rewardfig,f'{name}_reward.png');finish(compfig,f'{name}_components.png')
        lines.extend([f'## {name}','',f'![Original and actual XY]({name}_XY.png)','',
            f'[Roll and original errors]({name}_physical.png) · [Step and cumulative reward]({name}_reward.png) · '
            f'[Signed components]({name}_components.png) · [Complete command chain and wheel-speed proxy]({name}_command_chain.png)','',
            '| Alpha | Speed RMSE (m/s) | Path RMSE (m) | Peak roll (rad) | Corridor fail | Goal / hold | Endpoint primary fail |',
            '|---|---:|---:|---:|---|---|---|'])
        for key,alpha in alphas.items():
            m=report_cases[name]['alphas'][key]
            lines.append(f"| {alpha:g} | {m['speed_rmse']:.5f} | {m['path_rmse']:.5f} | {m['peak_roll']:.5f} | {m['corridor_failure']} | {m['goal']} / {m['final_hold']} | {m['primary_band_failure'] if m['primary_applicable'] else 'N/A (interior alpha)'} |")
        lines.append('')
    allfig.suptitle(title+' | five alpha values, same checkpoint + B0 | original world paths, metres')
    finish(allfig,'ALL_XY.png')
    atomic_json(out/'conditional_metrics.json',result);atomic_json(out/'identities.json',best)
    lines.extend(['[Full per-alpha metrics, complete-turn bins, source NPZ paths and reward audit](conditional_metrics.json)',
                  '[One shared checkpoint identity](identities.json)',
                  'The complete original turn is used for equal-bin MSE metrics. Missing bins remain missing; no common-fragment shortcut. '
                  'Middle-alpha primary thresholds are not invented. Intermediate-alpha trends and endpoint preference are reported separately from common physical diagnostics.'])
    dynamic=root/'dynamic_preference_checks'/'receipt.json'
    if strong:
        if dynamic.exists():
            dynamic_receipt=json.loads(dynamic.read_text())
            lines.extend(['', '[Dynamic alpha source-backed report](../dynamic_preference_checks/INDEX.md)',
                'Dynamic state: '+dynamic_receipt['state']+'; '+str(dynamic_receipt.get('reason') or 'see identity and goal/hold receipt')])
        else:
            lines.extend(['', 'Dynamic state: not_run; no dynamic receipt exists.'])
    (out/'INDEX.md').write_text('\n'.join(lines)+'\n')
    return result
