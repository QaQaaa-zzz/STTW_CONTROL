"""Fixed-path final comparison: actual geometry first, all commands unfiltered."""
from pathlib import Path
import json
import numpy as np

def reward_series(d,alpha,config,horizon=None):
    """Same-alpha reweighting of fixed physical traces; terminal tail retained."""
    c=config['path_reward'];dt=config['timing']['lower_dt_s'];chi=np.asarray(d['chi'],float)
    ev=np.asarray(d['actual_forward_speed'],float)-d['v_user'];ey=np.asarray(d['path_cross_track'],float);eh=np.asarray(d['path_heading_error'],float)
    huber=lambda x:np.where(abs(x)<=1,x*x,2*abs(x)-1)
    effective={k:np.asarray(d['effective_cost_'+k],float).copy() for k in c['component_caps']}
    raw=dict(speed=(10-9*chi*(1-alpha))*huber(ev/c['speed_scale_m_s']),
             path=(10-9*chi*alpha)*(huber(ey/c['cross_track_scale_m'])+c['path_heading_factor']*huber(eh/c['path_heading_scale_rad'])),
             primary=c['primary_weight']*chi*((1-alpha)*huber(np.maximum(abs(ey)-c['primary_path_band_m'],0)/c['primary_path_scale_m'])+alpha*huber(np.maximum(abs(ev)-c['primary_speed_band_m_s'],0)/c['primary_speed_scale_m_s'])))
    effective.update({k:np.minimum(v,c['component_caps'][k]) for k,v in raw.items()})
    components={k:-c['scale']*dt*v for k,v in effective.items()};components['failure_tail']=np.zeros(len(ev))
    if np.any(d['physical_failure']) or np.any(d.get('tracking_domain_failure',d.get('domain_exit',False))):
        last=len(ev)-1;interval_start=(last//4)*4
        for v in components.values():v[interval_start:]=0
        h=config['training_future_phase_C']['episode_s'] if horizon is None else horizon
        remaining=round(h/config['timing']['upper_dt_s'])-last//4;gamma=config['training_future_phase_C']['gamma']
        components['failure_tail'][-1]=-c['failure_extra']-c['scale']*config['timing']['upper_dt_s']*c['failure_cost_rate_bound']*(1-gamma**remaining)/(1-gamma)
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


def checked_final_sources(root):
    from .preference_best import digest
    root=Path(root);identities={}
    for alpha in (0,1):
        out=root/f'alpha{alpha}';record=json.loads((out/'best_task_model.json').read_text())
        for key in ['actor','checkpoint']:
            if digest(record[key])!=record[key+'_sha256']:raise ValueError('best_task file SHA mismatch')
        receipt=json.loads((out/'final_best'/'source.json').read_text())
        if any(receipt[key+'_sha256']!=record[key+'_sha256'] for key in ['actor','checkpoint']):raise ValueError('final physical panel source differs from best_task')
        identities[f'alpha{alpha}']={'best_task':record,'legacy_best':json.loads((out/'best_model.json').read_text()) if (out/'best_model.json').exists() else None,'last':json.loads((out/'last_completed.json').read_text())}
    return identities

def report_final_panel(root,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .path_command_selection import cases
    root=Path(root);identities=checked_final_sources(root);out=root/'report';out.mkdir(exist_ok=True);lines=['# Geometric upper fixed-path comparison','', 'Same immutable path and prepared state; nominal PP reference may differ by actual pose. No post-hoc smoothing. Best is fixed-DEV ranking, not independent validation or proof of alpha preference.','']
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
    (out/'paired_metrics.json').write_text(json.dumps(all_metrics,indent=2,allow_nan=False))
    lines.extend(['[Same-path paired windows, spatial bins, commands and lower evidence](paired_metrics.json)','Differences: Dv=Ev(alpha0)-Ev(alpha1), Dy=Ey(alpha1)-Ey(alpha0). Positive values support the intended direction; .03m/s and .02m are descriptive discernibility references, not reward terms. Lower causal blocking remains unproved from RMSE alone.'])
    (out/'identities.json').write_text(json.dumps(identities,indent=2));lines.append('[Exact checkpoint identities and fixed-validation metrics](identities.json)');(out/'INDEX.md').write_text('\n'.join(lines)+'\n')
