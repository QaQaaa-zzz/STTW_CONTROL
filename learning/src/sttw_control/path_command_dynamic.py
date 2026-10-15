"""Bounded dynamic-alpha checks using the existing batched evaluator."""
from pathlib import Path
import pickle
import json
import shutil
import numpy as np
import jax.numpy as j
from .preference_best import atomic_json,config_hash,digest


def dynamic_contract(config):
    c=dict(config['dynamic_preference_checks'])
    c.setdefault('enabled',True);c.setdefault('trigger','qualified_preference')
    if not isinstance(c['enabled'],bool) or c['trigger'] not in ('qualified_preference','qualified_research_preference'):
        raise ValueError('invalid dynamic enabled/trigger contract')
    if c['trigger']=='qualified_research_preference' and config.get('preference_selection_schema')!='path_preference_research_v1':
        raise ValueError('research dynamic trigger requires research selection schema')
    conflict=config.get('preference_selection_schema')=='path_conflict_preference_v1'
    expected={'rate_per_s':.5,'first_change_s':1. if conflict else .5,'return_s':4. if conflict else 6.,'max_lower_ticks':24000}
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('dynamic alpha contract mismatch')
    if config['timing']['upper_dt_s']!=.02 or config['timing']['lower_dt_s']!=.005:
        raise ValueError('dynamic alpha requires unchanged 20ms/5ms timing')
    return c


def apply_dynamic_alpha(state,mode,config):
    """Update only alpha before Actor/reward; mode 0=up, 1=down, 2=roundtrip."""
    c=dynamic_contract(config);time=state.tick*config['timing']['lower_dt_s']
    initial=j.where(mode==1,1.,0.)
    changed=j.where(mode==1,0.,1.)
    request=j.where(time+1e-7<c['first_change_s'],initial,changed)
    request=j.where((mode==2)&(time+1e-7>=c['return_s']),0.,request)
    delta=c['rate_per_s']*config['timing']['upper_dt_s']
    applied=j.clip(state.alpha+j.clip(request-state.alpha,-delta,delta),0.,1.)
    return state.replace(alpha=applied),request


def _metrics(trace,config,mode=None):
    dt=config['timing']['lower_dt_s'];h=config['training_future_phase_C']['episode_s']
    time=np.asarray(trace['time'])+dt
    ev=np.asarray(trace['actual_forward_speed'])-trace['v_user']
    ey=np.asarray(trace['path_cross_track']);eh=np.asarray(trace['path_heading_error'])
    physical=bool(np.any(trace['physical_failure']));domain=bool(np.any(trace['domain_exit']))
    if not all(np.all(np.isfinite(a)) for a in (time,ev,ey,eh,trace['alpha_applied'],trace['alpha_request'])):
        raise ValueError('nonfinite dynamic trace')
    if not np.allclose(trace['upper_alpha'],trace['alpha_applied'],rtol=0,atol=1e-7):
        raise ValueError('dynamic Actor/reward alpha mismatch')
    n=len(time)
    if n==0:raise ValueError('empty dynamic trace')
    ticks=np.rint(np.asarray(trace['time'])/dt).astype(int)
    if not np.array_equal(ticks,np.arange(n)) or not np.allclose(trace['time'],ticks*dt,atol=2e-6,rtol=0):
        raise ValueError('dynamic trace must be continuous true prefix from reset')
    request=np.asarray(trace['alpha_request']);applied=np.asarray(trace['alpha_applied'])
    contract=dynamic_contract(config)
    first=contract['first_change_s'];back=contract['return_s']
    initial=float(applied[0])
    if mode is None:mode=1 if initial>.5 else (2 if np.any(request[ticks*dt>=back]<.5) else 0)
    expected_request=np.where(ticks*dt+1e-7<first,1. if mode==1 else 0.,0. if mode==1 else 1.)
    if mode==2:expected_request=np.where(ticks*dt+1e-7>=back,0.,expected_request)
    current=1. if mode==1 else 0.;expected_applied=np.empty(n)
    for start in range(0,n,4):
        current+=np.clip(expected_request[start]-current,-.01,.01);expected_applied[start:start+4]=current
    if not np.allclose(request,expected_request,rtol=0,atol=1e-7) or not np.allclose(applied,expected_applied,rtol=0,atol=3e-6):
        raise ValueError('dynamic alpha request / applied ramp contract mismatch')
    if n==0 or n>round(h/dt) or (not(physical or domain) and n!=round(h/dt)):
        raise ValueError('incomplete dynamic trace')
    terminal=np.asarray(trace['physical_failure'])|np.asarray(trace['domain_exit'])
    if np.any(terminal[:-1]):raise ValueError('dynamic trace beyond failure endpoint')
    goal=bool(np.any((trace['goal_section_signed_distance']>=0)&(trace['path_progress']>=trace['goal_progress'])))
    from .path_command_selection import resolve_preference_acceptance,preference_hold_metrics
    acceptance=resolve_preference_acceptance(config)
    hold=preference_hold_metrics(trace,config,acceptance)
    legacy=resolve_preference_acceptance(dict(config,preference_selection_schema='path_strong_preference_v1'))
    strict=preference_hold_metrics(trace,config,legacy)
    return dict(observed_lower_ticks=n,physical_failure=physical,tracking_domain_failure=domain,
        goal=goal,**hold,legacy_strict_final_hold=strict['final_hold'],legacy_strict_hold_metrics=strict,
        final_time_s=float(time[-1]),peak_roll_rad=float(np.max(trace['peak_roll'])),
        qualified=goal and hold['final_hold'],alpha_final=float(trace['alpha_applied'][-1]))


def run_dynamic_preference_checks(root,cfg,best,*,evaluator=None,resume_parent=None):
    """No dynamic PPO; selected qualified checkpoint, max 24,000 active ticks."""
    from .path_command_policy import architecture_identity,validate_actor_parameters
    from .path_command_evaluation import _conditional_parameter_digest
    from .path_command_reward import reward_version
    from .path_command_selection import cases
    out=Path(root)/'dynamic_preference_checks'
    if out.exists():raise FileExistsError('dynamic results immutable')
    out.mkdir(parents=True)
    receipt=dict(schema='path_dynamic_preference_checks_v1',state='not_run',active_lower_ticks=0,
        max_lower_ticks=24000,reason='fixed-alpha preference not qualified')
    controls=cfg.get('dynamic_preference_checks',{})
    trigger=controls.get('trigger','qualified_preference')
    if controls.get('enabled',True) is False:
        receipt['reason']='dynamic checks disabled'
        atomic_json(out/'receipt.json',receipt);report_dynamic_preference(root,cfg,receipt);return receipt
    if controls and (not isinstance(controls.get('enabled',True),bool) or trigger not in ('qualified_preference','qualified_research_preference')):
        raise ValueError('invalid dynamic enabled/trigger contract')
    if best.get(trigger) is not True:
        atomic_json(out/'receipt.json',receipt);report_dynamic_preference(root,cfg,receipt);return receipt
    dynamic_contract(cfg)
    from .path_command_selection import resolve_preference_acceptance
    acceptance=resolve_preference_acceptance(cfg)
    acceptance_sha=config_hash(acceptance)
    receipt.update(trigger=trigger,selection_schema=cfg.get('preference_selection_schema'),acceptance=acceptance,acceptance_sha256=acceptance_sha)
    if cfg['training_future_phase_C']['episode_s']!=20.:raise ValueError('dynamic checks require frozen 20-second horizon')
    if evaluator is None or config_hash(evaluator.config)!=config_hash(cfg):
        raise ValueError('matching existing batched evaluator required')
    meta=best['panel_identity']
    if (meta.get('config_sha256')!=config_hash(cfg) or meta.get('architecture')!=architecture_identity(cfg)
            or meta.get('reward_version')!=reward_version(cfg) or not meta.get('completed')):
        raise ValueError('selected dynamic model identity mismatch')
    if cfg.get('preference_selection_schema') in ('path_preference_research_v1','path_conflict_preference_v1'):
        if best.get('selection_schema')!=cfg['preference_selection_schema']:
            raise ValueError('selected dynamic preference selection identity mismatch')
        if (meta.get('acceptance_sha256')!=acceptance_sha or best.get('acceptance_sha256')!=acceptance_sha):
            raise ValueError('selected dynamic preference acceptance identity mismatch')
    if cfg.get('preference_selection_schema')=='path_conflict_preference_v1':
        from .path_command_scenarios import explicit_case_declarations
        sha=config_hash(explicit_case_declarations(cfg))
        if meta.get('case_declarations_sha256')!=sha or best.get('case_declarations_sha256')!=sha:raise ValueError('dynamic case/script/turn identity mismatch')
    for key in ('actor','checkpoint'):
        if digest(best[key])!=best[key+'_sha256']:raise ValueError('selected dynamic '+key+' SHA mismatch')
    with Path(best['actor']).open('rb') as stream:params=pickle.load(stream)
    validate_actor_parameters(params,cfg)
    if _conditional_parameter_digest(params)!=meta.get('actor_parameters_sha256'):
        raise ValueError('selected dynamic complete Actor identity mismatch')
    high=cfg['stage_C_evaluation']['conflict_speed_m_s']
    declared=[row for row in cases(cfg) if row[1]!='straight' and row[2]==high]
    if cfg.get('preference_selection_schema')=='path_conflict_preference_v1':
        declared=[row for row in cases(cfg) if row[0] in ('compact_LR_2.8','compact_RL_2.8')]
    if len(declared)!=2:raise ValueError('exactly two high-speed turns required')
    rows=[(name+'_'+label,route,speed,mode) for name,route,speed in declared
        for mode,label in enumerate(('0to1','1to0','0to1to0'))]
    receipt.update(actor_sha256=best['actor_sha256'],checkpoint_sha256=best['checkpoint_sha256'],panel_identity=meta,cases=[r[0] for r in rows])
    parent_ticks=0
    if resume_parent is not None:
        parent=Path(resume_parent)/'dynamic_preference_checks'
        if (parent/'receipt.json').exists():
            prior=json.loads((parent/'receipt.json').read_text())
            parent_ticks=int(prior.get('lineage_lower_ticks',prior.get('active_lower_ticks',0)))
            if parent_ticks<0 or parent_ticks>24000:raise ValueError('invalid dynamic parent lineage budget')
            receipt.update(parent_consumed_lower_ticks=parent_ticks,lineage_lower_ticks=parent_ticks,
                remaining_lineage_lower_ticks=24000-parent_ticks)
            same=all(prior.get(k)==best[k] for k in ('actor_sha256','checkpoint_sha256','panel_identity'))
            if prior.get('state')=='completed' and same:
                expected={r[0]+suffix for r in rows for suffix in ('.npz','_path.npz')}
                records=prior.get('trace_files_sha256',{})
                if set(records)!=expected or set(prior.get('cases',[]))!={r[0] for r in rows}:
                    raise ValueError('dynamic parent six-case source inventory mismatch')
                for name,sha in records.items():
                    if digest(parent/'traces'/name)!=sha:raise ValueError('dynamic parent source SHA mismatch')
                traces={r[0]:dict(np.load(parent/'traces'/f'{r[0]}.npz')) for r in rows}
                metrics={name:_metrics(trace,cfg,{r[0]:r[3] for r in rows}[name]) for name,trace in traces.items()}
                if sum(m['observed_lower_ticks'] for m in metrics.values())!=parent_ticks:
                    raise ValueError('dynamic parent source tick accounting mismatch')
                shutil.copytree(parent/'traces',out/'traces')
                receipt.update(state='completed',reason=None,reused_cases=6,metrics=metrics,
                    qualified=all(m['qualified'] for m in metrics.values()),trace_files_sha256=records,
                    actor_sha256=best['actor_sha256'],checkpoint_sha256=best['checkpoint_sha256'],panel_identity=meta,cases=prior['cases'])
                atomic_json(out/'receipt.json',receipt)
                report_dynamic_preference(root,cfg,receipt)
                return receipt
            if parent_ticks:
                receipt['reason']='consumed dynamic parent has no complete matching six-case panel or resumable closed-loop state; replay prohibited'
                atomic_json(out/'receipt.json',receipt);report_dynamic_preference(root,cfg,receipt);return receipt
    receipt.update(parent_consumed_lower_ticks=parent_ticks,lineage_lower_ticks=parent_ticks,remaining_lineage_lower_ticks=24000-parent_ticks)
    receipt.update(state='running' ,reason=None,checkpoint_sha256=best['checkpoint_sha256'],
        actor_sha256=best['actor_sha256'],panel_identity=meta,cases=[row[0] for row in rows])
    atomic_json(out/'receipt.json',receipt)
    original_account=evaluator.account;original_status=evaluator.status
    def account(ticks):
        receipt['active_lower_ticks']+=int(ticks)
        receipt['lineage_lower_ticks']=parent_ticks+receipt['active_lower_ticks']
        receipt['remaining_lineage_lower_ticks']=24000-receipt['lineage_lower_ticks']
        if receipt['lineage_lower_ticks']>24000:raise RuntimeError('dynamic lower-tick budget exceeded')
        atomic_json(out/'receipt.json',receipt)
    def status(**kw):
        kw.update(stage='dynamic_preference_checks',dynamic_control_ticks_total=receipt['active_lower_ticks'])
        original_status(**kw)
    evaluator.account=account;evaluator.status=status
    try:
        traces=evaluator.evaluate(params,0.,out/'traces',dynamic_rows=rows)
        metrics={name:_metrics(trace,cfg,{r[0]:r[3] for r in rows}[name]) for name,trace in traces.items()}
        if set(metrics)!={row[0] for row in rows}:raise ValueError('dynamic six-case panel incomplete')
        records={file.name:digest(file) for file in (out/'traces').glob('*.npz')}
        receipt.update(trace_files_sha256=records,state='completed',metrics=metrics,qualified=all(x['qualified'] for x in metrics.values()))
        atomic_json(out/'receipt.json',receipt)
        report_dynamic_preference(root,cfg,receipt)
        status(state='running',dynamic_stage_state='completed',case=None,completed_intervals=1000)
        return receipt
    except Exception as exc:
        receipt.update(state='error',error=str(exc));atomic_json(out/'receipt.json',receipt);raise
    finally:evaluator.account=original_account;evaluator.status=original_status


def report_dynamic_preference(root,cfg,receipt=None):
    """Render saved physical sources only; dynamic rewards use applied alpha."""
    from .path_command_reporting import reward_series
    root=Path(root);out=root/'dynamic_preference_checks'
    receipt=receipt or json.loads((out/'receipt.json').read_text())
    if receipt['state']!='completed':
        (out/'INDEX.md').write_text('# Dynamic alpha checks\n\nState: '+receipt['state']+'\n\n'+str(receipt.get('reason',receipt.get('error','UNKNOWN')))+'\n')
        return
    records=receipt['trace_files_sha256']
    for name,sha in records.items():
        if digest(out/'traces'/name)!=sha:raise ValueError('dynamic report source SHA mismatch')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines=['# Dynamic alpha checks','',f"Shared checkpoint SHA: {receipt['checkpoint_sha256']}; fixed original paths; no external disturbance.",'',
        f"Active ticks: {receipt['active_lower_ticks']}; lineage ticks: {receipt['lineage_lower_ticks']}; remaining: {receipt['remaining_lineage_lower_ticks']}.",'']
    bases=sorted({name.rsplit('_',1)[0] for name in receipt['cases']})
    for base in bases:
        names=[base+'_'+mode for mode in ('0to1','1to0','0to1to0')]
        traces={name:dict(np.load(out/'traces'/f'{name}.npz')) for name in names}
        paths={name:dict(np.load(out/'traces'/f'{name}_path.npz')) for name in names}
        baseline=dict(np.load(root/'zero_upper'/f'{base}.npz'))
        path=dict(np.load(root/'zero_upper'/f'{base}_path.npz'))
        for other in paths.values():
            if set(other)!=set(path) or any(not np.array_equal(other[k],path[k]) for k in path):
                raise ValueError('dynamic original path differs from B0')
        fig,ax=plt.subplots(figsize=(8,7));ax.plot(*path['xy'].T,'k--',label='original fixed path')
        for name,d in {'B0 frozen local300 + ECBC/ESO':baseline,**traces}.items():
            ax.plot(*d['actual_xy'].T,label=name)
            failed=np.any(d['physical_failure']) or np.any(d['domain_exit'])
            ax.plot(*d['actual_xy'][-1],marker='x' if failed else 'o')
        ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=base+' | '+receipt['checkpoint_sha256'][:12]);ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/f'{base}_XY.png');plt.close(fig)
        fig,axes=plt.subplots(7,2,figsize=(15,24));cf,ca=plt.subplots(11,2,figsize=(15,27))
        bt=np.asarray(baseline['time'])+cfg['timing']['lower_dt_s']
        for row,col,y,label in [(1,0,baseline['actual_forward_speed']-baseline['v_user'],'speed'),(1,1,baseline['path_cross_track'],'path'),(2,0,baseline['path_heading_error'],'heading'),(2,1,baseline['phi'],'roll')]:
            axes[row,col].plot(bt,y,'k--',label='B0 '+label)
        for channel in (0,1):
            for key in ('nominal','governed'):axes[3,channel].plot(bt,baseline[key][:,channel],'--',label='B0 '+key)
            for key in ('target_offset','filtered_offset','offsets'):axes[4,channel].plot(bt,baseline[key][:,channel],'--',label='B0 '+key)
            axes[5,channel].plot(bt,baseline['final_command'][:,channel],'k--',label='B0')
            axes[3,channel].plot(bt,baseline['actual_forward_speed'] if channel==0 else baseline['actual_delta'],'k:',label='B0 actual')
        for name,d in traces.items():
            alpha=np.asarray(d['alpha_applied']);r,parts=reward_series(d,alpha,cfg)
            if not np.allclose(r,d['scored_tick_reward'],rtol=2e-4,atol=2e-6):raise ValueError('dynamic reward reconstruction mismatch')
            n=len(r);t=np.asarray(d['time'])+cfg['timing']['lower_dt_s'];steps=np.arange(1,n+1)
            # B0 physical prefix is rescored with the identical applied preference.
            bn=min(n,len(baseline['time']));ba=np.pad(alpha,(0,max(0,len(baseline['time'])-n)),mode='edge')[:len(baseline['time'])]
            br,bp=reward_series(baseline,ba,cfg);br=br[:bn];bp={k:v[:bn] for k,v in bp.items()}
            arrays=dict(reward=r,cumulative_reward=np.cumsum(r),baseline_reward=br,baseline_cumulative_reward=np.cumsum(br))
            arrays.update({'baseline_component_'+k:v for k,v in bp.items()});arrays.update({'baseline_cumulative_component_'+k:np.cumsum(v) for k,v in bp.items()})
            arrays.update({'component_'+k:v for k,v in parts.items()});arrays.update({'cumulative_component_'+k:np.cumsum(v) for k,v in parts.items()})
            np.savez_compressed(out/f'{name}_rewards.npz',**arrays)
            curves=[(0,0,d['alpha_request'],'requested alpha'),(0,1,alpha,'applied alpha'),
                (1,0,d['actual_forward_speed']-d['v_user'],'speed error (m/s)'),(1,1,d['path_cross_track'],'path error (m)'),
                (2,0,d['path_heading_error'],'heading error (rad)'),(2,1,d['phi'],'signed roll (rad)')]
            for row,col,y,label in curves:axes[row,col].plot(t,y,label=name);axes[row,col].set_ylabel(label)
            for channel in (0,1):
                for key in ('nominal','governed'):axes[3,channel].plot(t,d[key][:,channel],label=name+' '+key)
                actual=d['actual_forward_speed'] if channel==0 else d['actual_delta']
                axes[3,channel].plot(t,actual,label=name+' actual')
                for key in ('target_offset','filtered_offset','offsets'):axes[4,channel].plot(t,d[key][:,channel],label=name+' '+key)
                axes[5,channel].plot(t,d['final_command'][:,channel],label=name)
            for col,y,by in ((0,r,br),(1,np.cumsum(r),np.cumsum(br))):
                axes[6,col].plot(steps,y,label=name);axes[6,col].plot(np.arange(1,bn+1),by,'--',label=name+' matching B0 alpha')
            for ix,(key,value) in enumerate(parts.items()):
                for col,y,by in ((0,value,bp[key]),(1,np.cumsum(value),np.cumsum(bp[key]))):
                    ca[ix,col].plot(steps,y,label=name);ca[ix,col].plot(np.arange(1,bn+1),by,'--',label=name+' B0');ca[ix,col].set_ylabel(key)
        for row,col,label in ((3,0,'Nominal / governed speed (m/s)'),(3,1,'Nominal / governed steer (rad)'),(4,0,'Speed offsets (m/s)'),(4,1,'Steer offsets (rad)'),(5,0,'Final steer-rate command (rad/s)'),(5,1,'Final axle-rate command (rad/s)'),(6,0,'Per-step reward'),(6,1,'Cumulative reward')):axes[row,col].set_ylabel(label)
        for group in (axes,ca):
            for axis in group.flat:
                axis.legend(fontsize=5);axis.grid(alpha=.2);axis.set_xlabel('Actual lower steps' if group is ca else 'Time (s), reward axes: actual lower steps')
                if group is axes and axis not in axes[6]:
                    axis.axvline(dynamic_contract(cfg)['first_change_s'],color='k',ls=':');axis.axvline(dynamic_contract(cfg)['return_s'],color='k',ls=':')
        fig.suptitle(base+' | '+receipt['checkpoint_sha256'][:12]+' | seed='+str(receipt.get('seed','UNKNOWN'))+' | no disturbance; actual prefixes stop at true failure');cf.suptitle(base+' signed per-step / cumulative reward components')
        fig.tight_layout();cf.tight_layout();fig.savefig(out/f'{base}_commands_errors_rewards.png');cf.savefig(out/f'{base}_components.png');plt.close(fig);plt.close(cf)
        lines.extend([f'![{base} XY]({base}_XY.png)',f'[Alpha, commands, errors, roll, per-step and cumulative rewards]({base}_commands_errors_rewards.png)',f'[Signed per-step and cumulative components]({base}_components.png)',''])
    lines.extend(['Source traces and original paths: `traces/*.npz`; independently reconstructed signed rewards: `*_rewards.npz`.','[Identity, source hashes, physical failures and goal / final hold](receipt.json)'])
    (out/'INDEX.md').write_text('\n'.join(lines)+'\n')
