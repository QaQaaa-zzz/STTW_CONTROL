"""Frozen geometric reward reconstruction and complete paired diagnostics.

Uses captured post-step physics and PRE-step requests/alpha. It never advances
physics, extends failed traces, or changes a reference path to improve a score.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from .path import reference_table
from .tracking_reward import TrackingConfig, initial_return, return_observation, transition, tolerances


def trace_summary(trace, config):
    c=TrackingConfig(**config['tracking']);dt=config['controller']['dt']
    t=trace['time'];rs=trace['return_state'][-1]
    failed=bool(trace['terminated'][-1]);full=bool(t[-1]+1e-6>=config['horizon_seconds'])
    final=bool(full and not failed and not rs[2] and rs[4]+1e-6>=c.hold_seconds)
    qualified=final and not bool(rs[7])
    present=bool(np.any(trace['event'][0,[2,3,5]]!=0))
    ev=trace['true_forward_speed'][1:]-trace['motion_command'][:-1,1]
    path=trace['path_features'][1:]
    bv,by=tolerances(trace['priority_alpha'][:-1],c,xp=np)
    mature=t[:-1]+1e-7>=c.start_seconds
    def fraction(mask):return float(np.mean(mask[mature])) if np.any(mature) else None
    return dict(
        task_recovery_success=bool(qualified and rs[6] and rs[5]),
        recovered_after_excursion=bool(qualified and rs[6] and rs[5]),
        maintained_without_excursion=bool(qualified and not rs[6]),
        terminal_tracking_hold=final,final_joint_band_held=final,
        return_deadline_missed=bool(rs[7]),ever_left_tracking_band=bool(rs[6]),
        recovery_eligible=bool(rs[6]),disturbance_present=present,
        path_error_rmse_m=float(np.sqrt(np.mean(path[:,0]**2))),
        heading_error_rmse_rad=float(np.sqrt(np.mean(path[:,1]**2))),
        speed_error_rmse_m_s=float(np.sqrt(np.mean(ev**2))),
        speed_tolerance_exceed_fraction=fraction(np.abs(ev)>bv),
        path_tolerance_exceed_fraction=fraction(np.abs(path[:,0])>by),
        full_declared_horizon=full,
        scope='fixed-horizon geometric tracking; common final hold and no missed return deadline; no safety/generalization claim',
        recovery_criteria={'speed_m_s':c.final_speed_tolerance,'lateral_m':c.final_lateral_tolerance,
                           'heading_rad':c.final_heading_tolerance,'roll_rad':c.roll_working_limit,
                           'roll_rate_rad_s':c.final_roll_rate_tolerance,'hold_s':c.hold_seconds,
                           'return_budget_s':c.return_seconds,'return_clock':'from observable tracking-band departure, including forcing','initial_settling_s':c.start_seconds})


def audit_trace(path):
    path=Path(path);decl=json.loads((path/'declaration.json').read_text());config=decl['config']
    c=TrackingConfig(**config['tracking']);dt=config['controller']['dt']
    with np.load(path/'trace.npz',allow_pickle=False) as saved:tr={k:saved[k] for k in saved.files}
    if len(tr['time'])<2:raise ValueError('tracking audit needs at least one transition')
    if not np.allclose(np.diff(tr['time']),dt,rtol=1e-5,atol=1e-6):raise ValueError('nonuniform trace timing')
    state=initial_return(xp=np);parts={};max_state=0.
    for i in range(1,len(tr['time'])):
        event=tr['event'][i-1];tick=round(float(tr['time'][i-1])/dt)
        state,terms=transition(state,roll=tr['measurement'][i,0],roll_rate=tr['measurement'][i,1],
            speed_error=tr['true_forward_speed'][i]-tr['motion_command'][i-1,1],
            lateral_error=tr['path_features'][i,0],heading_error=tr['path_features'][i,1],
            action=tr['effective_action'][i],alpha=tr['priority_alpha'][i-1],dt=dt,
            alive_rate=config['alive_reward_rate'],failure_penalty=config['failure_penalty'],
            failed=tr['terminated'][i],enabled=tick*dt>=c.start_seconds,config=c,xp=np)
        for name,value in terms.items():parts.setdefault(name,[0.]).append(float(value))
        if int(tr['end_code'][i])!=3:  # Invalid physics can also invalidate hidden controller state.
            max_state=max(max_state,float(np.max(np.abs(return_observation(state,c,xp=np)-tr['return_state'][i]))))
    parts={k:np.asarray(v) for k,v in parts.items()};predicted=sum(parts.values())
    error=float(np.max(np.abs(predicted[1:]-tr['reward'][1:])))
    if not np.allclose(predicted[1:],tr['reward'][1:],rtol=3e-5,atol=3e-5):
        raise ValueError(f'tracking reward reconstruction mismatch {error}: {path}')
    for name,values in parts.items():
        if not np.allclose(values,tr['reward_'+name],rtol=3e-5,atol=3e-5):
            raise ValueError('tracking component mismatch: '+name)
    if max_state>2e-3:raise ValueError(f'return-state reconstruction mismatch {max_state}')
    return {'trace':tr,'config':config,'parts':parts,'summary':trace_summary(tr,config),
            'max_reward_error':error,'max_return_state_error':max_state,
            'trace_sha256':hashlib.sha256((path/'trace.npz').read_bytes()).hexdigest(),
            'declaration_sha256':hashlib.sha256((path/'declaration.json').read_bytes()).hexdigest()}


def write_diagnostics(path, *, baseline=None, audited=None, plots=True):
    path=Path(path);data=audit_trace(path) if audited is None else audited
    out=path/'analysis/tracking';out.mkdir(parents=True,exist_ok=True)
    tr=data['trace'];config=data['config'];parts=data['parts']
    result={k:v for k,v in data.items() if k not in ('trace','config','parts')}
    result['component_returns']={k:float(v[1:].sum()) for k,v in parts.items()}
    result['episode_return']=float(tr['reward'][1:].sum())
    base=audit_trace(baseline) if baseline is not None else None
    if base:
        bt=base['trace']
        if config!=base['config'] or not np.array_equal(tr['event'][0],bt['event'][0]):
            raise ValueError('paired tasks/events differ')
        if not np.allclose(tr['qpos'][0],bt['qpos'][0]) or not np.allclose(tr['qvel'][0],bt['qvel'][0]):
            raise ValueError('paired initial states differ')
        result['baseline_return']=float(bt['reward'][1:].sum())
        result['paired_return_delta']=result['episode_return']-result['baseline_return']
        result['baseline_trace_sha256']=base['trace_sha256']
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'components.npz',time=tr['time'],**parts)
    if not plots:return result
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    index=['# Frozen geometric tracking diagnostics','',
           'Every curve ends at the true recorded endpoint. Short failed returns use different exposure.',
           'The target is the original global geometric path, not a rebased trajectory.','']
    sources=[('residual' if baseline else 'recorded',data)]+([('baseline',base)] if base else [])
    def chart(name,ylabel,curves,*,steps=False):
        fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
        for label,source,values in curves:
            t=source['trace']['time'][1:];x=np.arange(1,len(t)+1) if steps else t
            ax.plot(x,values,label=label)
            if source['trace']['terminated'][-1]:ax.plot(x[-1],values[-1],marker='x')
        event=tr['event'][0];dt=config['controller']['dt']
        if np.any(event[[2,3,5]]!=0):
            ax.axvspan(event[0]*(1 if steps else dt),event[1]*(1 if steps else dt),alpha=.15)
        ax.set(xlabel='Control step' if steps else 'Time [s]',ylabel=ylabel,title=name)
        ax.legend(fontsize=8);ax.grid(alpha=.2)
        for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=130)
        plt.close(fig);index.append(f'- [{name}]({name}.png)')
    chart('step_reward','Signed reward / transition',[(l,d,d['trace']['reward'][1:]) for l,d in sources],steps=True)
    chart('cumulative_reward','Cumulative signed reward',[(l,d,np.cumsum(d['trace']['reward'][1:])) for l,d in sources],steps=True)
    chart('reward_components','Signed reward / transition',[(l+' '+k,d,v[1:]) for l,d in sources for k,v in d['parts'].items()],steps=True)
    chart('cumulative_components','Cumulative signed reward component',[(l+' '+k,d,np.cumsum(v[1:])) for l,d in sources for k,v in d['parts'].items()],steps=True)
    chart('lateral_error','Path error [m]',[(l,d,d['trace']['path_features'][1:,0]) for l,d in sources])
    chart('heading_error','Heading error [rad]',[(l,d,d['trace']['path_features'][1:,1]) for l,d in sources])
    chart('roll','Roll [rad]',[(l,d,d['trace']['measurement'][1:,0]) for l,d in sources])
    curves=[];estimates=[]
    for label,d in sources:
        trace=d['trace'];true=trace['true_forward_speed'][1:];est=trace['measurement'][1:,5]*.1
        curves.extend([(label+' true',d,true),(label+' wheel estimate',d,est),
                       (label+' request',d,trace['motion_command'][:-1,1])])
        estimates.append((label,d,est-true))
    chart('speed','Speed [m/s]',curves);chart('speed_estimation_error','Estimate minus true [m/s]',estimates)
    chart('return_elapsed','Return elapsed [s]',[(l,d,d['trace']['return_state'][1:,3]) for l,d in sources])
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    from .path import BendConfig,bend_table,CircleConfig,circle_reference,FigureEightConfig,eight_reference
    if config.get('bend'):reference=reference_table(config)[:,1:3]
    elif config.get('circle'):reference=circle_reference(CircleConfig(**config['circle']))
    else:reference=eight_reference(FigureEightConfig(**config['figure_eight']))
    ax.plot(reference[:,0],reference[:,1],ls='--',label='Original reference path')
    for label,d in sources:
        xy=d['trace']['pose'][:,:2];ax.plot(xy[:,0],xy[:,1],label=label)
        ax.plot(xy[-1,0],xy[-1,1],marker='x' if d['trace']['terminated'][-1] else 'o')
    ax.set(xlabel='World X [m]',ylabel='World Y [m]',title='Original path and recorded motion');ax.set_aspect('equal');ax.legend()
    for ext in ('png','pdf'):fig.savefig(out/('trajectory.'+ext),dpi=130)
    plt.close(fig);index.append('- [XY trajectory](trajectory.png)')
    (out/'INDEX.md').write_text('\n'.join(index)+'\n')
    return result


def generate_panel(root):
    root=Path(root);out=root/'analysis/reward_breakdown';out.mkdir(parents=True,exist_ok=True)
    declaration=json.loads((root/'declaration.json').read_text())
    panel_cfg=json.loads((root/'frozen/panel.json').read_text())
    panels=sorted((root/'evaluation').glob('alpha_*/seed_*'))
    expected={(float(a),int(s)) for a in declaration['priority_alphas'] for s in panel_cfg['evaluation_seeds']}
    seen=set();rows=[];index=['# Complete alpha/path tracking diagnostics','']
    checkpoints=set()
    for panel in panels:
        d=json.loads((panel/'declaration.json').read_text());pair=(float(d['priority_alpha_override']),int(d['panel']['seed']))
        if pair in seen:raise ValueError('duplicate alpha/seed panel')
        seen.add(pair);checkpoints.add(d['checkpoint'])
        if {p.parent.name for p in panel.glob('*/residual')}!=set(d['scenarios']):raise ValueError('incomplete scenarios')
        for case in d['scenarios']:
            path=panel/case/'residual';result=write_diagnostics(path,baseline=panel/case/'baseline')
            rows.append(dict(alpha=pair[0],seed=pair[1],scenario=case,**result))
            import os
            relative=os.path.relpath(path/'analysis/tracking/INDEX.md',out)
            index.append(f'- alpha={pair[0]}, seed={pair[1]}, {case}: [plots and data]({relative})')
    if seen!=expected or len(checkpoints)!=1:raise ValueError('incomplete alpha/seed or mixed checkpoint coverage')
    (out/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    index.insert(2,'- [Required paired trajectory and per-step reward overview](../comparison/INDEX.md)')
    (out/'INDEX.md').write_text('\n'.join(index)+'\n')
    write_panel_overview(root, rows)
    write_alpha_error_overview(root)
    return rows


def write_panel_overview(root, rows):
    """Always expose paired XY and per-step reward figures for every panel cell.

    Uses already audited traces; never advances physics or pads failed episodes.
    Alpha values come from declarations, not folder indices.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .path import BendConfig, bend_table, CircleConfig, circle_reference, FigureEightConfig, eight_reference
    root=Path(root);out=root/'analysis/comparison';out.mkdir(parents=True,exist_ok=True)
    index=['# Paired geometric trajectories and per-step rewards','',
           'Every declared scenario, alpha and seed is included. Crosses mark physical failure.',
           'Reward uses a symmetric-log axis to retain terminal penalties and small normal rewards.','']
    manifest=[]
    for row in rows:
        alpha,seed,case=row['alpha'],row['seed'],row['scenario']
        # Resolve the declaration rather than interpreting alpha_2 as alpha=2.
        candidates=[]
        for panel in (root/'evaluation').glob('alpha_*/seed_*'):
            declaration=json.loads((panel/'declaration.json').read_text())
            if declaration['priority_alpha_override']==alpha and declaration['panel']['seed']==seed:
                candidates.append(panel)
        if len(candidates)!=1:raise ValueError('ambiguous overview panel')
        panel=candidates[0];sources=[]
        for policy in ('baseline','residual'):
            directory=panel/case/policy
            with np.load(directory/'trace.npz',allow_pickle=False) as saved:
                trace={k:saved[k] for k in saved.files}
            config=json.loads((directory/'declaration.json').read_text())['config']
            sources.append((policy,trace))
        stem=f'{case}_alpha_{alpha:g}_seed_{seed}'
        if config.get('bend'):reference=reference_table(config)[:,1:3]
        elif config.get('circle'):reference=circle_reference(CircleConfig(**config['circle']))
        else:reference=eight_reference(FigureEightConfig(**config['figure_eight']))
        manifest.append({k:row[k] for k in ('alpha','seed','scenario','trace_sha256','baseline_trace_sha256','declaration_sha256')})
        fig,axes=plt.subplots(1,3,figsize=(18,5),layout='constrained')
        axes[0].plot(reference[:,0],reference[:,1],'--',color='.5',label='Original reference')
        for label,tr in sources:
            line,=axes[0].plot(tr['pose'][:,0],tr['pose'][:,1],label=label)
            axes[0].plot(*tr['pose'][-1,:2],marker='x' if tr['terminated'][-1] else 'o',color=line.get_color())
            steps=np.arange(1,len(tr['time']))
            line,=axes[1].plot(steps,tr['reward'][1:],label=label)
            if tr['terminated'][-1]:axes[1].plot(steps[-1],tr['reward'][-1],'x',color=line.get_color())
            cumulative=np.cumsum(tr['reward'][1:])
            line,=axes[2].plot(steps,cumulative,label=label)
            if tr['terminated'][-1]:axes[2].plot(steps[-1],cumulative[-1],'x',color=line.get_color())
        event=sources[1][1]['event'][0]
        if np.any(event[[2,3,5]]!=0):
            for ax in axes[1:]:ax.axvspan(event[0],event[1],alpha=.15,color='grey')
        axes[0].set(xlabel='X [m]',ylabel='Y [m]',title='Trajectory');axes[0].set_aspect('equal')
        axes[1].set(xlabel='Control step',ylabel='Signed reward (symlog)',yscale='symlog',title='Per-step total reward')
        axes[1].set_yscale('symlog',linthresh=.05)
        axes[2].set(xlabel='Control step',ylabel='Cumulative signed reward',title='Cumulative total reward')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        fig.suptitle(f'{case} | alpha={alpha:g} | seed={seed}')
        for ext in ('png','pdf'):fig.savefig(out/f'{stem}.{ext}',dpi=140)
        plt.close(fig)
        detail=os.path.relpath(panel/case/'residual/analysis/tracking/INDEX.md',out)
        index.append(f'- {case}, alpha={alpha:g}, seed={seed}: [trajectory + step reward + cumulative reward]({stem}.png), [PDF]({stem}.pdf), [components and errors]({detail})')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'INDEX.md').write_text('\n'.join(index)+'\n')
    return out


def write_alpha_error_overview(root):
    """Overlay all declared preferences for each frozen task/seed, without simulation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'analysis/alpha_errors';out.mkdir(parents=True,exist_ok=True)
    groups={};sources={}
    for panel in sorted((root/'evaluation').glob('alpha_*/seed_*')):
        d=json.loads((panel/'declaration.json').read_text())
        for case in d['scenarios']:
            groups.setdefault((case,int(d['panel']['seed'])),[]).append((float(d['priority_alpha_override']),panel/case))
    index=['# Same-task alpha tracking errors','', 'Speed error = true forward speed minus pre-step target [m/s]. Position error = signed right-normal geometric cross-track error [m], not timed XY distance. Crosses mark real failures; shaded interval is sustained external disturbance.','']
    for (case,seed),entries in sorted(groups.items()):
        fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained');arrays={};checkpoint=set()
        reference_config=None
        for i,(alpha,path) in enumerate(sorted(entries)):
            for policy in ['baseline','residual']:
                p=path/policy;d=json.loads((p/'declaration.json').read_text());cfg=d['config']
                if reference_config is None:reference_config=cfg
                if cfg!=reference_config:raise ValueError('alpha overlay task configurations differ')
                with np.load(p/'trace.npz') as data:t={k:data[k] for k in data.files}
                sources[str(p.relative_to(root))]=hashlib.sha256((p/'trace.npz').read_bytes()).hexdigest()
                if policy=='residual':checkpoint.add(json.loads((path.parent/'declaration.json').read_text())['checkpoint'])
                x=t['time'][1:];values=[t['true_forward_speed'][1:]-t['motion_command'][:-1,1],t['path_features'][1:,0]]
                label=f'{policy} alpha={alpha}';color='black' if policy=='baseline' else f'C{i}'
                for ax,y,key in zip(axes,values,['speed_error','lateral_error']):
                    ax.plot(x,y,label=label,color=color,linestyle='--' if policy=='baseline' else '-',alpha=.65 if policy=='baseline' else 1)
                    if t['terminated'][-1]:ax.plot(x[-1],y[-1],'x',color=color)
                    arrays[f'{policy}_alpha_{alpha}_{key}']=y
                arrays[f'{policy}_alpha_{alpha}_time']=x
                if i==0 and policy=='residual' and np.any(t['event'][0,[2,3,5]]!=0):
                    for ax in axes:ax.axvspan(cfg['disturbance_start'],cfg['disturbance_start']+cfg['disturbance_duration'],color='.5',alpha=.15)
        if len(checkpoint)!=1:raise ValueError('mixed checkpoints in alpha overlay')
        title=f'{case} | seed={seed} | {Path(next(iter(checkpoint))).name}'
        fig.suptitle(title)
        for ax,label in zip(axes,['True speed error [m/s]','Signed cross-track error [m]']):
            ax.axhline(0,color='.6',lw=.7);ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[-1].set_xlabel('Time [s]');name=f'{case}_seed_{seed}'
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=140)
        plt.close(fig);np.savez_compressed(out/f'{name}.npz',**arrays)
        index.append(f'- {title}: [PNG]({name}.png) / [PDF]({name}.pdf) / [data]({name}.npz)')
    (out/'source_hashes.json').write_text(json.dumps(sources,indent=2));(out/'INDEX.md').write_text('\n'.join(index)+'\n')
    return out
