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


def speed_error(trace, config):
    target=trace['reference_command'][:-1,0] if config.get('timed_reference') is not None else trace['motion_command'][:-1,1]
    return trace['true_forward_speed'][1:]-target


def reference_xy(trace, config):
    if 'reference_geometry' in trace:
        stride=config['timed_reference'].get('geometry_stride',10)
        count=round(config['horizon_seconds']/config['controller']['dt']/stride)+1
        return trace['reference_geometry'][:count,1:3]
    if config.get('timed_reference') is not None:return trace['reference_pose'][:,:2]
    from .path import CircleConfig,circle_reference,FigureEightConfig,eight_reference
    if config.get('bend'):return reference_table(config)[:,1:3]
    if config.get('circle'):return circle_reference(CircleConfig(**config['circle']))
    return eight_reference(FigureEightConfig(**config['figure_eight']))


def audit_timed_reference(trace, dt, geometric=False):
    """Independent exact SE(2) integration from captured pre-step commands."""
    ref=np.asarray(trace['reference_pose']);command=np.asarray(trace['reference_command'])
    if ref.shape!=(len(trace['time']),3) or command.shape!=(len(ref),2):raise ValueError('timed reference shape mismatch')
    if not np.isfinite(ref).all() or not np.isfinite(command).all():raise ValueError('nonfinite timed reference')
    if not np.allclose(ref[0],trace['pose'][0],atol=2e-5,rtol=0):raise ValueError('reference initial pose mismatch')
    expected=np.empty_like(ref);expected[0]=ref[0]
    for i,(speed,yaw) in enumerate(command[:-1],1):
        x,y,heading=expected[i-1];angle=yaw*dt
        distance=speed*dt*np.sinc(angle/(2*np.pi))
        expected[i]=[x+distance*np.cos(heading+angle/2),y+distance*np.sin(heading+angle/2),heading+angle]
    difference=ref-expected;difference[:,2]=np.arctan2(np.sin(difference[:,2]),np.cos(difference[:,2]))
    if not np.allclose(difference,0,atol=2e-4,rtol=0):raise ValueError('independent timed reference integration mismatch')
    delta=trace['pose'][:,:2]-ref[:,:2];heading=ref[:,2]
    longitudinal=delta[:,0]*np.cos(heading)+delta[:,1]*np.sin(heading)
    right=delta[:,0]*np.sin(heading)-delta[:,1]*np.cos(heading)
    heading_error=trace['pose'][:,2]-heading;heading_error=np.arctan2(np.sin(heading_error),np.cos(heading_error))
    if geometric:
        audit_committed_reference(trace, dt)
        right=trace['path_features'][:,0];heading_error=trace['path_features'][:,1]
    for name,actual,want in [('longitudinal error',trace['longitudinal_error'],longitudinal),
                              ('lateral error',trace['path_features'][:,0],right),
                              ('heading error',trace['path_features'][:,1],heading_error),
                              ('yaw rate error',trace['yaw_rate_error'][1:],trace['yaw_rate_world'][1:]-command[:-1,1])]:
        if not np.allclose(actual,want,rtol=3e-5,atol=2e-4):raise ValueError('timed '+name+' mismatch')
    return float(np.max(np.abs(difference)))



def audit_committed_reference(trace, dt):
    """Rebuild projection from captured committed original path, not logged errors."""
    from .timed_reference import project_committed_geometry
    ref=np.asarray(trace['reference_pose']);cmd=np.asarray(trace['reference_command'])
    arc=np.r_[0.,np.cumsum(cmd[:-1,0]*dt)]
    table=np.column_stack((arc,ref,cmd))
    progress=0.;segment=0;features=[np.zeros(3)];all_progress=[0.]
    for tick in range(1,len(ref)):
        feature,progress,segment=project_committed_geometry(trace['pose'][tick],table,segment,progress,
            np.linalg.norm(trace['pose'][tick,:2]-trace['pose'][tick-1,:2]),tick,xp=np)
        features.append(feature);all_progress.append(progress)
    if (not np.allclose(features,trace['path_features'],rtol=3e-4,atol=3e-4)
            or not np.allclose(all_progress,trace['path_progress'],rtol=3e-4,atol=3e-4)):
        # Distant points can tie on adjacent segments in float32. Reconstruct
        # with the runtime arithmetic before rejecting; do not widen tolerance
        # or seed this replay from logged projection state/features.
        import jax
        import jax.numpy as jp
        with jax.default_device(jax.devices('cpu')[0]):
            reference=jp.asarray(ref);command=jp.asarray(cmd);pose=jp.asarray(trace['pose'])
            zero=jp.asarray(0.,dtype=command.dtype)
            arc=jax.lax.scan(lambda a,v:(a+v*dt,a+v*dt),zero,command[:-1,0])[1]
            committed=jp.column_stack((jp.concatenate((zero[None],arc)),reference,command))
            @jax.jit
            def replay():
                def tick(carry,i):
                    feature,progress,segment=project_committed_geometry(pose[i],committed,carry[1],carry[0],
                        jp.linalg.norm(pose[i,:2]-pose[i-1,:2]),i)
                    return (progress,segment),(feature,progress)
                return jax.lax.scan(tick,(zero,jp.int32(0)),jp.arange(1,len(ref)))[1]
            replay_features,replay_progress=replay()
            features=np.vstack((np.zeros(3),np.asarray(replay_features)))
            all_progress=np.r_[0.,np.asarray(replay_progress)]
    if not np.allclose(features,trace['path_features'],rtol=3e-4,atol=3e-4):
        raise ValueError('geometric path projection mismatch')
    if not np.allclose(all_progress,trace['path_progress'],rtol=3e-4,atol=3e-4):
        raise ValueError('geometric path progress mismatch')
    return float(np.max(np.abs(np.asarray(features)-trace['path_features'])))


def trace_summary(trace, config):
    c=TrackingConfig(**config['tracking']);dt=config['controller']['dt']
    t=trace['time'];rs=trace['return_state'][-1]
    failed=bool(trace['terminated'][-1]);full=bool(t[-1]+1e-6>=config['horizon_seconds'])
    final=bool(full and not failed and not rs[2] and rs[4]+1e-6>=c.hold_seconds)
    qualified=final and not bool(rs[7])
    present=bool(np.any(trace['event'][0,[2,3,5]]!=0))
    ev=speed_error(trace,config)
    path=trace['path_features'][1:]
    bv,by=tolerances(trace['priority_alpha'][:-1],c,xp=np)
    if c.shrink_tolerances:
        fraction=np.clip(trace['return_state'][1:,3]/(c.return_seconds-c.hold_seconds),0,1)
        bv=bv+fraction*(c.final_speed_tolerance-bv)
        by=by+fraction*(c.final_lateral_tolerance-by)
    mature=t[:-1]+1e-7>=c.start_seconds
    def fraction(mask):return float(np.mean(mask[mature])) if np.any(mature) else None
    result=dict(
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
    if c.timed:
        result.update(longitudinal_error_rmse_m=float(np.sqrt(np.mean(trace['longitudinal_error'][1:]**2))),
                      xy_error_rmse_m=float(np.sqrt(np.mean(np.sum((trace['pose'][1:,:2]-trace['reference_pose'][1:,:2])**2,axis=1)))),
                      yaw_rate_error_rmse_rad_s=float(np.sqrt(np.mean(trace['yaw_rate_error'][1:]**2))))
        if not c.geometric:
            result['recovery_criteria'].update(longitudinal_m=c.final_longitudinal_tolerance,yaw_rate_rad_s=c.final_yaw_rate_tolerance)
            result['scope']='independent timed trajectory and command tracking; common final hold and no missed return deadline; no safety/generalization claim'
        else:
            result['scope']='original geometric path and requested speed; temporal XY/along/yaw errors are diagnostics only; no safety/generalization claim'
    if config.get('timed_reference') and config['timed_reference'].get('mode')=='geometry':
        result['reference_mode']='fixed geometric curve; no timed along-track or yaw-rate target'
        result['path_progress_m']=float(trace['path_progress'][-1])
    return result


def audit_trace(path):
    path=Path(path);decl=json.loads((path/'declaration.json').read_text());config=decl['config']
    c=TrackingConfig(**config['tracking']);dt=config['controller']['dt']
    with np.load(path/'trace.npz',allow_pickle=False) as saved:tr={k:saved[k] for k in saved.files}
    if len(tr['time'])<2:raise ValueError('tracking audit needs at least one transition')
    if not np.allclose(np.diff(tr['time']),dt,rtol=1e-5,atol=1e-6):raise ValueError('nonuniform trace timing')
    timed=config.get('timed_reference') is not None
    geometric=config.get('timed_reference',{}).get('mode')=='geometry' if timed else False
    reference_error=(audit_geometric_reference(tr,config,decl['seed']) if geometric else audit_timed_reference(tr,dt,c.geometric)) if timed else None
    state=initial_return(xp=np);parts={};max_state=0.
    ev=speed_error(tr,config)
    for i in range(1,len(tr['time'])):
        event=tr['event'][i-1];tick=round(float(tr['time'][i-1])/dt)
        state,terms=transition(state,roll=tr['measurement'][i,0],roll_rate=tr['measurement'][i,1],
            speed_error=ev[i-1],
            lateral_error=tr['path_features'][i,0],heading_error=tr['path_features'][i,1],
            action=tr['effective_action'][i],alpha=tr['priority_alpha'][i-1],dt=dt,
            alive_rate=config['alive_reward_rate'],failure_penalty=config['failure_penalty'],
            failed=tr['terminated'][i],enabled=tick*dt>=c.start_seconds,config=c,xp=np,
            **({'longitudinal_error':tr['longitudinal_error'][i],'yaw_rate_error':tr['yaw_rate_error'][i]} if timed else {}))
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
            'max_reward_error':error,'max_return_state_error':max_state,'max_reference_error':reference_error,
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
        scoring_alpha=float(tr['priority_alpha'][0])
        if not np.allclose(base['trace']['priority_alpha'],scoring_alpha):
            base=rescore_trace(base,scoring_alpha)
            result['baseline_scoring_scope']='one baseline physical trace, reward replayed at candidate alpha'
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
    sources=[('residual' if baseline else 'recorded',data)]+([('baseline',base)] if base else [])
    arrays={}
    for label,source in sources:
        trace=source['trace']
        fields=dict(time=trace['time'][1:],speed_error=speed_error(trace,config),lateral_error=trace['path_features'][1:,0],heading_error=trace['path_features'][1:,1])
        if config.get('timed_reference') is not None:
            fields.update(reference_pose=trace['reference_pose'][1:],reference_command=trace['reference_command'][:-1],
                          longitudinal_error=trace['longitudinal_error'][1:],yaw_rate_error=trace['yaw_rate_error'][1:],
                          xy_error=np.linalg.norm(trace['pose'][1:,:2]-trace['reference_pose'][1:,:2],axis=1))
        arrays.update({label+'_'+key:value for key,value in fields.items()})
    np.savez_compressed(out/'tracking_errors.npz',**arrays)
    write_step_csv(data,out/'steps.csv')
    if base:write_step_csv(base,out/'baseline_steps.csv')
    if not plots:return result
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    index=['# Frozen geometric tracking diagnostics','',
           'Every curve ends at the true recorded endpoint. Short failed returns use different exposure.',
           'The target is the original global reference; timed trajectories retain their independent clock.',
           '- [Paired tracking error data](tracking_errors.npz)', '- [Reward component data](components.npz)','']
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
        ax.legend(fontsize=8,ncol=2 if len(curves)>8 else 1);ax.grid(alpha=.2)
        for ext in ('png','pdf'):fig.savefig(out/(name+'.'+ext),dpi=130)
        plt.close(fig);index.append(f'- [{name}]({name}.png)')
    chart('step_reward','Signed reward / transition',[(l,d,d['trace']['reward'][1:]) for l,d in sources],steps=True)
    chart('cumulative_reward','Cumulative signed reward',[(l,d,np.cumsum(d['trace']['reward'][1:])) for l,d in sources],steps=True)
    chart('reward_components','Signed reward / transition',[(l+' '+k,d,v[1:]) for l,d in sources for k,v in d['parts'].items()],steps=True)
    chart('cumulative_components','Cumulative signed reward component',[(l+' '+k,d,np.cumsum(v[1:])) for l,d in sources for k,v in d['parts'].items()],steps=True)
    chart('lateral_error','Path error [m]',[(l,d,d['trace']['path_features'][1:,0]) for l,d in sources])
    chart('heading_error','Heading error [rad]',[(l,d,d['trace']['path_features'][1:,1]) for l,d in sources])
    if config.get('timed_reference') is not None:
        chart('longitudinal_error',('Projection tangential residual [m], not time lag' if config['timed_reference'].get('mode')=='geometry' else 'Along-track error [m]'),[(l,d,d['trace']['longitudinal_error'][1:]) for l,d in sources])
        chart('xy_error',('Geometric projection distance [m]' if config['timed_reference'].get('mode')=='geometry' else 'Timed XY distance [m]'),[(l,d,np.linalg.norm(d['trace']['pose'][1:,:2]-d['trace']['reference_pose'][1:,:2],axis=1)) for l,d in sources])
        chart('yaw_rate_error',('Yaw minus path feedforward [rad/s], diagnostic only' if config['timed_reference'].get('mode')=='geometry' else 'Yaw-rate error [rad/s]'),[(l,d,d['trace']['yaw_rate_error'][1:]) for l,d in sources])
        chart('yaw_rate','World yaw rate [rad/s]',[(l+' '+kind,d,d['trace']['yaw_rate_world'][1:] if kind=='actual' else d['trace']['reference_command'][:-1,1]) for l,d in sources for kind in ('actual','request')])
    chart('roll','Roll [rad]',[(l,d,d['trace']['measurement'][1:,0]) for l,d in sources])
    curves=[];estimates=[]
    for label,d in sources:
        trace=d['trace'];true=trace['true_forward_speed'][1:];est=trace['measurement'][1:,5]*.1
        curves.extend([(label+' true',d,true),(label+' wheel estimate',d,est),
                       (label+' request',d,trace['reference_command'][:-1,0] if config.get('timed_reference') is not None else trace['motion_command'][:-1,1])])
        estimates.append((label,d,est-true))
    chart('speed','Speed [m/s]',curves);chart('speed_estimation_error','Estimate minus true [m/s]',estimates)
    chart('return_elapsed','Return elapsed [s]',[(l,d,d['trace']['return_state'][1:,3]) for l,d in sources])
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    reference=reference_xy(tr,config)
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
        reference=reference_xy(max(sources,key=lambda item:len(item[1]['time']))[1],config)
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
        fig.suptitle(f'{case} | alpha={alpha:g} | seed={seed} | checkpoint={Path(declaration["checkpoint"]).name}')
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
    index=['# Same-task alpha tracking errors','', 'Speed error = true forward speed minus pre-step target [m/s]. Position error uses the frozen reference: geometric cross-track for legacy paths; right and along-track components at the current reference time for legacy timed trajectories. Geometric mode uses continuous original-path projection; timed along/yaw are diagnostic only. Crosses mark real failures; shaded interval is sustained external disturbance.','']
    for (case,seed),entries in sorted(groups.items()):
        first_config=json.loads((entries[0][1]/'residual/declaration.json').read_text())['config']
        timed=first_config.get('timed_reference') is not None
        keys=['speed_error','lateral_error']+(['longitudinal_error','yaw_rate_error'] if timed else [])
        labels=['True speed error [m/s]','Signed right error [m]']+(['Along-track error [m]','Yaw-rate error [rad/s]'] if timed else [])
        fig,axes=plt.subplots(len(keys),1,figsize=(10,3.5*len(keys)),sharex=True,layout='constrained');arrays={};checkpoint=set()
        reference_config=None;trajectory_sources=[];initial_state=None;event=None
        for i,(alpha,path) in enumerate(sorted(entries)):
            for policy in ['baseline','residual']:
                p=path/policy;d=json.loads((p/'declaration.json').read_text());cfg=d['config']
                if reference_config is None:reference_config=cfg
                if cfg!=reference_config:raise ValueError('alpha overlay task configurations differ')
                with np.load(p/'trace.npz') as data:t={k:data[k] for k in data.files}
                current_initial=(t['qpos'][0],t['qvel'][0])
                if initial_state is None:initial_state=current_initial;event=t['event'][0]
                if not all(np.allclose(a,b,rtol=0,atol=1e-7) for a,b in zip(initial_state,current_initial)) or not np.array_equal(event,t['event'][0]):
                    raise ValueError('alpha overlay initial state or event differs')
                trajectory_sources.append((alpha,policy,t))
                arrays[f'{policy}_alpha_{alpha}_xy']=t['pose'][:,:2]
                arrays[f'{policy}_alpha_{alpha}_trajectory_time']=t['time']
                arrays[f'{policy}_alpha_{alpha}_terminated']=t['terminated']
                sources[str(p.relative_to(root))]=hashlib.sha256((p/'trace.npz').read_bytes()).hexdigest()
                if policy=='residual':checkpoint.add(json.loads((path.parent/'declaration.json').read_text())['checkpoint'])
                x=t['time'][1:];values=[speed_error(t,cfg),t['path_features'][1:,0]]
                if timed:values.extend([t['longitudinal_error'][1:],t['yaw_rate_error'][1:]])
                label=f'{policy} alpha={alpha}';color='black' if policy=='baseline' else f'C{i}'
                for ax,y,key in zip(axes,values,keys):
                    ax.plot(x,y,label=label,color=color,linestyle='--' if policy=='baseline' else '-',alpha=.65 if policy=='baseline' else 1)
                    if t['terminated'][-1]:ax.plot(x[-1],y[-1],'x',color=color)
                    arrays[f'{policy}_alpha_{alpha}_{key}']=y
                arrays[f'{policy}_alpha_{alpha}_time']=x
                if i==0 and policy=='residual' and np.any(t['event'][0,[2,3,5]]!=0):
                    for ax in axes:ax.axvspan(t['event'][0,0]*cfg['controller']['dt'],t['event'][0,1]*cfg['controller']['dt'],color='.5',alpha=.15)
        if len(checkpoint)!=1:raise ValueError('mixed checkpoints in alpha overlay')
        title=f'{case} | seed={seed} | {Path(next(iter(checkpoint))).name}'
        fig.suptitle(title)
        for ax,label in zip(axes,labels):
            ax.axhline(0,color='.6',lw=.7);ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[-1].set_xlabel('Time [s]');name=f'{case}_seed_{seed}'
        for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=140)
        plt.close(fig)
        fig,xyaxes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
        reference_trace=max(trajectory_sources,key=lambda item:len(item[2]['time']))[2]
        ref=reference_xy(reference_trace,reference_config);arrays['reference_xy']=ref
        residuals=[t for a,p,t in trajectory_sources if p=='residual']
        n=min(len(t['time']) for t in residuals)
        if any(not np.allclose(t['time'][:n],residuals[0]['time'][:n]) for t in residuals):raise ValueError('alpha trajectory time grids differ')
        stack=np.stack([t['pose'][:n,:2] for t in residuals])
        spread=np.max(np.linalg.norm(stack[:,None]-stack[None,:],axis=-1),axis=(0,1))
        peak=int(np.argmax(spread));center=stack[:,peak].mean(axis=0)
        radius=max(.15,float(spread[peak])*1.5)
        arrays['same_time_max_alpha_separation_m']=spread
        arrays['same_time_comparison_time']=residuals[0]['time'][:n]
        arrays['zoom_time_s']=np.asarray(residuals[0]['time'][peak])
        for ax in xyaxes:ax.plot(ref[:,0],ref[:,1],color='.5',ls=':',label='Timed reference' if timed else 'Reference')
        baseline_seen=[]
        for alpha,policy,t in trajectory_sources:
            if policy=='baseline' and any(t['pose'].shape==v.shape and np.allclose(t['pose'],v,rtol=0,atol=1e-7) for v in baseline_seen):continue
            if policy=='baseline':baseline_seen.append(t['pose'])
            color='black' if policy=='baseline' else f'C{[a for a,p,tr in trajectory_sources if p=="residual"].index(alpha)}'
            label='ECBC+ESO' if policy=='baseline' else f'Residual alpha={alpha:g}'
            for ax in xyaxes:
                ax.plot(t['pose'][:,0],t['pose'][:,1],label=label,color=color,ls='--' if policy=='baseline' else '-',lw=1.5)
                ax.plot(*t['pose'][-1,:2],marker='x' if t['terminated'][-1] else 'o',color=color,ms=6)
                if np.any(t['event'][0,[2,3,5]]!=0):
                    active=(t['time']>=event[0]*cfg['controller']['dt'])&(t['time']<=event[1]*cfg['controller']['dt'])
                    ax.plot(t['pose'][active,0],t['pose'][active,1],color=color,lw=4,alpha=.45)
        xyaxes[0].set_title('Same-scene XY trajectories')
        xyaxes[1].set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),
                      title=f'Zoom at max same-time alpha separation: t={residuals[0]["time"][peak]:.2f}s')
        for ax in xyaxes:ax.set_aspect('equal');ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle(title+'\nThick segments: disturbance; x: failure; o: recorded endpoint. Zoom selected by maximum alpha separation.')
        for ext in ['png','pdf']:fig.savefig(out/f'{name}_xy.{ext}',dpi=150)
        plt.close(fig);np.savez_compressed(out/f'{name}.npz',**arrays)
        index.append(f'- {title}: **[same-scene alpha XY trajectories]({name}_xy.png)** / [XY PDF]({name}_xy.pdf)')
        index.append(f'- {title}: [PNG]({name}.png) / [PDF]({name}.pdf) / [data]({name}.npz)')
    (out/'source_hashes.json').write_text(json.dumps(sources,indent=2));(out/'INDEX.md').write_text('\n'.join(index)+'\n')
    return out


def write_step_csv(data, destination):
    """Human-readable actual transitions; reset row excluded, failure retained."""
    import csv
    trace=data['trace'];config=data['config'];parts=data['parts']
    n=len(trace['time'])-1
    fields=dict(step=np.arange(1,n+1),time_s=trace['time'][1:],alpha=trace['priority_alpha'][:-1],
        x_m=trace['pose'][1:,0],y_m=trace['pose'][1:,1],
        true_speed_m_s=trace['true_forward_speed'][1:],speed_error_m_s=speed_error(trace,config),
        lateral_error_m=trace['path_features'][1:,0],heading_error_rad=trace['path_features'][1:,1],
        reward=trace['reward'][1:],cumulative_reward=np.cumsum(trace['reward'][1:]),
        terminated=trace['terminated'][1:],end_code=trace['end_code'][1:])
    if config.get('timed_reference') is not None:
        fields.update(reference_x_m=trace['reference_pose'][1:,0],reference_y_m=trace['reference_pose'][1:,1],
            reference_speed_m_s=trace['reference_command'][:-1,0],reference_yaw_rate_rad_s=trace['reference_command'][:-1,1],
            longitudinal_error_m=trace['longitudinal_error'][1:],yaw_rate_error_rad_s=trace['yaw_rate_error'][1:])
    for key,value in parts.items():
        fields['reward_'+key]=value[1:]
        fields['cumulative_'+key]=np.cumsum(value[1:])
    with Path(destination).open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(fields)
        writer.writerows(zip(*fields.values()))


def audit_geometric_reference(trace, config, seed):
    """Independent NumPy integration plus sequential projection replay.

    The original curve is verified from the declared seed/schedule, not from a
    moving trace of projection points. No dynamics are rerun or rebased.
    """
    import math
    import jax
    from dataclasses import replace
    from .timed_reference import TimedReferenceConfig, schedule, reference_trace, project_geometry
    c=TimedReferenceConfig(**config['timed_reference']);dt=config['controller']['dt']
    rows=np.asarray(schedule(jax.random.fold_in(jax.random.PRNGKey(seed),51),c,config['speed_reference']))
    if not np.allclose(rows,trace['geometric_schedule'],rtol=0,atol=1e-6):
        raise ValueError('geometric schedule does not match declared reset seed/config')
    ticks=math.ceil(math.ceil((config['horizon_seconds']+c.extension_seconds)/dt)/c.geometry_stride)*c.geometry_stride
    rebuilt=reference_trace(replace(c,fixed=tuple(map(tuple,rows))),dt,ticks*dt,
                            config['speed_reference'],trace['pose'][0])
    cmd=rebuilt['reference_command'];arc=np.r_[0.,np.cumsum(cmd[:-1,0]*dt)]
    original=np.column_stack((arc,rebuilt['reference_pose'],cmd[:,1]/cmd[:,0]))[::c.geometry_stride]
    saved=trace['reference_geometry']
    if saved.shape!=original.shape or not np.allclose(saved,original,rtol=3e-5,atol=2e-3):
        raise ValueError('fixed geometric curve differs from independently integrated schedule')
    progress=0.;max_error=float(np.max(np.abs(saved-original)))
    for i in range(1,len(trace['time'])):
        window=2*np.linalg.norm(trace['pose'][i,:2]-trace['pose'][i-1,:2])+c.projection_margin
        progress,foot,k=project_geometry(trace['pose'][i],saved,progress,window,xp=np)
        delta=foot-trace['reference_pose'][i];delta[2]=np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
        if not np.allclose(delta,0,atol=3e-4,rtol=0) or not np.isclose(progress,trace['path_progress'][i],atol=3e-4):
            raise ValueError('geometric projection replay mismatch')
        d=trace['pose'][i]-foot
        ey=np.sin(foot[2])*d[0]-np.cos(foot[2])*d[1]
        ep=np.arctan2(np.sin(d[2]),np.cos(d[2]))
        if not np.allclose(trace['path_features'][i],[ey,ep,k],atol=3e-4,rtol=3e-5):
            raise ValueError('geometric tracking features mismatch')
    # Speed still follows the exogenous clock, independently of progress/alpha.
    if not np.allclose(trace['reference_command'][:,0],cmd[:len(trace['time']),0],atol=2e-4,rtol=3e-5):
        raise ValueError('geometric task speed request changed with vehicle progress')
    return max_error


def rescore_trace(data, alpha):
    """Counterfactual reward replay of audited physics, not a new policy rollout.

    Used both for cross-alpha scoring and reuse of a single baseline trajectory.
    Source trace and captured observations are not overwritten on disk.
    """
    if not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('invalid scoring alpha')
    tr={k:v.copy() for k,v in data['trace'].items()};config=data['config']
    c=TrackingConfig(**config['tracking']);dt=config['controller']['dt']
    state=initial_return(xp=np);parts={};returns=[np.asarray(return_observation(state,c,xp=np))]
    ev=speed_error(tr,config)
    for i in range(1,len(tr['time'])):
        state,terms=transition(state,roll=tr['measurement'][i,0],roll_rate=tr['measurement'][i,1],
            speed_error=ev[i-1],lateral_error=tr['path_features'][i,0],heading_error=tr['path_features'][i,1],
            action=tr['effective_action'][i],alpha=alpha,dt=dt,alive_rate=config['alive_reward_rate'],
            failure_penalty=config['failure_penalty'],failed=tr['terminated'][i],
            enabled=round(tr['time'][i-1]/dt)*dt>=c.start_seconds,config=c,xp=np,
            **({'longitudinal_error':tr['longitudinal_error'][i],'yaw_rate_error':tr['yaw_rate_error'][i]} if c.timed else {}))
        for k,v in terms.items():parts.setdefault(k,[0.]).append(float(v))
        returns.append(np.asarray(return_observation(state,c,xp=np)))
    parts={k:np.asarray(v) for k,v in parts.items()}
    tr['reward']=sum(parts.values());tr['priority_alpha']=np.full_like(tr['priority_alpha'],alpha)
    tr['return_state']=np.asarray(returns)
    for k,v in parts.items():tr['reward_'+k]=v
    return {**data,'trace':tr,'parts':parts,'summary':trace_summary(tr,config),
            'scoring_alpha':float(alpha),'scoring_scope':'frozen physical trace counterfactual reward replay'}


def review_checkpoint(training_run, panel_path, output, *, checkpoint=None, selection=None, compact=False):
    """Explicit post-training review; no retraining, no automatic extra scenarios.

    One baseline physical run per scenario is reused with counterfactual reward
    replay at each alpha. A training-reward best is not called a qualified model.
    """
    import csv
    import shutil
    import zipfile
    from dataclasses import asdict,replace
    from .env import RecoveryEnv,config_from_dict
    from .network import make_policy_identity,load_policy
    from .evaluation import evaluate
    training_run=Path(training_run).resolve();output=Path(output).resolve()
    source=json.loads((training_run/'declaration.json').read_text())
    status=json.loads((training_run/'status.json').read_text())
    if not status.get('complete'):raise ValueError('finish the declared training stage before review')
    selection_meta=None
    if selection is not None:
        if checkpoint is not None:raise ValueError('use selection OR explicit checkpoint, not both')
        selection_meta=json.loads(Path(selection).read_text())
        if (selection_meta.get('schema')!='sttw_fixed_checkpoint_comparison_v1'
                or not selection_meta.get('complete')
                or selection_meta.get('training_declaration_sha256')!=hashlib.sha256((training_run/'declaration.json').read_bytes()).hexdigest()):
            raise ValueError('incomplete or mismatched fixed checkpoint selection')
        selected=selection_meta.get('selected_checkpoint')
        entry=next((e for e in selection_meta.get('candidates',[]) if e.get('checkpoint')==selected),None)
        if not entry or hashlib.sha256((Path(selected)/'actor.msgpack').read_bytes()).hexdigest()!=entry.get('actor_sha256'):
            raise ValueError('selected Actor changed after comparison')
    else:
        selected=checkpoint or status.get('best_reward_checkpoint')
    if not selected:raise ValueError('no scored best checkpoint; specify --checkpoint explicitly')
    checkpoint=Path(selected).resolve()
    cfg=config_from_dict(source['task']);panel=json.loads(Path(panel_path).read_text())
    if cfg.timed_reference is None or cfg.timed_reference.mode!='geometry':
        raise ValueError('this runner requires the explicit geometric reference task')
    alphas=panel['alphas'];seed=panel['seed'];cases=panel['scenarios']
    if type(seed) is not int or not alphas or len(set(alphas))!=len(alphas) or any(not np.isfinite(a) or not 0<=a<=1 for a in alphas):
        raise ValueError('invalid review seed/alphas')
    if not cases or len({c['name'] for c in cases})!=len(cases):raise ValueError('duplicate/empty scenarios')
    configs=[]
    for case in cases:
        name=case['name'];ev=case.get('event',{})
        if not name or '/' in name or '\\' in name or name in ('.','..'):raise ValueError('invalid scenario name')
        start=ev.get('start',4.);duration=ev.get('duration',1.)
        if start+duration+cfg.tracking.return_seconds>cfg.horizon_seconds+1e-7:
            raise ValueError('scenario does not leave the declared return observation window')
        configs.append(replace(cfg,random_events=None,timed_reference=replace(cfg.timed_reference,fixed=case['commands']),
            disturbance_start=start,disturbance_duration=duration,disturbance_force=ev.get('force',0.),
            disturbance_steer_rate=ev.get('steer_rate',0.),disturbance_rear_torque=ev.get('rear_torque',0.),
            disturbance_waveform=ev.get('waveform','constant')))
    output.mkdir(parents=True,exist_ok=False)
    declaration={'training_run':str(training_run),'checkpoint':str(checkpoint),'panel':panel,
        'source_task':source['task'],'residual_episodes':len(cases)*len(alphas),'baseline_physical_episodes':len(cases),
        'selection':('fixed completed-episode candidate subset' if selection_meta else 'explicit checkpoint or training-sampled reward best; not certified successful'),
        'fixed_selection_development_gates_passed':selection_meta.get('development_gates_passed') if selection_meta else None,
        'baseline_reuse':'same physical trace; reward is replayed at each scoring alpha',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    (output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    def progress(phase,**kwargs):(output/'status.json').write_text(json.dumps({'phase':phase,**kwargs},indent=2)+'\n')
    rows=[];cross=[];index=['# Geometric speed/path reward review','',
        'Selection provenance is in declaration.json; physical failure, deadline and final hold determine task qualification.',
        'Cross-score maxima need not lie on the diagonal. Shared good trajectories are valid at every alpha.','']
    try:
        for case,c in zip(cases,configs):
            name=case['name'];root=output/name;root.mkdir()
            env=RecoveryEnv(c,backend='cpu')
            expected=make_policy_identity(env.bundle.identity,asdict(cfg),cfg.observation.history_steps)
            policy=load_policy(checkpoint,expected=expected)
            identity={**expected,'checkpoint':str(checkpoint),
                      'checkpoint_sidecar_sha256':hashlib.sha256((checkpoint/'identity.json').read_bytes()).hexdigest(),
                      'evaluation_overrides':'explicit fixed reference schedule and event only'}
            progress('baseline',scenario=name,complete=False)
            basepath=root/'baseline'
            evaluate(env,basepath,seed=seed,priority_alpha=0.)
            base=audit_trace(basepath);candidates=[]
            for a in alphas:
                progress('residual',scenario=name,alpha=a,complete=False)
                path=root/f'alpha_{a:g}'
                evaluate(env,path,seed=seed,priority_alpha=a,policy=policy,policy_identity=identity)
                data=audit_trace(path);candidates.append((a,data))
                result=write_diagnostics(path,baseline=basepath,audited=data)
                rows.append({'scenario':name,'alpha':a,'seed':seed,'physical_failure':bool(data['trace']['terminated'][-1]),**result})
                index.append(f'- {name}, alpha={a}: [{path.name}/analysis/tracking/INDEX.md]({name}/{path.name}/analysis/tracking/INDEX.md)')
            for score_alpha in alphas:
                for actor_alpha,data in candidates:
                    score=rescore_trace(data,score_alpha)
                    cross.append({'scenario':name,'scoring_alpha':score_alpha,'actor_alpha':actor_alpha,
                                  'return':float(score['trace']['reward'][1:].sum()),
                                  'scope':'same recorded physics, counterfactual preference score'})
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
            ref=reference_xy(base['trace'],base['config'])
            ax.plot(ref[:,0],ref[:,1],ls='--',label='original fixed path')
            for label,data in [('ECBC+ESO',base)]+[(f'alpha={a:g}',d) for a,d in candidates]:
                t=data['trace'];xy=t['pose'][:,:2]
                line,=ax.plot(xy[:,0],xy[:,1],label=label)
                ev=t['event'][0];dt=c.controller.dt
                during=(t['time']>=ev[0]*dt)&(t['time']<=ev[1]*dt)
                if np.any(ev[[2,3,5]]!=0) and np.any(during):
                    ax.plot(xy[during,0],xy[during,1],linewidth=3,alpha=.6,color=line.get_color())
                ax.plot(xy[-1,0],xy[-1,1],marker='x' if t['terminated'][-1] else 'o')
            ax.set(xlabel='X [m]',ylabel='Y [m]',title=name+' | '+checkpoint.name)
            ax.set_aspect('equal');ax.legend()
            for ext in ('png','pdf'):fig.savefig(root/('alpha_trajectories.'+ext),dpi=140)
            plt.close(fig)
            index.append(f'- {name}: [all-alpha trajectory overlay]({name}/alpha_trajectories.png)')
        (output/'summary.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
        with (output/'cross_scores.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(cross[0]));w.writeheader();w.writerows(cross)
        evidence=output/'training_evidence';evidence.mkdir()
        for name in ('declaration.json','status.json','best_model.json','metrics.jsonl','setup_timings.json'):
            p=training_run/name
            if p.exists():shutil.copy2(p,evidence/name)
        if selection_meta is not None:(evidence/'fixed_selection.json').write_text(json.dumps(selection_meta,indent=2)+'\n')
        for name in ('actor.msgpack','identity.json','training.json'):
            p=checkpoint/name
            if p.exists():shutil.copy2(p,evidence/('selected_'+name))
        index.extend(['','- [All metrics](summary.json)','- [Counterfactual cross scores](cross_scores.csv)'])
        (output/'INDEX.md').write_text('\n'.join(index)+'\n')
        progress('complete',complete=True,residual_episodes=len(cases)*len(alphas),baseline_physical_episodes=len(cases))
        if compact:
            archive=compact_review_archive(output,training_run=training_run)
        else:
            archive=output.with_suffix('.zip')
            with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
                for p in sorted(output.rglob('*')):
                    if p.is_file():z.write(p,p.relative_to(output.parent))
        print(json.dumps({'review':str(output),'archive':str(archive),'archive_bytes':archive.stat().st_size,'complete':True}),flush=True)
        return rows
    except Exception as exc:
        progress('error',complete=False,error=repr(exc));raise


def compact_review_archive(output,*,training_run=None):
    """Small transfer bundle, full local evidence untouched; no temporal thinning.

    Numeric traces retain each actual control step and the fields required for
    reward/reference analysis. High-dimensional Actor observations, full qpos/
    qvel and detailed actuator/force tensors are intentionally excluded. Named
    trace_compact.npz, never masquerading as the original trace.npz.
    """
    import io,zipfile
    output=Path(output).resolve();archive=output.with_name(output.name+'_compact.zip')
    excluded={'observation','qpos','qvel','actuator_force','generalized_actuator_force',
              'actuator_velocity','actuator_ctrl','applied_generalized_force','applied_wrench'}
    manifest={'scope':'partial transfer package; full local raw traces unchanged; all actual timesteps retained',
              'excluded_trace_fields':sorted(excluded),'traces':[],'not_included':'per-alpha PDF/PNG duplicates, optimizer snapshots, full Actor input histories'}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ('summary.json','cross_scores.csv','declaration.json','status.json'):
            p=output/name
            if p.exists():z.write(p,name)
        for p in sorted((output/'training_evidence').glob('*')):
            if p.is_file():z.write(p,'training_evidence/'+p.name)
        for p in sorted(output.glob('*/alpha_trajectories.png')):z.write(p,str(p.relative_to(output)))
        for p in sorted(output.glob('*/*/trace.npz')):
            relative=p.parent.relative_to(output)
            with np.load(p,allow_pickle=False) as original:
                data={k:original[k] for k in original.files if k not in excluded}
            buffer=io.BytesIO();np.savez_compressed(buffer,**data)
            z.writestr(str(relative/'trace_compact.npz'),buffer.getvalue())
            manifest['traces'].append({'path':str(relative),'full_trace_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                                       'samples':len(data['time']),'included_fields':sorted(data)})
            for name in ('declaration.json','commands.json','event.json','summary.json'):
                q=p.parent/name
                if q.exists():z.write(q,str(relative/name))
        if training_run is not None:
            for p in sorted((Path(training_run)/'tensorboard').glob('events.out.tfevents.*')):
                z.write(p,'training_evidence/tensorboard/'+p.name)
        z.writestr('compact_manifest.json',json.dumps(manifest,indent=2)+'\n')
        z.writestr('README.txt','Compact transfer: summary/cross-score tables, selected Actor, training evidence and all-step numeric traces.\n'
            'trace_compact.npz intentionally excludes observation/qpos/qvel/force tensors; full traces remain on the training computer.\n'
            'No reward, failure endpoint, timestep, or score is changed to make this archive smaller.\n')
    return archive
