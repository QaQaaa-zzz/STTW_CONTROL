"""Offline audit of the declared frozen single-preference pulse screen."""
from pathlib import Path
import csv,json,math
import numpy as np


def normalized_risk(features,force,*,effort_budget,gamma,duration):
    if features is None:return None
    lam=features['lambda_u'];b=features['b_u'];z=features['z_u']
    horizon=min(1.,features['remaining_s'])
    w=b*b*(-math.expm1(-2*lam*horizon))/(2*lam)
    k=gamma*(-math.expm1(-lam*duration))/lam
    return abs(z+k*force)/math.sqrt(effort_budget*w)


def paired_nominal_response(rows,traces,window=2.):
    """Signed pulse response relative to own matched nominal, with no new label."""
    context=lambda r:tuple(r[k] for k in ('policy','route','speed','alpha','onset_s'))
    nominal={context(r):r for r in rows if r['force_N']==0}
    result=[]
    for row in rows:
        if row['force_N']==0:continue
        d=traces[row['name']];n=traces[nominal[context(row)]['name']]
        count=min(len(d['time']),len(n['time']))
        np.testing.assert_array_equal(d['time'][:count],n['time'][:count])
        t=d['time'][:count];after=t>=row['onset_s']-1e-5
        mask=after&(t<row['onset_s']+window-1e-5)
        item=dict(name=row['name'],alpha=row['alpha'],force_N=row['force_N'],
            window_complete=bool(mask.sum()==round(window/.005)))
        for field in ('phi','path_cross_track','actual_forward_speed'):
            delta=d[field][:count]-n[field][:count]
            item[field+'_delta_rms_2s']=float(np.sqrt(np.mean(delta[mask]**2))) if mask.any() else None
            item[field+'_peak_delta_post']=float(np.max(np.abs(delta[after]))) if after.any() else None
        result.append(item)
    return result


def screen_summary(audit,policies,alphas):
    summary=[]
    for policy in policies:
      for alpha in alphas:
        group=[r for r in audit if r['policy']==policy and r['alpha']==alpha]
        nominal=[r for r in group if r['force_N']==0]
        nonzero=[r for r in group if r['force_N']!=0]
        inside=[r for r in nonzero if r['predicted_within'] is True]
        summary.append(dict(policy=policy,alpha=alpha,n=len(group),task=sum(r['task_recovered'] for r in group),
            performance=sum(r['performance_success'] for r in group),physical=sum(r['physical_failure'] for r in group),
            domain=sum(r['domain_failure'] for r in group),nominal_task=sum(r['task_recovered'] for r in nominal),
            nominal_n=len(nominal),nonzero_n=len(nonzero),predicted_within=len(inside),
            false_optimistic=sum(not r['task_recovered'] for r in inside),
            conservative=sum(r['task_recovered'] and r['predicted_within'] is False for r in nonzero)))
    return summary


def report(run,output):
    from .preference_best import atomic_json
    from .recovery_margin_scan import outcome_labels
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);out=Path(output)
    status=json.loads((run/'status.json').read_text())
    if status['phase']!='completed':raise ValueError('screen incomplete; no result claim')
    manifest=json.loads((run/'manifest.json').read_text());decl=manifest['declaration'];cfg=manifest['config']
    rows=json.loads((run/'results.json').read_text())['cases']
    if len(rows)!=status['declared_cases'] or len({r['name'] for r in rows})!=len(rows):raise ValueError('case accounting mismatch')
    out.mkdir(parents=True,exist_ok=False);(out/'curves').mkdir()
    traces={};audit=[];spec=dict(effort_budget=.030839085298575167,gamma=.815560907125473,duration=decl['duration_s'])
    prefix=cfg['path']['straight_prefix_m'];turn_end=prefix+abs(cfg['path']['turn_angle_rad'])*cfg['path']['turn_radius_m']+cfg['path']['curvature_transition_length_m']
    b=cfg['stage_B_baseline'];horizon=cfg['training_future_phase_C']['episode_s']
    for row in rows:
        with np.load(run/(row['name']+'.npz')) as file:d={k:file[k] for k in file.files}
        traces[row['name']]=d
        path=np.load(run/(row['route']+'_path.npz'))
        label=outcome_labels(d,speed=row['speed'],goal_s=float(path['goal']),working_roll=b['working_roll_rad'],horizon=horizon,hold_s=b['hold_s'])
        for k in label:
            if label[k]!=row[k]:raise ValueError(f'logged label differs: {row["name"]}/{k}')
        reward=float(d['scored_tick_reward'].sum())
        if not np.isclose(reward,row['logged_return'],atol=1e-5):raise ValueError('logged reward sum differs')
        turn=np.ones(len(d['time']),bool) if row['route'].startswith('straight') else (d['path_progress']>=prefix)&(d['path_progress']<=turn_end)
        rms=lambda x,m:float(np.sqrt(np.mean(np.asarray(x)[m]**2))) if m.any() else None
        risk=normalized_risk(row['prefix_features'],row['force_N'],**spec)
        a=dict(**row,predicted_risk=risk,predicted_within=None if risk is None else risk<=1,
            turn_speed_rmse=rms(d['actual_forward_speed']-row['speed'],turn),
            turn_path_rmse=rms(d['path_cross_track'],turn),
            wheel_proxy_rmse=rms(d['wheel_speed_proxy']-d['actual_forward_speed'],np.ones(len(d['time']),bool)))
        audit.append(a)
    atomic_json(out/'paired_nominal_response.json',paired_nominal_response(audit,traces))
    summary=screen_summary(audit,decl['policies'],decl['alphas'])
    contexts=sorted({(r['policy'],r['route'],r['speed'],r['onset_s']) for r in rows})
    links=[]
    for policy,route,speed,onset in contexts:
        selected=[r for r in audit if (r['policy'],r['route'],r['speed'],r['onset_s'])==(policy,route,speed,onset)]
        name=(policy+'_' if len(decl['policies'])>1 else '')+f'{route}_v{speed:g}_t{onset:g}'
        fig,axes=plt.subplots(len(decl['forces_N']),4,figsize=(18,18),layout='constrained',squeeze=False)
        for i,force in enumerate(decl['forces_N']):
            group=sorted([r for r in selected if r['force_N']==force],key=lambda r:r['alpha'])
            for r in group:
                d=traces[r['name']];label=f"α={r['alpha']:g}, task={int(r['task_recovered'])}"
                axes[i,0].plot(*d['actual_xy'].T,label=label)
                axes[i,1].plot(d['time'],d['actual_forward_speed'],label=label)
                axes[i,2].plot(d['time'],d['path_cross_track'],label=label)
                axes[i,3].plot(d['time'],d['phi'],label=label)
                for ax in axes[i,1:]:ax.axvline(d['time'][-1],alpha=.12)
            for col,title in enumerate(('XY [m]','true speed [m/s]','signed path error [m]','roll [rad]')):
                ax=axes[i,col];ax.set_title(f'{force:+g}N | {title}');ax.grid(alpha=.2)
                if col:ax.axvspan(onset,onset+decl['duration_s'],alpha=.15,color='gray');ax.set_xlabel('task time [s]')
            path=np.load(run/(route+'_path.npz'));mask=path['s']<speed*horizon+1
            axes[i,0].plot(*path['xy'][mask].T,'k--',label='fixed reference');axes[i,0].set_aspect('equal',adjustable='datalim')
            axes[i,1].axhline(speed,color='k',ls='--');axes[i,3].axhline(b['working_roll_rad'],color='k',ls=':');axes[i,3].axhline(-b['working_roll_rad'],color='k',ls=':')
        axes[0,0].legend(fontsize=7);fig.suptitle(f'{policy} frozen evaluation | {name}')
        for ext in ('png','pdf'):fig.savefig(out/'curves'/f'{name}.{ext}',dpi=110)
        plt.close(fig);links.append(f'- [{name}](curves/{name}.png)')
        for alpha in decl['alphas']:
            fig,axes=plt.subplots(len(decl['forces_N']),3,figsize=(17,16),layout='constrained',squeeze=False)
            for i,force in enumerate(decl['forces_N']):
                r=next(r for r in selected if r['alpha']==alpha and r['force_N']==force);d=traces[r['name']];t=d['time']
                axes[i,0].plot(t,d['scored_tick_reward'],label='logged reward/5ms');axes[i,0].set_title(f'{force:+g}N | reward')
                for k,key in enumerate(manifest['reward_components']):axes[i,1].plot(t,d['effective_costs'][:,k],label=key)
                axes[i,1].set_title('effective cost rates before terminal replacement')
                axes[i,2].plot(t,d['actual_forward_speed'],label='true body speed');axes[i,2].plot(t,d['wheel_speed_proxy'],label='rear wheel × 0.1m');axes[i,2].axhline(speed,color='k',ls='--',label='user speed [m/s]')
                for ax in axes[i]:ax.axvspan(onset,onset+decl['duration_s'],alpha=.15,color='gray');ax.axvline(t[-1],color='r',alpha=.3);ax.set_xlabel('task time [s]');ax.grid(alpha=.2)
            for ax in axes[0]:ax.legend(fontsize=6)
            fig.suptitle(f'{policy} | {name} | α={alpha:g} | original V3 reward, true executed endpoint')
            for ext in ('png','pdf'):fig.savefig(out/'curves'/f'{name}_a{alpha:g}_reward.{ext}',dpi=100)
            plt.close(fig)
    atomic_json(out/'summary.json',dict(summary=summary,cases=audit,formula_parameters=spec))
    scalar=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in audit]
    with (out/'cases.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(scalar[0]));w.writeheader();w.writerows(scalar)
    lines=['# Frozen single-alpha policy real-pulse evaluation','',
        f'{len(rows)}-case development diagnostic; fixed local300, original V3 and10s/1s hold. This is a frozen-policy evaluation, not training or final TEST.',
        'Task recovery = no physical/domain failure + original goal + final1s path≤0.1m, heading≤0.05rad, speed≤0.1m/s. Performance additionally requires working roll≤0.302rad.',
        f"Zero-force controls may repeat across pulse onsets; unique nominal contexts are{len({(r['route'],r['speed'],r['alpha']) for r in rows if r['force_N']==0})} per policy. Cases are paired deterministic branches, not independent training seeds.",
        'Formula parameters are frozen from earlier dual-network calibration; no fitting to this screen. Predicted outside does not imply unavoidable failure.','',
        '| policy | α | task | performance | physical | domain | nominal | predicted-within nonzero | optimistic task errors | conservative task passes |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in summary:lines.append(f"| {r['policy']} | {r['alpha']:g} | {r['task']}/{r['n']} | {r['performance']}/{r['n']} | {r['physical']} | {r['domain']} | {r['nominal_task']}/{r['nominal_n']} | {r['predicted_within']} | {r['false_optimistic']} | {r['conservative']} |")
    lines+=['','Per-context response overlays (all declared alphas; pulse shaded; actual endpoints marked):','']+links
    lines+=['','Per-alpha reward/component/wheel-proxy PNG+PDF are alongside each overview. Raw logged arrays remain in the immutable run. No matched pure-ECBC real-pulse control was run in this diagnostic; each pulse response uses the same policy own nominal context.','',f"Source: {run}; code {manifest['code_revision']}; actual lower ticks {status['active_lower_ticks']}/{status['maximum_lower_ticks']}; evaluation PPO=0."]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary
