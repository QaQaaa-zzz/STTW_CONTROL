"""Reconstruct original native user costs for zero8 plus cached learned16."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from sttw_control.preference_best import digest,atomic_json
from sttw_control.preference_trace_rescore import rescore_trace_alpha
from sttw_control.controller import ControllerConfig
from sttw_control.direct_command_policy import q_ratio
from sttw_control.geometric_path import build_path
from sttw_control.path_command_reward import configured_cost_keys


def analyze(root):
    root=root.resolve();p=json.loads((root/'plan.json').read_text());out=root/'analysis'
    status=json.loads((root/'evaluation/status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=8:raise ValueError('complete eight numerical cases required')
    if digest(root/'plan.json')!=json.loads((root/'seal.json').read_text())['plan_sha256']:raise ValueError('changed plan')
    for f,sha in {**p['input_sha256'],**p['native_source_sha256'],**p['runner_sha256']}.items():
        if digest(f)!=sha:raise ValueError('sealed input/source changed '+f)
    cfg=p['config'];cc=ControllerConfig(**p['initial_source']['controller'])
    cached=json.loads((root/'cached_records.json').read_text())
    zero=json.loads((root/'evaluation/results.json').read_text())['cases']
    if len(cached)!=p['cached_cases'] or len(zero)!=8:raise ValueError('complete declared native panel needed')
    families=('H75','current','zero','equilibrium') if p.get('target_policy')=='equilibrium' else ('H75','current','zero')
    paths={};goals={}
    for c in p['cases']:
        if c['route'] in paths:continue
        file=root/'evaluation'/(c['route']+'_path.npz')
        with np.load(file) as z:paths[c['route']]={k:z[k] for k in z.files}
        path=paths[c['route']];goals[c['route']]=np.array([np.interp(float(path['goal']),path['s'],path['xy'][:,k]) for k in (0,1)])
    records=[];traces={};actual=0
    for row in cached+[dict(r,family=p.get('target_policy','zero')) for r in zero]:
        file=Path(row['raw_file'])
        if digest(file)!=row['raw_sha256']:raise ValueError('changed raw')
        with np.load(file) as z:raw={k:z[k] for k in z.files}
        family=row['family'];n=len(raw['active_tick'])
        if not 0<n<=2000 or not raw['active_tick'].all():raise ValueError('native contiguous active prefix required')
        np.testing.assert_allclose(raw['time'],np.arange(n)*.005,atol=2e-6,rtol=0)
        if np.any(raw['policy_fault']|raw['lower_fault']) or not np.isfinite(raw['effective_costs']).all():raise ValueError('engineering unknown, no optimistic scoring')
        if np.any(raw['applied_force_impulse']):raise ValueError('nominal only')
        if family==p.get('target_policy','zero'):
            actual+=n
        rule_audit={}
        if family=='equilibrium':
            from sttw_control.native_equilibrium_rule import target
            z,meta=jax.jit(jax.vmap(lambda nominal:target(nominal,row['alpha'],cfg,lambda v:q_ratio(v,cc),p['rule']['budget_rad'],p['rule']['grid_points'])))(j.asarray(raw['nominal'][::4]))
            expected=np.repeat(np.asarray(z),4,axis=0)[:n]
            np.testing.assert_allclose(raw['latent_z'],expected,atol=2e-5,rtol=2e-5)
            rule_audit=dict(rule_target_checked_policy_steps=len(z),speed_branch_grid_no_feasible_policy_steps=int((~np.asarray(meta['grid_has_feasible'])).sum()),latent_permission_limited_policy_steps=int(np.asarray(meta['latent_permission_limited']).sum()))
        if family=='zero':
            for f in ('latent_z','target_offset','filtered_offset','offsets'):
                if np.any(raw[f]):raise ValueError('nonzero neural correction '+f)
            np.testing.assert_allclose(raw['governed'],raw['nominal'],atol=1e-6,rtol=1e-6)
        case=next(c for c in p['cases'] if c['route']==row['route'])
        curve=build_path(case['builder'],np.zeros(3),cfg)
        k=np.interp(raw['path_progress'],np.asarray(curve.s),np.asarray(curve.curvature)).astype(np.float32)
        ff=np.arctan(np.float32(cc.wheelbase)*k/np.cos(np.float32(cc.caster)))
        chi=np.clip((abs(ff/np.asarray(q_ratio(j.float32(row['speed']),cc)))-.18)/.12,0,1)
        terminal=raw['physical_failure'].astype(bool)|raw['domain_exit'].astype(bool)
        data=dict(raw,requested_speed=raw['v_user'],chi=chi,contact_failure=terminal)
        scores={};maxerr=0.
        for user in (0.,1.):
            score=rescore_trace_alpha(data,alpha=user,config=cfg);scores[str(user)]=score['scored_cost']
            if user==row['alpha']:
                maxerr=float(np.max(abs(score['scored_reward']-raw['scored_tick_reward'])))
                np.testing.assert_allclose(score['scored_reward'],raw['scored_tick_reward'],rtol=3e-5,atol=3e-6)
            if 'user_objective_costs' in row:np.testing.assert_allclose(scores[str(user)],row['user_objective_costs'][str(user)],rtol=3e-5,atol=3e-5)
        fatal=np.flatnonzero(terminal);stop=int(fatal[0]) if len(fatal) else n
        distances=np.linalg.norm(raw['actual_xy'][:stop]-goals[row['route']],axis=1)
        within=np.flatnonzero(distances<=1.)
        physical=bool(raw['physical_failure'].any());domain=bool(raw['domain_exit'].any());complete=n==2000
        np.testing.assert_allclose(raw['v_user'],row['speed'],rtol=0,atol=1e-6)
        rec=dict(**rule_audit,family=family,route=row['route'],speed=row['speed'],alpha=row['alpha'],
            raw_file=str(file),raw_sha256=row['raw_sha256'],active_lower_ticks=n,observed_s=n*.005,
            native_physical_failure=physical,native_domain_failure=domain,native_complete_horizon=complete,
            native_physical_survived_horizon=False if physical else True if complete else None,
            goal1m_before_native_stop=bool(len(within)),first_arrival_s=float((within[0]+1)*.005) if len(within) else None,
            minimum_goal_distance_before_native_stop_m=float(distances.min()) if len(distances) else None,
            path_rmse_m=float(np.sqrt(np.mean(raw['path_cross_track']**2))),
            speed_rmse_m_s=float(np.sqrt(np.mean((raw['actual_forward_speed']-raw['v_user'])**2))),
            user_objective_costs=scores,own_tick_reward_max_abs_error=maxerr)
        records.append(rec);traces[(family,row['route'],row['speed'],row['alpha'])]=raw
    if actual!=status['active_lower_ticks']:raise ValueError('actual tick charge mismatch')
    matched={}
    for r in records:matched.setdefault((r['route'],r['speed'],r['alpha']),{})[r['family']]=r
    if len(matched)!=8 or any(set(g)!=set(families) for g in matched.values()):raise ValueError('complete native paired panel required')
    comparisons=[]
    pairings=[(f,'zero') for f in families if f!='zero']
    if 'equilibrium' in families:pairings += [('H75','equilibrium'),('current','equilibrium')]
    for family,control in pairings:
        for alpha in (0.,1.):
            detail=[]
            for k,g in matched.items():
                if k[-1]!=alpha:continue
                a,b=g[family],g[control];field=str(alpha)
                detail.append(dict(context=k,candidate_cost=a['user_objective_costs'][field],control_cost=b['user_objective_costs'][field],
                    cost_delta=a['user_objective_costs'][field]-b['user_objective_costs'][field],
                    both_complete=a['native_complete_horizon'] and b['native_complete_horizon'],
                    arrival_change=int(a['goal1m_before_native_stop'])-int(b['goal1m_before_native_stop'])))
            comparisons.append(dict(model=family,control=control,user_alpha=alpha,n=4,lower_own_user_cost=sum(d['cost_delta']<0 for d in detail),
                mean_cost_delta=float(np.mean([d['cost_delta'] for d in detail])),details=detail))
    matrices={f:[[float(np.mean([r['user_objective_costs'][str(u)] for r in records if r['family']==f and r['alpha']==a])) for a in (0.,1.)] for u in (0.,1.)] for f in families}
    out.mkdir(exist_ok=False)
    atomic_json(out/'review.json',dict(records=records,matrices_rows_user_cols_executed=matrices,own_user_vs_zero=[c for c in comparisons if c['control']=='zero'],own_user_comparisons=comparisons,
        new_actual_lower_ticks=actual,cached_cases=p['cached_cases'],new_cases=8,optimizer_updates=0,
        scope='Original native10s/physical-domain stops and original V3 user cost. No modern geometric feedback/support/replan; zero still contains nominal follower and frozen learned lower. Four correlated seen DEV contexts, not independent repeats or TEST. Domain truncation is not a fall or whole-window survival. No causal training attribution.'))
    render(out,records,traces,paths,goals,cfg)
    lines=['# Native preference versus matched zero-middle control','',
        '|Candidate|Control|Alpha|Lower own-user cost /4|Mean cost difference (candidate minus control)|','|---|---|---:|---:|---:|']
    lines += [f'|{x["model"]}|{x["control"]}|{x["user_alpha"]:g}|{x["lower_own_user_cost"]}|{x["mean_cost_delta"]:.5f}|' for x in comparisons]
    lines += ['', 'All declared native cases retained. Compare only SAME user alpha/config. Native failures keep original terminal penalties; observed-window RMSE of early failures is not credited as better full-task error.',
        'Both alpha traces of zero are retained as matched controls, not independent training repeats.1m arrival is observed before native physical/domain stop, not a hypothetical contact-only extension.',
        'Source runtime and raw identities checked; native zero targets/offsets and nominal/governed agreement checked; original V3 scores reconstructed for every case including both user preferences. Any static rule target is reconstructed at original policy boundaries; original user speed is unchanged.',
        'See review.json for every outcome, cost and matched difference; all_native_three_xy.png for actual original-route trajectories.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'receipt.json',dict(phase='completed',physics_steps=0,optimizer_updates=0,raw_sha_checked=len(records),
        native_zero_action_chain_checked=8,own_reward_reconstructed=len(records),analysis_source_sha256=digest(__file__),visual_review='pending'))
    print(json.dumps(dict(matrices=matrices,comparison=[{k:v for k,v in c.items() if k!='details'} for c in comparisons])))


def render(out,records,traces,paths,goals,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    colors=dict(H75='#0072B2',current='#D55E00',zero='#555555')
    if any(r['family']=='equilibrium' for r in records):colors['equilibrium']='#009E73'
    contexts=sorted({(r['route'],r['speed']) for r in records})
    fig,axes=plt.subplots(2,2,figsize=(12,10),layout='constrained')
    for ax,(route,v) in zip(axes.flat,contexts):
        path=paths[route];ax.plot(*path['xy'][path['s']<=path['goal']].T,c='black',ls=':',lw=1,label='Original route')
        ax.add_patch(Circle(goals[route],1.,fill=False,ec='black',lw=.7))
        for family,color in colors.items():
            for alpha,style in ((0.,'-'),(1.,'--')):
                r=next(x for x in records if (x['family'],x['route'],x['speed'],x['alpha'])==(family,route,v,alpha))
                raw=traces[(family,route,v,alpha)];xy=raw['actual_xy']
                ax.plot(*xy.T,c=color,ls=style,lw=1.1,label=f'{family} alpha{alpha:g}')
                marker='x' if r['native_physical_failure'] else 'D' if r['native_domain_failure'] else 's'
                ax.scatter(*xy[-1],c=color,marker=marker,s=23)
        ax.set(title=f'{route} / requested {v} m/s',xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(f'Native10s: all{len(records)} frozen model / ordinary reference controls; seed77001 / bank3\nActual terminal points; x=native physical failure, diamond=domain stop, square=10s')
    fig.savefig(out/'all_native_three_xy.png',dpi=150,bbox_inches='tight');plt.close(fig)
    diag=out/'diagnostics';diag.mkdir();keys=list(configured_cost_keys(cfg))
    for route,v in contexts:
        for alpha in (0.,1.):
            fig,axes=plt.subplots(5,1,figsize=(10,11),sharex=True,layout='constrained')
            components,ca=plt.subplots(4,3,figsize=(13,11),sharex=True,layout='constrained')
            for family,color in colors.items():
                raw=traces[(family,route,v,alpha)];t=raw['time']+.005
                values=(raw['actual_forward_speed'],raw['path_cross_track'],raw['phi'],raw['scored_tick_reward'],np.cumsum(raw['scored_tick_reward'],dtype=np.float64))
                for ax,y in zip(axes,values):ax.plot(t,y,c=color,label=family,lw=1)
                axes[0].plot(t,raw['wheel_speed_proxy'],c=color,ls='--',alpha=.35)
                for k,ax in zip(keys,ca.flat):ax.plot(t,raw['effective_costs'][:,keys.index(k)],c=color,lw=.9,label=family);ax.set_title(k,fontsize=8)
            for ax,name in zip(axes,('Forward speed (m/s)','Path error (m)','Roll (rad)','Tick reward','Cumulative reward')):ax.set_ylabel(name);ax.grid(alpha=.15)
            axes[0].axhline(v,c='black',ls=':',lw=.7);axes[0].legend();axes[-1].set_xlabel('Native task time (s)')
            fig.suptitle(f'{route} / {v}m/s / user alpha{alpha:g}; native10s, real endpoints, no smoothing')
            fig.savefig(diag/f'{route}_v{v:g}_a{alpha:g}.png',dpi=125);plt.close(fig)
            for ax in ca.flat[len(keys):]:ax.set_visible(False)
            ca.flat[0].legend(fontsize=7);components.suptitle(f'{route} / {v}m/s / alpha{alpha:g}: effective original V3 component costs (native rate units)')
            components.savefig(diag/f'{route}_v{v:g}_a{alpha:g}_components.png',dpi=110);plt.close(components)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    from engineer_reference_bias import notify
    try:analyze(a.root)
    except BaseException:
        import traceback
        (a.root/'analysis_error.txt').write_text(traceback.format_exc())
        notify(a.root,'STTW原生zero分析报错','原始轨迹保留，分析停止，不重跑物理。');raise
    else:notify(a.root,'STTW原生偏好强对照分析完成','新参考规则/zero8与缓存网络对照已重算原用户效用和到达/原生停止，全部XY及奖励图已生成，等待审阅。')
