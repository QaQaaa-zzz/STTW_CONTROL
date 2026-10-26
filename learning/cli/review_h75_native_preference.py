"""Reuse eight native-interface H75 zero-force endpoints; no simulation."""
import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from sttw_control.controller import ControllerConfig
from sttw_control.direct_command_policy import q_ratio
from sttw_control.geometric_path import build_path
from sttw_control.path_command_reward import configured_cost_keys
from sttw_control.preference_best import digest
from sttw_control.preference_trace_rescore import rescore_trace_alpha


def review(source,out):
    source=Path(source).resolve();out=Path(out).resolve()
    manifest=json.loads((source/'manifest.json').read_text());cfg=manifest['config'];decl=manifest['declaration']
    if cfg!=decl['config'] or cfg['path_reward']['reward_version']!='path_persistent_preference_v3':raise ValueError('native V3 config required')
    if json.loads((source/'status.json').read_text())['phase']!='completed':raise ValueError('completed historical panel required')
    registry=json.loads(Path('learning/configs/frozen_middle_registry.json').read_text());record=registry['records']['STTW_MIDDLE_FULLTAIL_PPO75_U77']
    model=decl['policies']['ppo75']
    for name in ['actor','checkpoint']:
        if model[name+'_sha256']!=record[name+'_sha256'] or digest(record[name])!=record[name+'_sha256']:raise ValueError('H75 model identity changed')
    native_cfg=json.loads((Path(registry['root'])/'STTW_MIDDLE_FULLTAIL_PPO75_U77_resolved_config.json').read_text())
    for name in ['residual_interface','network','path_reward','timing','follower']:
        if cfg[name]!=native_cfg[name]:raise ValueError('native interface/reward differs: '+name)
    rows=json.loads((source/'results.json').read_text())['cases']
    # Fixed first onset represents duplicate nominal schedules, not selected outcomes.
    chosen=[r for r in rows if r['policy']=='ppo75' and r['force_N']==0 and r['onset_s']==1 and r['alpha'] in [0.,1.]]
    if len(chosen)!=8 or len({(r['route'],r['speed'],r['alpha']) for r in chosen})!=8:raise ValueError('four bend/speed contexts x two endpoints required')
    cc=ControllerConfig(**manifest['source']['controller']);keys=list(configured_cost_keys(cfg))
    if keys!=manifest['reward_components']:raise ValueError('native component order differs')
    route_builders={r['name']:r['builder'] for r in decl['routes']}; records=[];traces={}
    for row in chosen:
        path=source/(row['name']+'.npz')
        with np.load(path) as z:raw={k:z[k] for k in z.files}
        active=raw['active_tick'].astype(bool)
        if len(active)!=2000 or not active.all() or not row['physical_survived_horizon'] or not row['complete_horizon']:
            raise ValueError('whole native ten-second cohorts required; do not omit failures')
        if np.any(raw['applied_force_impulse']):raise ValueError('nominal force must be zero')
        curve=build_path(route_builders[row['route']],np.zeros(3),cfg)
        curvature=np.interp(raw['path_progress'],np.asarray(curve.s),np.asarray(curve.curvature)).astype(np.float32)
        # Reconstructed from original declared geometry/progress/speed, not a logged sensor field.
        ff=np.arctan(np.float32(cc.wheelbase)*curvature/np.cos(np.float32(cc.caster)))
        q=np.asarray(q_ratio(jnp.float32(row['speed']),cc));chi=np.clip((np.abs(ff/q)-.18)/.12,0,1)
        data=dict(raw,chi=chi,requested_speed=raw['v_user'],contact_failure=raw['physical_failure'])
        scores={};max_error=0.
        for user in [0.,1.]:
            s=rescore_trace_alpha(data,alpha=user,config=cfg);scores[str(user)]=s['scored_cost']
            if user==row['alpha']:
                max_error=float(np.max(abs(s['effective_costs']-raw['effective_costs'])))
                np.testing.assert_allclose(s['effective_costs'],raw['effective_costs'],rtol=3e-5,atol=3e-5)
                np.testing.assert_allclose(s['scored_reward'],raw['scored_tick_reward'],rtol=3e-5,atol=3e-6)
                np.testing.assert_allclose(s['scored_cost'],-float(raw['scored_tick_reward'].sum(dtype=np.float64)),rtol=3e-5,atol=3e-5)
        records.append(dict(name=row['name'],route=row['route'],speed=row['speed'],alpha=row['alpha'],
            raw_file=str(path),raw_sha256=digest(path),user_objective_costs=scores,
            reconstructed_chi_max=float(chi.max()),reconstructed_chi_active_s=float((chi>0).sum()*.005),
            observed_primary_nonzero_ticks=int((raw['effective_costs'][:,keys.index('primary')]>0).sum()),
            own_component_reconstruction_max_abs_error=max_error,
            path_rmse_m=float(np.sqrt(np.mean(raw['path_cross_track']**2))),
            speed_rmse_m_s=float(np.sqrt(np.mean((raw['actual_forward_speed']-raw['v_user'])**2))),
            historical_goal=row['goal'],historical_final_hold=row['final_hold'],physical_survived_horizon=True))
        traces[row['name']]=raw
    matrix=[[float(np.mean([r['user_objective_costs'][str(u)] for r in records if r['alpha']==a])) for a in [0.,1.]] for u in [0.,1.]]
    out.mkdir(exist_ok=False)
    report=dict(records=records,cost_matrix_rows_user_cols_executed=matrix,
        source_manifest_sha256=digest(source/'manifest.json'),frozen_actor_sha256=record['actor_sha256'],
        primary_cost_observed_nonzero_all_eight=all(r['observed_primary_nonzero_ticks']>0 for r in records),
        actual_historical_trajectories=8,new_physics_steps=0,optimizer_updates=0,
        nominal_deduplication='ppo75 F0, onset1 only, alpha0/1, all four declared bends/speeds; onset2 and interior alphas retained at original source and not counted as repeats.',
        chi_semantics='Reconstructed from original configuration/curvature interpolated at logged progress/requested speed; validated against native own effective cost and scored reward. Not original logged chi telemetry.',
        scope='Historical native-interface DEV diagnostic of H75 only, same-source correlated contexts. No new control benefit, current-policy native comparison, independent training repetition or original goal redefinition.')
    (out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    render(records,traces,out)
    lines=['# H75原生高需求用途复查','',report['scope'],'',report['chi_semantics'],'',
        '|路线|速度|α|路径RMSE/m|速度RMSE/(m/s)|重建χ非零秒|原primary非零步|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:lines.append(f"|{r['route']}|{r['speed']}|{r['alpha']:g}|{r['path_rmse_m']:.4f}|{r['speed_rmse_m_s']:.4f}|{r['reconstructed_chi_active_s']:.3f}|{r['observed_primary_nonzero_ticks']}|")
    lines+=['','原用户目标均值成本矩阵（行user0/1、列实际执行0/1）：'+str(np.round(matrix,4).tolist()),'',
        '八条均整窗存活；旧goal/hold标记保留在JSON，不改作近期50cm点圆到达。两个任务的原始成本不直接跨场景排名。既有F0重复按固定onset1去重，不按结果挑选，不把八条当独立训练重复。不自动把H75切回采用模型、不声称当前控制器在该原生面板已测。']
    (out/'README.md').write_text('\n'.join(lines)+'\n');print(out)


def render(records,traces,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,9),layout='constrained')
    contexts=sorted({(r['route'],r['speed']) for r in records})
    for ax,(route,speed) in zip(axes.flat,contexts):
        for alpha,color,style in [(0.,'#0072B2','-'),(1.,'#D55E00','--')]:
            row=next(r for r in records if r['route']==route and r['speed']==speed and r['alpha']==alpha);raw=traces[row['name']]
            if alpha==0:ax.plot(raw['reference_xy'][:,0],raw['reference_xy'][:,1],':',color='.4',label='Recorded original path samples')
            ax.plot(raw['actual_xy'][:,0],raw['actual_xy'][:,1],color=color,ls=style,label=f'H75 alpha{alpha:g}')
            ax.scatter(*raw['actual_xy'][-1],color=color,s=25)
        ax.set(title=f'{route}, requested {speed}m/s',xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9);fig.suptitle('H75 native-interface historical DEV / all8 zero-force traces\nReference samples and physical endpoints preserved; no new simulation or modern-goal relabeling')
    fig.savefig(out/'all_native_xy_readable.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();review(a.source,a.output)
