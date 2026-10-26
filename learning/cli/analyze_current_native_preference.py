"""Compare eight new native-interface current trajectories with eight cached H75."""
import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from sttw_control.controller import ControllerConfig
from sttw_control.direct_command_policy import q_ratio
from sttw_control.geometric_path import build_path
from sttw_control.preference_best import digest
from sttw_control.preference_trace_rescore import rescore_trace_alpha


def analyze(root):
    root=Path(root).resolve();source=root/'evaluation'
    status=json.loads((source/'status.json').read_text())
    if status['phase']!='completed' or status['completed_cases']!=8:raise ValueError('complete declared eight required; no failed-case replacement')
    manifest=json.loads((source/'manifest.json').read_text());decl=json.loads((root/'declaration.json').read_text());cfg=decl['config']
    if manifest['config']!=cfg or manifest['source']!=decl['source']:raise ValueError('native config/source mismatch')
    cache=json.loads((root/'cache_receipt.json').read_text());old_source=Path(cache['source_run']);old_manifest=json.loads((old_source/'manifest.json').read_text())
    if digest(old_source/'manifest.json')!=cache['source_manifest_sha256'] or old_manifest['config']!=cfg or old_manifest['source']!=decl['source']:raise ValueError('native baseline context/config/source differs')
    current=json.loads((source/'results.json').read_text())['cases']
    historical=[r for r in json.loads((old_source/'results.json').read_text())['cases'] if r['policy']=='ppo75' and r['force_N']==0 and r['onset_s']==1 and r['alpha'] in [0.,1.]]
    if len(current)!=8 or len(historical)!=8:raise ValueError('all16 actual trajectories required')
    cc=ControllerConfig(**decl['source']['controller']);routes={r['name']:r['builder'] for r in decl['routes']};records=[];traces={};goals={}
    for route in routes:
        with np.load(source/(route+'_path.npz')) as p:goals[route]=np.array([np.interp(float(p['goal']),p['s'],p['xy'][:,i]) for i in [0,1]])
    cached_sha={r['name']:r['raw_sha256'] for r in cache['original_records']}
    for family,rows,base in [('current',current,source),('H75',historical,old_source)]:
        for row in rows:
            file=base/(row['name']+'.npz')
            if family=='H75' and digest(file)!=cached_sha[row['name']]:raise ValueError('historical raw changed')
            with np.load(file) as z:raw={k:z[k] for k in z.files}
            if not np.isfinite(raw['effective_costs']).all() or np.any(raw['policy_fault']|raw['lower_fault']):raise ValueError('engineering unknown retained, no optimistic score')
            active=raw['active_tick'].astype(bool);n=len(active)
            if not active.all() or not 0<n<=2000:raise ValueError('contiguous actually executed native prefix required')
            np.testing.assert_allclose(raw['time'],np.arange(n)*.005,atol=2e-6,rtol=0)
            curve=build_path(routes[row['route']],np.zeros(3),cfg)
            k=np.interp(raw['path_progress'],np.asarray(curve.s),np.asarray(curve.curvature)).astype(np.float32)
            ff=np.arctan(np.float32(cc.wheelbase)*k/np.cos(np.float32(cc.caster)))
            chi=np.clip((abs(ff/np.asarray(q_ratio(jnp.float32(row['speed']),cc)))-.18)/.12,0,1)
            # The ORIGINAL native contract penalizes physical OR domain termination.
            # This mask is for cost reconstruction, not a claim domain exit is a fall.
            terminal=raw['physical_failure'].astype(bool)|raw['domain_exit'].astype(bool)
            data=dict(raw,requested_speed=raw['v_user'],chi=chi,contact_failure=terminal)
            scores={};error=None
            for user in [0.,1.]:
                score=rescore_trace_alpha(data,alpha=user,config=cfg);scores[str(user)]=score['scored_cost']
                if user==row['alpha']:
                    error=float(np.max(abs(score['scored_reward']-raw['scored_tick_reward'])))
                    np.testing.assert_allclose(score['scored_reward'],raw['scored_tick_reward'],rtol=3e-5,atol=3e-6)
            before=np.arange(n)<(np.flatnonzero(terminal)[0] if terminal.any() else n)
            distance=np.linalg.norm(raw['actual_xy']-goals[row['route']],axis=1)
            rec=dict(family=family,route=row['route'],speed=row['speed'],alpha=row['alpha'],name=row['name'],raw_file=str(file),raw_sha256=digest(file),
                active_lower_ticks=n,native_physical_failure=row['physical_failure'],native_domain_failure=row['domain_failure'],
                native_complete_horizon=row['complete_horizon'],native_physical_survived_horizon=row['physical_survived_horizon'],
                historical_goal_section=row['goal'],historical_hold=row['final_hold'],
                spatial_goal0p5_before_native_stop=bool(np.any(before&(distance<=.5))),
                minimum_goal_distance_before_native_stop_m=float(distance[before].min()) if before.any() else None,
                path_rmse_m=float(np.sqrt(np.mean(raw['path_cross_track']**2))),speed_rmse_m_s=float(np.sqrt(np.mean((raw['actual_forward_speed']-raw['v_user'])**2))),
                reconstructed_chi_active_s=float((chi>0).sum()*.005),user_objective_costs=scores,own_tick_reward_max_abs_error=error)
            records.append(rec);traces[(family,row['route'],row['speed'],row['alpha'])]=raw
    matrices={family:[[float(np.mean([r['user_objective_costs'][str(u)] for r in records if r['family']==family and r['alpha']==a])) for a in [0.,1.]] for u in [0.,1.]] for family in ['H75','current']}
    out=root/'analysis';out.mkdir(exist_ok=False)
    report=dict(records=records,matrices_rows_user_cols_executed=matrices,new_cases=8,cached_cases=8,
        new_actual_lower_ticks=status['active_lower_ticks'],cached_actual_lower_ticks=16000,optimizer_updates=0,
        scope='Shared native high-demand four-context DEV; frozen weight comparison, not independent repeat, causal training ablation or new safety method. Reconstructed chi, not original telemetry. Native physical/domain stop semantics retained; domain exit is NOT equated with fall. Spatial goal is analysis only and does not change control/reward. Incomplete native windows retain their failure cost and observed-window RMSE, no outcome filtering.')
    (out/'review.json').write_text(json.dumps(report,indent=2)+'\n');render(out,records,traces)
    lines=['# 当前模型的原生偏好保持核对','',report['scope'],'',
        '|模型|路线|速度|α|路径RMSE/m|速度RMSE/(m/s)|原生完整窗|0.5m到达|',
        '|---|---|---:|---:|---:|---:|---|---|']
    for r in records:lines.append(f"|{r['family']}|{r['route']}|{r['speed']}|{r['alpha']:g}|{r['path_rmse_m']:.4f}|{r['speed_rmse_m_s']:.4f}|{r['native_complete_horizon']}|{r['spatial_goal0p5_before_native_stop']}|")
    lines+=['','同一用户目标成本矩阵（行user0/1，列执行0/1）：']
    for family,m in matrices.items():lines.append(f'- {family}: {np.round(m,4).tolist()}')
    lines+=['','原截面/保持标记与新增原目标50cm点圆并非嵌套，均保留；点圆到达是原native停止前观测，不推断未观测尾段。新旧失败不筛选，不设置零成本差/全部成功门槛，不据此自动采用模型或归因训练算法。']
    (out/'README.md').write_text('\n'.join(lines)+'\n');(out/'status.json').write_text(json.dumps(dict(phase='completed',actual_trajectories=16,new_physics_steps=0,optimizer_updates=0,visual_review_complete=False),indent=2)+'\n')


def render(out,records,traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,10),layout='constrained')
    for ax,(route,speed) in zip(axes.flat,sorted({(r['route'],r['speed']) for r in records})):
        old=traces[('H75',route,speed,0.)];ax.plot(old['reference_xy'][:,0],old['reference_xy'][:,1],':',color='.4',label='Recorded original reference samples')
        for family,color in [('H75','#D55E00'),('current','#0072B2')]:
            for alpha,style in [(0.,'-'),(1.,'--')]:
                raw=traces[(family,route,speed,alpha)];r=next(x for x in records if (x['family'],x['route'],x['speed'],x['alpha'])==(family,route,speed,alpha))
                ax.plot(raw['actual_xy'][:,0],raw['actual_xy'][:,1],color=color,ls=style,label=f'{family} alpha{alpha:g}')
                ax.scatter(*raw['actual_xy'][-1],color=color,s=22,marker='x' if r['native_physical_failure'] else ('o' if alpha==0 else 's'))
        ax.set(title=f'{route}, requested {speed}m/s',xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
    fig.suptitle('Native high-demand preference retention:8 new current +8 cached H75\nShared original task/interface; actual endpoints, no new feedback or training')
    fig.savefig(out/'all_native_pair_xy.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);analyze(p.parse_args().run)
