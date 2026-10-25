"""Read complete fixed64 DEV traces; no new physics or optimizer updates."""
import json,os,time,traceback
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest
from .live_endpoint_analysis import ROLES,paired_endpoint_report,tracking_recovery_time,block_reward_series


def analyze(root):
    from prepare_short_recovery_campaign import register
    from .anticipatory_analysis import cost_accounting,save_csv,load_analysis_cases
    from .anticipatory_run import physical_outcome
    from .rejoin_comparison import geometric_task_trace
    from .survival_endpoint_task_metrics import task_metrics
    from .preference_trace_rescore import rescore_trace_alpha
    from .path_command_reward import configured_cost_keys
    root=Path(root).resolve();plan=json.loads((root/'plan.json').read_text())
    for c in plan['routes'].values():
        local=Path(c['path']);assert digest(local/'plan.json')==c['plan_sha256']
        s=json.loads((local/'reentry/status.json').read_text())
        if s['phase']!='completed' or s['completed_cases']!=100:raise ValueError('both complete fixed100 route stages required')
    out=root/'analysis';out.mkdir(exist_ok=False)
    state=dict(phase='running',pid=os.getpid(),completed_cases=0,physics_steps=0,optimizer_updates=0)
    def update(**kw):state.update(kw,checked_unix=time.time());atomic_json(out/'status.json',state)
    update();register(out/'ACTIVE_RUN.json','STTW_CONTROL 同合同普通PPO固定终点五alpha只读分析',out/'status.json')
    try:
        rows=[];traces={};paths={};configs={};queries=[];audits=[]
        for context,c in plan['routes'].items():
            local=Path(c['path']);p=json.loads((local/'plan.json').read_text());cfg=json.loads((local/'common_environment_config.json').read_text())
            for f,sha in p['sealed_files'].items():assert digest(f)==sha,f
            assert digest(p['path_file'])==p['path_sha256']
            with np.load(p['path_file'],allow_pickle=False) as z:path={k:z[k] for k in z.files}
            paths[context]=path;configs[context]=cfg
            results,index=load_analysis_cases(local,'reentry',p);byid={x['case_id']:x for x in index}
            declared={x['case_id']:x for x in json.loads((local/'inventory.json').read_text())['cases']}
            assert set(declared)=={r['case_id'] for r in results}
            for r in results:
                for k,v in declared[r['case_id']].items():assert r[k]==v,(r['case_id'],k)
                item=byid[r['case_id']];file=local/item['file'];assert digest(file)==item['sha256']
                with np.load(file,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
                active=raw['active_tick'].astype(bool)
                outcome=physical_outcome(active,raw['contact_failure'],raw['nonfinite']|raw['lower_fault']|raw['policy_fault'],2000)
                assert outcome==r['physical_outcome']
                geometry=geometric_task_trace(raw,path,cfg,initial_xy=item['initial_xy'])
                metrics=task_metrics(dict(raw,**geometry,domain_exit=np.zeros_like(active),task_reward_valid=active),
                    speed=r['requested_speed_m_s'],goal_s=float(path['goal']),config=cfg)
                category=('unknown' if outcome=='unknown' or metrics['task_recovered'] is None else
                    'contact' if outcome=='contact' else 'safe_task' if metrics['task_recovered'] else 'safe_task_failed')
                costs=cost_accounting(raw,scale=cfg['path_reward']['scale'],dt=.005)
                band_return=tracking_recovery_time(dict(raw,**geometry),speed=r['requested_speed_m_s'],onset_tick=r['pulse_onset_task_tick'],config=cfg)
                case_id=context+'/'+r['case_id'];xy=raw['actual_xy'][active]
                ref=raw['reference'][active]
                row=dict(r,case_id=case_id,context=context,category=category,**metrics,**costs,
                    band_return_status=band_return['status'],band_return_delay_s=band_return['delay_from_onset_s'],
                    band_return_confirmed_s=band_return['confirmed_at_s'],raw_file=str(file),raw_sha256=item['sha256'],
                    max_original_path_error_m=float(abs(geometry['path_cross_track'][active]).max()),
                    traveled_distance_m=float(np.linalg.norm(np.diff(np.vstack([item['initial_xy'],xy]),axis=0),axis=1).sum()),
                    reference_speed_variation_m_s=float(abs(np.diff(ref[:,0])).sum()),reference_steer_variation_rad=float(abs(np.diff(ref[:,1])).sum()))
                rows.append(row);traces[case_id]=dict(raw,**geometry)
                audits.append(dict(case_id=case_id,raw_sha256=item['sha256'],cost_components_verified=True,physical_outcome_verified=True))
                # Score original logged geometry; reconstructed geometry is only
                # for original-task diagnostics, never to rewrite rewards.
                for alpha in [0.,.25,.5,.75,1.]:
                    q=rescore_trace_alpha(raw,alpha=alpha,config=cfg)
                    if alpha==r['alpha']:assert np.allclose(q['scored_reward'],raw['scored_tick_reward'],rtol=2e-5,atol=2e-6)
                    queries.append(dict(case_id=case_id,executed_alpha=r['alpha'],objective_alpha=alpha,scored_cost=q['scored_cost']))
                if len(rows)%20==0:update(completed_cases=len(rows))
        assert len(rows)==plan['declared_cases']==200
        summary={name:paired_endpoint_report(selected) for name,selected in dict(all=rows,
            nominal=[r for r in rows if not r['force_N']],disturbed=[r for r in rows if r['force_N']]).items()}
        byalpha={str(a):paired_endpoint_report([r for r in rows if r['alpha']==a]) for a in [0.,.25,.5,.75,1.]}
        zeros={(r['context'],r['role'],r['alpha']):r for r in rows if not r['force_N']};own_zero=[]
        assert len(zeros)==40
        for r in rows:
            if not r['force_N']:continue
            z=zeros[r['context'],r['role'],r['alpha']];unknown='unknown' in (r['category'],z['category'])
            own_zero.append(dict(case_id=r['case_id'],zero_case_id=z['case_id'],
                scored_cost_change=r['scored_cost']-z['scored_cost'],
                physical_change=None if unknown else int(r['category'] in ['safe_task','safe_task_failed'])-int(z['category'] in ['safe_task','safe_task_failed']),
                task_change=None if unknown else int(r['category']=='safe_task')-int(z['category']=='safe_task')))
        atomic_json(out/'results.json',rows);save_csv(out/'results.csv',rows)
        atomic_json(out/'summary.json',summary);atomic_json(out/'by_alpha.json',byalpha)
        save_csv(out/'paired_changes.csv',summary['all']['pairs']);save_csv(out/'own_zero_changes.csv',own_zero)
        save_csv(out/'fixed_trace_objective_queries.csv',queries);atomic_json(out/'raw_audit.json',audits)
        figures=_render(out,rows,traces,paths,configs)
        atomic_json(out/'receipt.json',dict(raw_cases=200,fixed_trace_queries=1000,figures=figures,
            physics_steps=0,optimizer_updates=0,visual_review_complete=False,
            transformations='Full active real XY, original path to goal in view, 20ms reward sums, originalalpha reward reconstruction; no smoothing or interpolation.',
            scope='Two old routes, correlated event branches, one paired training repeat; no p-values or independent-geometry claim.'))
        lines=['# 固定64终点：同合同普通PPO有无速度支持','',
            '200条完整确定性DEV轨迹；图与数值尚需人工研究裁决。未采用新模型。','',
            '|角色|无触地且任务完成|无触地但任务失败|触地|未知|','|---|---:|---:|---:|---:|']
        for role,counts in summary['all']['role_counts'].items():lines.append('|'+role+'|'+'|'.join(str(counts[k]) for k in ['safe_task','safe_task_failed','contact','unknown'])+'|')
        lines+=['','全部配对新增/丢失见paired_changes.csv；自身零力变化见own_zero_changes.csv。',
            '回到误差带并保持1秒是描述性指标，不能替代全程防摔或原目标完成；无扰也可能发生自然离带。',
            '名义任务成本—受扰存活图按原alpha分面；每点仅两条旧路线的名义成本与8个相关受扰分支，不作独立置信区间。',
            '实际轨迹全部保留，参考显示到原任务目标；contact叉号/时域末端圆点，未延伸失败轨迹。',
            '方法辅助：Scientific Agent Skills scientific-critical-thinking/scientific-visualization；Kassis等（2026），https://doi.org/10.48550/arXiv.2609.00065 。']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n');update(phase='completed',stage='numeric_and_figures_complete_review_pending',completed_cases=200)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise


def _render(out,rows,traces,paths,configs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .path_command_reward import configured_cost_keys
    colors=dict(zip(ROLES,['#0072B2','#E69F00','#009E73','#D55E00']))
    styles=dict(zip(ROLES,[':', '-.', '--', '-']))
    groups={};figures=[]
    for r in rows:groups.setdefault((r['context'],r['alpha'],r['event_id']),[]).append(r)
    for number,((context,alpha,event),group) in enumerate(groups.items()):
        assert len(group)==4
        path=paths[context];cfg=configs[context];ref=path['xy'][path['s']<=float(path['goal'])]
        cloud=np.vstack([ref]+[traces[r['case_id']]['actual_xy'][traces[r['case_id']]['active_tick']] for r in group])
        lo=cloud.min(0)-.5;hi=cloud.max(0)+.5
        fig,grid=plt.subplots(3,3,figsize=(14,11),layout='constrained');axes=grid.flat
        axes[0].plot(ref[:,0],ref[:,1],color='.55',ls=':',label='Original path to goal');axes[0].scatter(*ref[-1],marker='*',color='k')
        for r in group:
            role=r['role'];d=traces[r['case_id']];a=d['active_tick'];ids=np.flatnonzero(a);t=d['physical_tick'][a]*.005;xy=d['actual_xy'][a]
            style=dict(color=colors[role],ls=styles[role],label=role)
            axes[0].plot(xy[:,0],xy[:,1],**style);axes[0].scatter(*xy[-1],color=colors[role],marker='x' if r['category']=='contact' else 'o',s=28)
            for panel,key in [(1,'path_cross_track'),(3,'path_heading_error'),(4,'phi'),(5,'actual_delta')]:axes[panel].plot(t,d[key][a],**style)
            axes[2].plot(t,d['actual_forward_speed'][a]-r['requested_speed_m_s'],**style)
            series=block_reward_series(d,config=cfg);u=series['time_s'];rewards=series['scored_reward'];components=series['signed_running_components']
            axes[6].plot(u,rewards,**style);axes[7].plot(u,np.cumsum(rewards),**style);axes[8].plot(t,d['actor_action_valid'][a],**style)
            np.savez_compressed(out/f'series_{number:03d}_{role}.npz',time_s=u,xy_time_s=t,xy=xy,
                scored_reward=rewards,cumulative_reward=np.cumsum(rewards),signed_components=components,
                cumulative_components=np.cumsum(components,axis=0),terminal_replacement_reward=series['terminal_replacement_reward'],
                cumulative_terminal_replacement_reward=np.cumsum(series['terminal_replacement_reward']),
                reference=d['reference'][a],nominal_reference=d['nominal_reference'][a],actual_speed=d['actual_forward_speed'][a],
                actual_delta=d['actual_delta'][a],path_error=d['path_cross_track'][a],heading_error=d['path_heading_error'][a],
                cost_keys=np.asarray(configured_cost_keys(cfg)),component_semantics=np.asarray('signed running reward; terminal block replacement is a separate channel'))
        axes[0].set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='World X (m)',ylabel='World Y (m)');axes[0].set_aspect('equal',adjustable='box');axes[0].legend(fontsize=6)
        titles=['Actual XY / original goal','Signed path error (m)','Speed error (m/s)','Signed heading error (rad)',
            'Roll (rad)','Actual steering (rad)','Reward per20ms block','Cumulative originalalpha reward','Legal Actor decisions']
        for i,(ax,title) in enumerate(zip(axes,titles)):
            ax.set_title(title);ax.grid(alpha=.2)
            if i:
                ax.set(xlim=(0,10),xlabel='Task time (s)')
                if group[0]['force_N']:ax.axvspan(group[0]['pulse_onset_task_tick']*.005,group[0]['pulse_onset_task_tick']*.005+.1,color='red',alpha=.08)
        fig.suptitle(f'{context} | original alpha{alpha:g} | {event} | fixed64 deterministic DEV')
        file=f'paired_{number:03d}_{context}_a{alpha:g}.png';fig.savefig(out/file,dpi=115);plt.close(fig);figures.append(file)
    fig,axes=plt.subplots(1,5,figsize=(17,4),layout='constrained')
    for ax,alpha in zip(axes,[0.,.25,.5,.75,1.]):
        for role in ROLES:
            nominal=[r for r in rows if r['role']==role and r['alpha']==alpha and not r['force_N']]
            disturbed=[r for r in rows if r['role']==role and r['alpha']==alpha and r['force_N']]
            assert len(nominal)==2 and len(disturbed)==8
            unknown=sum(r['category']=='unknown' for r in nominal+disturbed)
            if unknown:
                ax.text(.02,.98-ROLES.index(role)*.1,f'{role}: UNKNOWN {unknown}/10; point omitted',
                    color=colors[role],transform=ax.transAxes,va='top',fontsize=5);continue
            ax.scatter(np.mean([r['scored_cost'] for r in nominal]),np.mean([r['category'] in ['safe_task','safe_task_failed'] for r in disturbed]),
                color=colors[role],marker=dict(zip(ROLES,['o','s','^','D']))[role],label=role)
        ax.set(title=f'Original alpha{alpha:g}',xlabel='Mean nominal originalalpha cost',ylabel='Disturbed survival (8 branches)',ylim=(-.05,1.05));ax.grid(alpha=.2)
    axes[-1].legend(fontsize=6);file='nominal_cost_vs_disturbed_survival.png';fig.savefig(out/file,dpi=120);plt.close(fig);figures.append(file)
    return figures
