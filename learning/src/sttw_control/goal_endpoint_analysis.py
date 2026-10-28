"""Complete fixed-panel analysis and all-case XY; never select a model."""
import csv
import json
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest
from .goal_evaluation import score_endpoint_case,rescore_original_components


def _read_panel(root,expected_cases):
    root=Path(root).resolve();state=json.loads((root/'status.json').read_text())
    if state['phase']!='completed' or state['completed_cases']!=expected_cases:raise ValueError(f'complete fixed{expected_cases} required; incomplete/error is not failure evidence')
    plan=json.loads((root/'plan.json').read_text());index=json.loads((root/'case_index.json').read_text())
    seal=json.loads((root/'seal.json').read_text());receipt=json.loads((root/'physics_receipt.json').read_text())
    if digest(root/'plan.json')!=seal['plan_sha256'] or receipt['plan_sha256']!=seal['plan_sha256']:raise ValueError('original evaluation plan changed')
    if digest(root/'case_index.json')!=receipt['case_index_sha256']:raise ValueError('original evaluated case index changed')
    if digest(root/'path_manifest.json')!=receipt['path_manifest_sha256']:raise ValueError('original path manifest changed')
    paths=json.loads((root/'path_manifest.json').read_text())
    for item in paths.values():
        file=root/item['file']
        if digest(file)!=item['sha256']:raise ValueError('original generated path changed')
        with np.load(file) as z:
            goal=[float(np.interp(float(z['goal']),z['s'],z['xy'][:,i])) for i in range(2)]
        if not np.allclose(goal,item['original_goal_xy'],rtol=0,atol=1e-6):raise ValueError('original path goal mismatch')
    if len(index)!=expected_cases or len({r['case']['case_id'] for r in index})!=expected_cases:raise ValueError('missing or duplicated original inventory')
    if [r['case'] for r in index]!=plan['cases']:raise ValueError('original case identity changed')
    cfg=json.loads((Path(plan['training'])/'effective_config.json').read_text())
    if digest(Path(plan['training'])/'effective_config.json')!=plan['inputs'][str(Path(plan['training'])/'effective_config.json')]:raise ValueError('analysis config identity drift')
    return plan,index,paths,cfg


def _validate_h75_comparison(primary,supplement,primary_paths,supplement_paths):
    from .goal_endpoint_run import h75_supplement_cases
    if not supplement.get('h75_supplement',False) or supplement['cases']!=h75_supplement_cases(primary['cases']):
        raise ValueError('supplement is not the original48 H75 queries')
    if primary.get('replanning')!=supplement.get('replanning'):
        raise ValueError('H75 comparison contract differs: replanning')
    for key in ('training','endpoint','horizon_s','radius_m'):
        if primary[key]!=supplement[key]:raise ValueError('H75 comparison contract differs: '+key)
    for name in (str(Path(primary['training'])/'plan.json'),str(Path(primary['training'])/'effective_config.json'),primary['endpoint']):
        if primary['inputs'][name]!=supplement['inputs'][name]:raise ValueError('H75 comparison input differs: '+name)
    # These sources define shared control semantics, rather than analysis-only
    # edits or an exact numerical replay gate on physical states.
    names=('teleop_env.py','path_command_env.py','live_path_continuation.py','anticipatory_reference.py',
           'reference_reentry.py','goal_evaluation.py','goal_endpoint_run.py','geometric_path.py','contact_only_environment.py')
    for name in names:
        a={k:v for k,v in primary['sources'].items() if Path(k).name==name}
        b={k:v for k,v in supplement['sources'].items() if Path(k).name==name}
        if len(a)!=1 or a!=b:raise ValueError('H75 shared control source differs: '+name)
    # Includes controller/actuator/observer/reward/network/vendor dependencies,
    # not merely the outer step. Only this post-physics analyzer is exempt.
    shared=lambda p:{k:v for k,v in p['sources'].items() if Path(k).name!='goal_endpoint_analysis.py'}
    if shared(primary)!=shared(supplement):raise ValueError('H75 shared runtime dependency differs')
    if set(primary_paths)!=set(supplement_paths):raise ValueError('H75 path inventory differs')
    for key in primary_paths:
        a=primary_paths[key];b=supplement_paths[key]
        if not np.allclose(a['original_goal_xy'],b['original_goal_xy'],rtol=0,atol=1e-6):
            raise ValueError('H75 original goal differs')


def _score_panel(root,index,paths,cfg,rows,groups):
    active_ticks=allocated_slots=0
    for item in index:
        path_key=item['case']['suite']+'/'+item['case']['context']
        if not np.allclose(item['original_goal_xy'],paths[path_key]['original_goal_xy'],rtol=0,atol=1e-6):raise ValueError('case original goal changed')
        file=root/item['file']
        if digest(file)!=item['sha256']:raise ValueError('original trajectory changed: '+item['file'])
        with np.load(file) as z:physical={k:z[k] for k in z.files}
        active_ticks+=int(np.asarray(physical['active_tick']).sum())
        allocated_slots+=int(np.asarray(physical['active_tick']).size)
        views=score_endpoint_case(physical,item['case'],item['original_goal_xy'],config=cfg)
        rows.extend(views)
        key=tuple(item['case'][k] for k in ('suite','context','force_N','pulse_onset_task_tick'))
        groups[key].append((item,physical,views))
    receipt=json.loads((root/'physics_receipt.json').read_text())
    if (active_ticks!=receipt['active_lower_ticks'] or allocated_slots!=receipt['allocated_lower_slots']):
        raise ValueError('raw trajectory totals disagree with physics receipt')


def analyze(root,output=None,*,h75_root=None):
    root=Path(root).resolve();plan,index,paths,cfg=_read_panel(root,128)
    panels=[(root,index,paths,cfg)]
    roles=['fixed_endpoint64','parent_task64','zero','zero_plus_profile7_speed_support']
    source=dict(plan_sha256=digest(root/'plan.json'),case_index_sha256=digest(root/'case_index.json'))
    if h75_root is not None:
        h75_root=Path(h75_root).resolve()
        other,other_index,other_paths,other_cfg=_read_panel(h75_root,48)
        _validate_h75_comparison(plan,other,paths,other_paths)
        if cfg!=other_cfg:raise ValueError('H75 environment config differs')
        # Independently bound paths can differ in file container bytes, but
        # their actual numerical geometry must describe the same task.
        for key in paths:
            with np.load(root/paths[key]['file']) as a,np.load(h75_root/other_paths[key]['file']) as b:
                if set(a.files)!=set(b.files) or any(a[k].shape!=b[k].shape or not np.allclose(a[k],b[k],rtol=1e-6,atol=1e-6) for k in a.files):
                    raise ValueError('H75 generated path geometry differs')
        panels.append((h75_root,other_index,other_paths,other_cfg));roles.append('frozen_H75')
        source['h75']=dict(root=str(h75_root),plan_sha256=digest(h75_root/'plan.json'),case_index_sha256=digest(h75_root/'case_index.json'))
    output=(root/'analysis_with_h75' if h75_root is not None else root/'analysis') if output is None else Path(output).resolve()
    output.mkdir(exist_ok=False)
    rows=[];groups=defaultdict(list)
    for panel in panels:_score_panel(*panel,rows,groups)
    physical_count=128+(48 if h75_root is not None else 0)
    view_count=192+(48 if h75_root is not None else 0)
    if len(rows)!=view_count:raise ValueError('all declared alpha score aliases required')
    summary=[]
    for suite in ('transfer','training_geometry'):
        for role in roles:
            selected=[r for r in rows if r['suite']==suite and r['role']==role]
            distinct={r['case_id']:r for r in selected};counts=Counter((r['goal'],r['survival']) for r in distinct.values())
            summary.append(dict(suite=suite,role=role,physical_count=len(distinct),score_views=len(selected),
                arrived=sum(r['goal']=='arrived' for r in distinct.values()),survived=sum(r['survival']=='survived' for r in distinct.values()),
                arrived_alive=counts['arrived','survived'],alive_not_arrived=counts['not_arrived','survived'],
                arrived_contact=counts['arrived','contact'],not_arrived_contact=counts['not_arrived','contact'],
                contact=sum(r['survival']=='contact' for r in distinct.values()),
                unknown=sum(r['survival'] not in ('survived','contact') for r in distinct.values()),
                pulse_timing_mismatches=sum(r['pulse_timing_status']=='mismatch' for r in distinct.values()),
                pulse_timing_unknown=sum(r['pulse_timing_status']=='unknown' for r in distinct.values())))
    match=defaultdict(dict)
    for row in rows:
        key=tuple(row[k] for k in ('suite','context','force_N','onset_task_tick','user_alpha'))
        if row['role'] in match[key]:raise ValueError('duplicate matched user query')
        match[key][row['role']]=row
    pairs=[]
    for key,comparison in match.items():
        if set(comparison)!=set(roles):raise ValueError('incomplete same-event control panel')
        a=comparison['fixed_endpoint64']
        for role,b in comparison.items():
            if role=='fixed_endpoint64':continue
            item=dict(suite=key[0],context=key[1],force_N=key[2],onset_task_tick=key[3],user_alpha=key[4],baseline=role,
                learned_case_id=a['case_id'],baseline_case_id=b['case_id'],learned_exposure=a['pulse_exposure'],baseline_exposure=b['pulse_exposure'],
                learned_pulse_timing=a['pulse_timing_status'],baseline_pulse_timing=b['pulse_timing_status'])
            for name,known,positive in [('goal',('arrived','not_arrived'),'arrived'),('survival',('survived','contact'),'survived')]:
                item[name+'_change']=int(a[name]==positive)-int(b[name]==positive) if a[name] in known and b[name] in known else None
            item['raw_pre_goal_cost_difference']=a['pre_goal_cost']-b['pre_goal_cost'] if a['pre_goal_cost'] is not None and b['pre_goal_cost'] is not None else None
            complete=a['survival']==b['survival']=='survived' and a['physical_ticks']==b['physical_ticks']==2000
            item['both_complete10s']=complete
            for metric in ('path_rmse_m','speed_rmse_m_s'):
                item[metric+'_difference']=a[metric]-b[metric] if complete else None
            pairs.append(item)
    preference=[]
    for event in sorted({key[:-1] for key in match}):
        for role in ['fixed_endpoint64','parent_task64']+(['frozen_H75'] if h75_root is not None else []):
            a=match[event+(0.,)][role];b=match[event+(1.,)][role]
            complete=a['survival']==b['survival']=='survived' and a['physical_ticks']==b['physical_ticks']==2000
            preference.append(dict(event=list(event),role=role,alpha0_goal=a['goal'],alpha1_goal=b['goal'],
                alpha0_survival=a['survival'],alpha1_survival=b['survival'],both_complete10s=complete,
                speed_rmse_a1_minus_a0=b['speed_rmse_m_s']-a['speed_rmse_m_s'] if complete else None,
                path_rmse_a1_minus_a0=b['path_rmse_m']-a['path_rmse_m'] if complete else None,
                scope='Conditional common10s physical-metric difference, not population preference-retention proof; failed cases retained above'))
    figures=[]
    for key,items in groups.items():figures.append(_plot(output,key,items,root,cfg))
    report=dict(physical_summary=summary,paired_query_changes=pairs,preference_endpoint_pairs=preference,score_views=rows,figures=figures,
        source=source,
        interpretation=[f'{physical_count} physical executions;{view_count} correlated alpha score views, not independent trials.',
            'Transfer jointly changes route radius, requested speed and onset, using related original physical origins and declared upper/lower-history reset.',
            'G is first1m original-goal entry before contact/fault; S is whole10s no-contact. Latercontact does not erase earlierG.',
            'Cost is original-unit running cost, contact/fault transitions excluded; legacy scored return retains entire factual terminal replacement block.',
            'minimum_goal_distance includes a fatal sample; minimum_prefatal_goal_distance separately excludes it and must not be confused with goal admission.',
            'No p-values, independent-seed SD or continuous force threshold inferred from this deterministic DEV panel.',
            'Figures are internal provisional PNGs, not publisher-compliance evidence; manual visual review remains required.'])
    if plan.get('replanning'):
        report['interpretation'].append('Path costs computed against replanned geometry are execution-route diagnostics; they cannot be called original-route J. Original-goal admission and original reference plots retain the frozen task.')
    atomic_json(output/'results.json',report)
    with (output/'results.csv').open('w',newline='') as f:
        fields=[k for k in rows[0] if k not in ('pre_goal_component_cost','impulse_Ns','expected_impulse_Ns')]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    mode='周期重规划' if plan.get('replanning') else '固定路线'
    lines=[f'# 固定64普通目标学习（{mode}）：完整{physical_count}物理轨迹评价','',
        '内部DEV记录，尚未人工核验投稿声明。先查全部XY，再解读配对变化；没有事后选checkpoint。','',
        '|面板|方法|物理轨迹|评分视图|到达|完整存活|到达且存活|存活未到|到达后接触|未到且接触|接触合计|未知|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append('|'+ '|'.join(str(r[k]) for k in ['suite','role','physical_count','score_views','arrived','survived','arrived_alive','alive_not_arrived','arrived_contact','not_arrived_contact','contact','unknown'])+'|')
    if plan.get('replanning'):
        lines += ['', f'模型身份：A64为冻结普通目标 learner_0064.pt（{plan["endpoint"]}）；parent64为原任务冻结64端点，H75为冻结H75对照。仅增加所有方法共用的周期重规划；未训练新网络。', '警告：按用户alpha在执行路径上评分；重规划几何路径费用仅为执行路径诊断，不能称为原始路线J；原始目标与原始参考保持不变。']
    lines += ['', '## 全例XY与执行路径费用图' if plan.get('replanning') else '## 全例XY与原始费用图','']
    lines += [f'- [{f}](./{f})' for f in figures]
    lines += ['', '## 配对增失','', '|面板|对照|查询数（相关）|到达增/失|存活增/失|','|---|---|---:|---|---|']
    for suite in ('transfer','training_geometry'):
        for baseline in roles[1:]:
            selected=[p for p in pairs if p['suite']==suite and p['baseline']==baseline]
            changes=lambda key:f"{sum(p[key]==1 for p in selected)}/{sum(p[key]==-1 for p in selected)}"
            lines.append(f"|{suite}|{baseline}|{len(selected)}|{changes('goal_change')}|{changes('survival_change')}|")
    lines += ['', '原单位逐例费用差、G/S与closest-distance、首达时间、实际冲量在results.csv/json；未知不计成失败。',
        '对照跨α复用物理数据，配对查询相关，不能据此报告独立重复显著性。G增加不是容扰增强；固定力档不是连续防摔阈值。',
        '最近距分别列包含致命tick与prefatal两种；speed/path/heading RMSE附实际观察时长。偏好端点差只在两端均完整10s时描述，失败仍在主表，不以自身α加权费用变化证明偏好响应。',
        '另列首达前路径/速度RMSE和对应观察时长，避免把到达后的驶离直接当作到达前跟踪退化；这些各自首达窗长度不同，不据此单独宣称总体任务代价更优。',
        'Cpre到首达或触地前，完整running排除接触/故障tick；canonical legacy scored单列整块终止替代。图上费用是原单位、非优化时除B后的数值。',
        '实际/预期世界冲量和受力none/partial/full分开保留；起扰前已触地的差异不能直接解释为抗扰提升，触地/未完整受力样本仍保留原分母。zero保留原路径跟随器和冻结底层，只将中层参考修正置零。',
        '另核每个实际执行lower tick的冲量与设计起止窗口重叠，报告pulse_timing_status及实际首末受力task tick；相同总冲量不代表时序相同。提前触地后未执行的脉冲尾段属于未观察，不能误报施力错误。该核验不认证子步内力臂动态。',
        '图表先供内部审阅；正式论文需人工核对全图/原始结果与具体期刊要求。scientific-visualization1.4辅助图形设计，不认证研究结论。',
        '方法流程引用：Kassis等，Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents，2026，https://doi.org/10.48550/arXiv.2609.00065。','']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    atomic_json(output/'analysis_receipt.json',dict(phase='completed',physics_integrations=0,optimizer_steps=0,
        cases=physical_count,score_views=view_count,figures=len(figures),manual_visual_review='pending',code=__file__))
    return report


def _plot(output,key,items,root,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from .path_command_reward import configured_cost_keys
    suite,context,force,onset=key
    replanning=bool(json.loads((root/'plan.json').read_text()).get('replanning'))
    colors=['#0072B2','#D55E00','#333333','#009E73','#AA4499'];styles={0.:'-',.5:'--',1.:'-.'}
    names=['A64','parent64','zero','zero+support','H75'];keys=configured_cost_keys(cfg)
    with plt.rc_context({'font.size':8,'savefig.facecolor':'white'}):
        fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        with np.load(root/f'path_{suite}_{context}.npz') as z:
            s=z['s'];xy=z['xy'];goal_s=float(z['goal'])
        extent=max([goal_s+2]+[float(np.max(p['path_progress'][p['active_tick']]))+1 for _,p,_ in items])
        ref=s<=extent;ax[0,0].plot(xy[ref,0],xy[ref,1],':',color='#777777',label='Original route (displayed range)')
        goal=items[0][0]['original_goal_xy'];ax[0,0].add_patch(Circle(goal,1.,fill=False,color='black',linestyle='--'))
        ax[0,0].plot(*goal,'k*',label='Original goal /1m disk')
        bars=[];bar_labels=[]
        for item,p,views in items:
            c=item['case'];active=p['active_tick'];role=c['role'];alpha=c['controller_alpha']
            label=f"{names[role]} a{alpha:g}" if role in (0,1,4) else names[role]
            actual=p['actual_xy'][active]
            ax[0,0].plot(actual[:,0],actual[:,1],color=colors[role],linestyle=styles[alpha],label=label)
            if views[0]['goal_time_s'] is not None:
                hit=round(views[0]['goal_time_s']/.005)-1
                ax[0,0].plot(*p['actual_xy'][hit],marker='s',color=colors[role],markersize=4)
            first_force=views[0]['first_applied_task_tick']
            force_rows=np.flatnonzero(active & (p['physical_tick']==first_force)) if first_force is not None else []
            if len(force_rows):ax[0,0].plot(*p['actual_xy'][force_rows[0]],marker='v',color=colors[role],markersize=4)
            if len(actual):ax[0,0].plot(*actual[-1],marker='x' if np.any(p['contact_failure']&active) else 'o',color=colors[role],markersize=4)
            rows=active&~p['contact_failure']&~p['nonfinite']&~p['lower_fault']&~p['policy_fault']&p['task_reward_valid']
            shown=active.reshape(-1,4).any(axis=1);t=(np.arange(len(shown))+1)*.02
            for v in views:
                a=v['user_alpha'];components=rescore_original_components(p,user_alpha=a,config=cfg,recorded_alpha=c['controller_alpha'])
                reward=np.where(rows,-cfg['path_reward']['scale']*.005*components.sum(axis=1),0.).reshape(-1,4).sum(axis=1)
                legend=f"{names[role]} user a{a:g}"
                ax[0,1].plot(t[shown],reward[shown],color=colors[role],linestyle=styles[a],label=legend)
                ax[1,0].plot(t[shown],np.cumsum(reward)[shown],color=colors[role],linestyle=styles[a])
                bars.append(v['pre_goal_component_cost']);bar_labels.append(legend+(' UNKNOWN' if v['pre_goal_component_cost'] is None else ''))
        ax[0,0].set(xlabel='World X (m)',ylabel='World Y (m)',title='Actual XY; square first goal, x contact, o end, triangle effective force')
        ax[0,0].set_aspect('equal',adjustable='datalim');ax[0,0].legend(fontsize=5)
        ax[0,1].set(xlabel='Task time (s)',ylabel='Pre-contact running reward /20ms',title=('User alpha on EXECUTION path; no terminal replacement' if replanning else 'Original user-alpha scoring; no terminal replacement'))
        ax[0,1].legend(fontsize=5,ncol=2)
        ax[1,0].set(xlabel='Task time (s)',ylabel='Cumulative pre-contact running reward')
        values=np.array([b if b is not None else [np.nan]*len(keys) for b in bars]);bottom=np.zeros(len(values));palette=plt.get_cmap('tab20')
        for i,name in enumerate(keys):
            height=-values[:,i];ax[1,1].bar(np.arange(len(values)),height,bottom=bottom,color=palette(i),label=name,hatch=['','/','\\','.','x'][i%5]);bottom+=np.nan_to_num(height)
        ax[1,1].axhline(0,color='black',linewidth=.5);ax[1,1].set(xticks=np.arange(len(values)),xticklabels=bar_labels,ylabel='Signed pre-goal cost (raw units)')
        ax[1,1].tick_params(axis='x',labelrotation=65,labelsize=5);ax[1,1].legend(fontsize=5,ncol=2)
        fig.suptitle(f'{suite}/{context}, R={items[0][0]["case"]["path_radius_m"]}m, F={force:+g}N/.1s, onset={onset*.005:g}s; H10; {"periodic replanning" if replanning else "fixed route"}; related DEV')
        name=f'{suite}_{context}_F{force:+g}_t{onset}.png';fig.savefig(output/name,dpi=160);plt.close(fig)
    return name
