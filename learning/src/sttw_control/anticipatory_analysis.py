"""Offline original-alpha accounting on complete anticipatory-reference stages."""
import csv
import json
from pathlib import Path
import numpy as np
from .preference_best import atomic_json,digest


def cost_accounting(data,*,scale,dt):
    active=np.asarray(data['active_tick'],bool)
    components=np.asarray(data['effective_costs'],float)[active]
    running=np.asarray(data['tick_reward'],float)[active]
    scored=np.asarray(data['scored_tick_reward'],float)[active]
    if not all(np.isfinite(x).all() for x in [components,running,scored]):
        raise ValueError('nonfinite active costs')
    expected=-scale*dt*components.sum(axis=-1)
    if not np.allclose(expected,running,rtol=2e-5,atol=2e-6):
        raise ValueError('logged running reward differs from original component sum')
    running_cost=float(-running.sum());scored_cost=float(-scored.sum())
    return dict(running_cost=running_cost,scored_cost=scored_cost,
        terminal_replacement_delta=scored_cost-running_cost,
        component_integrals=(scale*dt*components.sum(axis=0)).tolist())


def paired_changes(rows,*,baseline_role='parent'):
    parents={(r['alpha'],r['event_id']):r for r in rows if r['role']==baseline_role}
    if len(parents)!=sum(r['role']==baseline_role for r in rows):raise ValueError('duplicate parent key')
    pairs=[]
    for r in rows:
        if r['role'] in (baseline_role,'zero_middle'):continue
        p=parents[(r['alpha'],r['event_id'])]
        unknown='unknown' in (r['category'],p['category'])
        safe=lambda x:x['category'] in ('safe_task','safe_task_failed')
        pairs.append(dict(case_id=r['case_id'],parent_case_id=p['case_id'],role=r['role'],
            alpha=r['alpha'],event_id=r['event_id'],
            physical_change=None if unknown else int(safe(r))-int(safe(p)),
            joint_task_change=None if unknown else int(r['category']=='safe_task')-int(p['category']=='safe_task'),
            scored_cost_change=r['scored_cost']-p['scored_cost']))
    return pairs


def save_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def load_analysis_cases(root,stage,plan):
    results=json.loads((root/stage/'results.json').read_text())
    index=json.loads((root/stage/'case_index.json').read_text())
    if len(results)!=plan['stage_counts'][stage] or len(index)!=len(results):
        raise ValueError('incomplete new physical stage inventory')
    if plan.get('inherited_cases',0):
        inherited=json.loads((root/'reuse_cases.json').read_text())
        if len(inherited)!=plan['inherited_cases']:raise ValueError('incomplete inherited inventory')
        results += [r['case'] for r in inherited];index += [r['index'] for r in inherited]
        if len(results)!=plan['logical_cases']:raise ValueError('logical case accounting mismatch')
    if len({r['case_id'] for r in results})!=len(results) or {r['case_id'] for r in index}!={r['case_id'] for r in results}:
        raise ValueError('duplicate or mismatched actual/inherited identities')
    return results,index


def analyze(root,stage,*,analysis_name='analysis'):
    if not analysis_name.startswith('analysis') or Path(analysis_name).name!=analysis_name:
        raise ValueError('local named analysis directory required')
    from .rejoin_comparison import geometric_task_trace
    from .survival_endpoint_task_metrics import task_metrics
    from .path_command_reward import configured_cost_keys
    source=root/stage;status=json.loads((source/'status.json').read_text())
    if status['phase']!='completed':raise ValueError('completed physical stage required; do not rerun collection')
    out=source/analysis_name;out.mkdir(exist_ok=False)
    cfg=json.loads((root/'common_environment_config.json').read_text())
    plan=json.loads((root/'plan.json').read_text())
    if digest(plan['path_file'])!=plan['path_sha256']:raise ValueError('original path drift')
    with np.load(plan['path_file']) as z:path={k:z[k] for k in z.files}
    results,index=load_analysis_cases(root,stage,plan)
    byid={r['case_id']:r for r in index}
    rows=[];traces={};cost_keys=list(configured_cost_keys(cfg))
    for r in results:
        item=byid[r['case_id']];file=root/item['file']
        if digest(file)!=item['sha256']:raise ValueError('raw case drift '+r['case_id'])
        with np.load(file) as z:data={k:z[k] for k in z.files}
        active=data['active_tick'];ticks=data['physical_tick'][active]
        if not len(ticks) or ticks[0]!=0 or np.any(np.diff(ticks)!=1):raise ValueError('incomplete active clock')
        geometry=geometric_task_trace(data,path,cfg,initial_xy=item['initial_xy'])
        diagnostic=dict(data,**geometry,domain_exit=np.zeros_like(active),task_reward_valid=active)
        metrics=task_metrics(diagnostic,speed=r['requested_speed_m_s'],goal_s=float(path['goal']),config=cfg)
        category=('contact' if r['physical_outcome']=='contact' else
                  'safe_task' if metrics['task_recovered'] else 'safe_task_failed')
        if r['physical_outcome']=='unknown' or metrics['task_recovered'] is None:category='unknown'
        costs=cost_accounting(data,scale=cfg['path_reward']['scale'],dt=.005)
        ref=data['reference'][active];variation=np.abs(np.diff(ref,axis=0)).sum(axis=0)
        xy=data['actual_xy'][active]
        distance=float(np.linalg.norm(np.diff(np.vstack([item['initial_xy'],xy]),axis=0),axis=1).sum())
        row=dict(r,**metrics,**costs,category=category,
            max_original_path_error_m=float(abs(geometry['path_cross_track'][active]).max()),
            traveled_distance_m=distance,reference_speed_variation_m_s=float(variation[0]),
            reference_steer_variation_rad=float(variation[1]))
        rows.append(row);traces[r['case_id']]=dict(data,**geometry)
    pairs=paired_changes(rows,baseline_role=plan.get('comparison_baseline_role','parent'))
    atomic_json(out/'results.json',rows);save_csv(out/'results.csv',rows);save_csv(out/'paired_parent.csv',pairs)
    costs=[dict(case_id=r['case_id'],**dict(zip(cost_keys,r['component_integrals']))) for r in rows]
    save_csv(out/'cost_components.csv',costs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(parent='#0072B2',relay_dual0='#E69F00',relay_dual1='#009E73',dual='#E69F00',bc='#009E73',v3='#CC79A7',task_only='#E69F00',physical_credit='#009E73',floor='#CC79A7',
        speed_gate='#D55E00',steer_gate='#56B4E9',joint_gate='#882255',causal_eso='#999933',zero_middle='#555555')
    extra_roles=sorted({r['role'] for r in rows}-set(colors))
    colors.update({role:plt.get_cmap('turbo')(i/max(1,len(extra_roles)-1)) for i,role in enumerate(extra_roles)})
    figures=[]
    cloud=np.concatenate([d['actual_xy'][d['active_tick']] for d in traces.values()])
    task_path=path['xy'][path['s']<=float(path['goal'])]
    extent=np.vstack([cloud,task_path]);lo=extent.min(axis=0)-.5;hi=extent.max(axis=0)+.5
    alphas=sorted({r['alpha'] for r in rows if r['alpha'] is not None})
    if not 1<=len(alphas)<=2:raise ValueError('one or two declared alpha values required for XY overview')
    for event in sorted({r['event_id'] for r in rows}):
        fig,grid=plt.subplots(1,len(alphas),figsize=(7*len(alphas),6),squeeze=False,layout='constrained');axes=grid[0]
        for ax,alpha in zip(axes,alphas):
            ax.plot(path['xy'][:,0],path['xy'][:,1],':',color='.5',label='Original path')
            selected=[r for r in rows if r['event_id']==event and r['alpha'] in (alpha,None)]
            for r in selected:
                d=traces[r['case_id']];xy=d['actual_xy'][d['active_tick']]
                ax.plot(xy[:,0],xy[:,1],color=colors[r['role']],label=r['role'])
                ax.scatter(*xy[-1],color=colors[r['role']],marker='x' if r['category']=='contact' else 'o',s=25)
                if r.get('handover_xy') is not None:ax.scatter(*r['handover_xy'],color=colors[r['role']],marker='s',s=22)
            ax.set(xlabel='World X (m)',ylabel='World Y (m)',title=f'{event} | original alpha {alpha:g}')
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]))
            ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.legend(fontsize=5 if len(selected)>20 else 7,ncol=2 if len(selected)>20 else 1)
        name=f'xy_{event}.png';fig.savefig(out/name,dpi=125);plt.close(fig);figures.append(name)
    # Every factual case gets its own control/physics/reward trace, including failures.
    for number,r in enumerate(rows):
        d=traces[r['case_id']];a=d['active_tick'];t=d['physical_tick'][a]*.005
        fig,axes=plt.subplots(3,2,figsize=(13,11),layout='constrained');ax=axes.flat
        ax[0].plot(t,d['path_cross_track'][a]);ax[0].set_ylabel('Original path error (m)')
        ax[1].plot(t,d['actual_forward_speed'][a]-r['requested_speed_m_s']);ax[1].set_ylabel('Speed error (m/s)')
        ax[2].plot(t,d['phi'][a],label='roll');ax[2].plot(t,d['phi_dot'][a],label='roll rate');ax[2].legend()
        for k,label in enumerate(['speed ref (m/s)','steer ref (rad)']):ax[3].plot(t,d['reference'][a,k],label=label)
        ax[3].legend();ax[4].plot(t,d['tick_reward'][a],label='running');ax[4].plot(t,d['scored_tick_reward'][a],label='scored');ax[4].legend();ax[4].set_ylabel('Reward / 5ms tick')
        for k,label in enumerate(cost_keys):ax[5].plot(t,-cfg['path_reward']['scale']*.005*d['effective_costs'][a,k],label=label)
        ax[5].legend(fontsize=6);ax[5].set_ylabel('Signed cost components')
        for axy in axes.flat:
            axy.set_xlabel('Task time (s)');axy.grid(alpha=.2)
            if r.get('handover_s') is not None:axy.axvline(r['handover_s'],color='.4',linestyle='--',alpha=.7)
            if r['force_N']:
                onset=r['pulse_onset_task_tick']*.005;axy.axvspan(onset,onset+r['pulse_duration_s'],color='red',alpha=.1)
        fig.suptitle(r['case_id']+' | '+r['category']+' | actual endpoint retained')
        name=f'trace_{number:03d}.png';fig.savefig(out/name,dpi=110);plt.close(fig);figures.append(name)
    summary={}
    for r in rows:
        key=f"{r['role']}/alpha{r['alpha']}";s=summary.setdefault(key,dict(n=0,safe_task=0,safe_task_failed=0,contact=0,unknown=0))
        s['n']+=1;s[r['category']]+=1
    atomic_json(out/'receipt.json',dict(stage=stage,source_status_sha256=digest(source/'status.json'),
        raw_cases_sha_checked=len(rows),new_actual_cases=plan['stage_counts'][stage],inherited_cases=plan.get('inherited_cases',0),comparison_baseline_role=plan.get('comparison_baseline_role','parent'),new_case_unit=plan.get('new_case_unit','full_episode'),inherited_unique_prefix_ticks=plan.get('inherited_unique_prefix_ticks',0),physics_steps=0,optimizer_updates=0,
        analyzer_sha256=digest(__file__),summary=summary,figures=figures,
        visual_review_complete=False,cost_keys=cost_keys,
        caveat='Related DEV branches. Original user-alpha cost; terminal block replacement separate from running costs. No post-hoc threshold change or controller adoption.'))
    lines=['# 提前参考阶段结果',f'阶段：{stage}；{len(rows)}条原始轨迹SHA核对，0新增仿真。',
        '原用户alpha评分，原路径与请求速度不变；离域不自动等同任务失败，真实触地与任务失败分开。',
        '成本按实际执行的原V3分量核对，终止控制块替换为失败余项的差额单列；短前缀成本小不能当更优。',
        '## 分组结局','', '|角色/alpha|n|任务且无触地|无触地任务失败|触地|未知|','|---|---:|---:|---:|---:|---:|']
    lines += ['|'+key+'|'+'|'.join(str(s[k]) for k in ['n','safe_task','safe_task_failed','contact','unknown'])+'|' for key,s in summary.items()]
    lines += ['', '逐例原alpha成本见results.csv，各分量见cost_components.csv，匹配声明基线的增失见paired_parent.csv。',
        'XY图与逐例完整时序已生成，尚需实际目视后才可声明图形审查完成。',
        '本阶段为开发筛查；不将同快照多个事件当独立重复，不把逐事件最佳候选当在线方法。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary
