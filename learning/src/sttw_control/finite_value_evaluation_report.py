"""Read-only original-alpha/control consequence analysis of fixed300 evaluation."""
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

from .finite_value_evaluation import ROLES,CATEGORIES,paired_report,verify_gate_log
from .preference_best import atomic_json,digest


def evaluation_inputs(root,plan):
    """Read completed new records and an explicitly sealed exact cache."""
    root=Path(root)
    declared=json.loads((root/'inventory.json').read_text())['cases']
    records=json.loads((root/'physical/results.json').read_text())
    indices=json.loads((root/'physical/case_index.json').read_text())
    cached_actual=0
    if plan.get('cached_cases'):
        cache_file=root/'cache_receipt.json'
        if digest(cache_file)!=plan['sealed_files'][str(cache_file.resolve())]:
            raise ValueError('sealed cache receipt changed')
        cache=json.loads(cache_file.read_text())
        for field in ['declared','records','indices']:
            if len(cache[field])!=plan['cached_cases']:raise ValueError('incomplete exact cache')
        declared+=cache['declared'];records+=cache['records'];indices+=cache['indices']
        cached_actual=cache['active_lower_ticks']
    ids=[c['case_id'] for c in declared]
    expected=plan['declared_cases']+plan.get('cached_cases',0)
    if len(ids)!=expected or len(set(ids))!=expected or len(records)!=expected or len(indices)!=expected:
        raise ValueError('unique complete new/cache records required')
    if {c['case_id'] for c in records}!=set(ids) or {c['case_id'] for c in indices}!=set(ids):
        raise ValueError('new/cache record identities differ')
    return declared,records,indices,cached_actual


def analyze(root):
    root=Path(root).resolve();state=json.loads((root/'status.json').read_text())
    if state.get('phase')!='completed':
        raise ValueError('complete declared evaluation required; do not rerun physics')
    plan=json.loads((root/'plan.json').read_text());n=plan['declared_cases']
    roles=plan.get('evaluation_roles',ROLES)
    if state.get('phase')!='completed' or state.get('completed_cases')!=n:
        raise ValueError('complete declared evaluation required; do not rerun physics')
    out=root/'analysis';out.mkdir(exist_ok=False)
    status=dict(phase='running',pid=os.getpid(),completed_cases=0,physics_steps=0,optimizer_updates=0)
    def update(**kw):status.update(kw,checked_unix=time.time());atomic_json(out/'status.json',status)
    update()
    from prepare_short_recovery_campaign import register
    register(out/'ACTIVE_RUN.json',f'STTW_CONTROL {n}轨迹偏好抗扰只读分析',out/'status.json')
    try:
        from .anticipatory_analysis import cost_accounting,save_csv
        from .anticipatory_run import physical_outcome
        from .rejoin_comparison import geometric_task_trace
        from .survival_endpoint_task_metrics import task_metrics,goal_arrival_metrics
        from .preference_trace_rescore import rescore_trace_alpha
        from .live_endpoint_analysis import tracking_recovery_time
        p=json.loads((root/'plan.json').read_text())
        if digest(root/'plan.json')!=json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('sealed evaluation plan changed')
        for file,sha in p['sealed_files'].items():
            if digest(file)!=sha:raise ValueError('sealed evaluation input changed')
        cfg=json.loads((root/'common_environment_config.json').read_text())
        paths={}
        for context,c in p['contexts'].items():
            if digest(c['path'])!=c['path_sha256']:raise ValueError('original path changed')
            with np.load(c['path'],allow_pickle=False) as z:paths[context]={k:z[k] for k in z.files}
        declared,records,indices,cached_actual=evaluation_inputs(root,p)
        n=p.get('analysis_cases',n)
        ids={c['case_id'] for c in declared}
        if (len(declared)!=n or len(records)!=n or len(indices)!=n or len(ids)!=n
                or {c['case_id'] for c in records}!=ids or {c['case_id'] for c in indices}!=ids):
            raise ValueError('complete unique300 raw/index/definition records required')
        byid={c['case_id']:c for c in records};index={c['case_id']:c for c in indices}
        rows=[];traces={};audits=[];queries=[];actual=0
        for case in declared:
            c=byid[case['case_id']];item=index[c['case_id']]
            if any(c[k]!=v for k,v in case.items()):raise ValueError('actual case differs from sealed definition')
            file=root/item['file']
            if digest(file)!=item['sha256']:raise ValueError('raw trajectory changed')
            with np.load(file,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
            active=raw['active_tick'].astype(bool);ticks=int(active.sum());actual+=ticks
            if not np.array_equal(raw['physical_tick'][active],np.arange(ticks)):raise ValueError('incomplete/non-reset clock')
            outcome=physical_outcome(active,raw['contact_failure'],raw['nonfinite']|raw['lower_fault']|raw['policy_fault'],2000)
            if outcome!=c['physical_outcome']:raise ValueError('physical label changed')
            verify_gate_log(raw,c)
            if p.get('initial_state_confirmation'):
                from .nominal_initial_confirmation import verify_initial_feedback
                verify_initial_feedback(raw,c,cfg,p)
            elif 'feedback_kind' in c:
                from .nominal_reference_feedback import verify_nominal_feedback
                profile=p['nominal_profiles'][f"{c['context']}/a{c['alpha']:g}"]
                with np.load(profile['file'],allow_pickle=False) as z:table={k:z[k] for k in z.files}
                verify_nominal_feedback(raw,c,cfg,p['nominal_feedback_spec'],table)
            elif 'feedback_gain_rad' in c:
                from .reference_task_feedback import verify_feedback_log
                verify_feedback_log(raw,c,cfg,p['feedback_spec'])
            path=paths[c['context']];geometry=geometric_task_trace(raw,path,cfg,initial_xy=item['initial_xy'])
            metric=task_metrics(dict(raw,**geometry,domain_exit=np.zeros_like(active),task_reward_valid=active),
                                speed=c['requested_speed_m_s'],goal_s=float(path['goal']),config=cfg)
            category=('unknown' if outcome=='unknown' or metric['task_recovered'] is None else
                      'contact' if outcome=='contact' else 'safe_task' if metric['task_recovered'] else 'safe_task_failed')
            cost=cost_accounting(raw,scale=cfg['path_reward']['scale'],dt=.005)
            band=tracking_recovery_time(dict(raw,**geometry),speed=c['requested_speed_m_s'],
                                        onset_tick=c['pulse_onset_task_tick'],config=cfg)
            for alpha in [0.,.25,.5,.75,1.]:
                q=rescore_trace_alpha(raw,alpha=alpha,config=cfg)
                if alpha==c['alpha']:np.testing.assert_allclose(q['scored_reward'],raw['scored_tick_reward'],rtol=2e-5,atol=2e-6)
                queries.append(dict(case_id=c['case_id'],executed_alpha=c['alpha'],objective_alpha=alpha,scored_cost=q['scored_cost']))
            ref=raw['reference'][active];xy=raw['actual_xy'][active]
            impulse=raw['applied_force_impulse'][active].sum(0,dtype=np.float64)
            expected=c['force_N']*c['pulse_duration_s']*np.asarray(c['world_direction'])
            row=dict(c,category=category,**metric,**cost,raw_file=str(file),raw_sha256=item['sha256'],
                band_return_status=band['status'],band_return_delay_s=band['delay_from_onset_s'],
                max_original_path_error_m=float(abs(geometry['path_cross_track'][active]).max()),
                traveled_distance_m=float(np.linalg.norm(np.diff(np.vstack([item['initial_xy'],xy]),axis=0),axis=1).sum()),
                reference_speed_variation_m_s=float(abs(np.diff(ref[:,0])).sum()),
                reference_steer_variation_rad=float(abs(np.diff(ref[:,1])).sum()),
                causal_support_gate_s=float(np.sum(active&raw['relay_active'])*.005),
                legal_support_s=float(np.sum(active&raw['actor_action_valid']&(raw['relay_profile']==7))*.005),
                full_pulse_exposure=bool(np.allclose(impulse,expected,atol=.025,rtol=.05)))
            if 'goal_radius_m' in p:
                goal_xy=[np.interp(float(path['goal']),path['s'],path['xy'][:,i]) for i in [0,1]]
                row.update(goal_arrival_metrics(raw,goal_xy=goal_xy,radius_m=p['goal_radius_m'],dt=.005,horizon_s=10.))
            rows.append(row);traces[c['case_id']]=dict(raw,**geometry)
            audits.append(dict(case_id=c['case_id'],raw_sha256=item['sha256'],reward_reconstruction=True,
                               physical_label_verified=True,measured_gate_replayed=True))
            if len(rows)%30==0:update(completed_cases=len(rows))
        if actual!=state['active_lower_ticks']+cached_actual:raise ValueError('actual lower cost accounting mismatch')
        summary={name:paired_report(group,roles=roles) for name,group in dict(all=rows,
            nominal=[r for r in rows if not r['force_N']],disturbed=[r for r in rows if r['force_N']]).items()}
        alphas=sorted({r['alpha'] for r in rows})
        byalpha={str(a):paired_report([r for r in rows if r['alpha']==a],roles=roles) for a in alphas}
        zeros={(r['context'],r['role'],r['alpha']):r for r in rows if not r['force_N']}
        if len(zeros)!=len(paths)*len(roles)*len(alphas):raise ValueError('one shared zero per route/role/alpha required')
        own=[]
        for r in rows:
            if not r['force_N']:continue
            z=zeros[r['context'],r['role'],r['alpha']];unknown='unknown' in [r['category'],z['category']]
            own.append(dict(case_id=r['case_id'],zero_case_id=z['case_id'],
                scored_cost_change=None if unknown else r['scored_cost']-z['scored_cost'],
                physical_change=None if unknown else int(r['category']!='contact')-int(z['category']!='contact'),
                task_change=None if unknown else int(r['category']=='safe_task')-int(z['category']=='safe_task')))
        atomic_json(out/'results.json',rows);save_csv(out/'results.csv',rows)
        atomic_json(out/'summary.json',summary);atomic_json(out/'by_alpha.json',byalpha)
        save_csv(out/'own_zero_changes.csv',own);save_csv(out/'fixed_trace_objective_queries.csv',queries)
        pairs=[dict(comparison=name,**d) for name,v in summary['all']['comparisons'].items() for d in v['details']]
        save_csv(out/'paired_changes.csv',pairs);atomic_json(out/'raw_audit.json',audits)
        if 'goal_radius_m' in p:
            atomic_json(out/'goal_summary.json',goal_report(rows,roles))
        figures=render(out,rows,traces,paths,cfg,goal_radius=p.get('goal_radius_m'))
        atomic_json(out/'receipt.json',dict(raw_cases=n,new_cases=p['declared_cases'],cached_cases=p.get('cached_cases',0),
            new_actual_lower_ticks=state['active_lower_ticks'],cached_actual_lower_ticks=cached_actual,
            fixed_trace_queries=n*5,actual_lower_ticks=actual,
            figures=figures,physics_steps=0,optimizer_updates=0,visual_review_complete=False,
            scope=p.get('data_role','Correlated DEV branches, not independent final test.')))
        lines=['# 固定64终点：价值/回报估计器三臂与简单强对照','',
            '300条确定性完整DEV轨迹；原alpha、任务与参考权限保持。图已生成待人工裁决，未采用新模型。','',
            '|角色|未摔且任务完成|未摔但任务失败|触地|未知|','|---|---:|---:|---:|---:|']
        for role,counts in summary['all']['role_counts'].items():lines.append('|'+role+'|'+'|'.join(str(counts[k]) for k in CATEGORIES)+'|')
        if 'goal_radius_m' in p:
            lines=[f'# 新几何低速：冻结策略与参考支持的{n}轨迹对照','',
                f"主要任务为进入原目标{p['goal_radius_m']*100:g}cm区域，不要求末段保持。整窗触地和旧严格跟踪指标另存，不作主要任务门槛。",'',
                '|角色|目标到达|整窗未触地|触地|到达未知|','|---|---:|---:|---:|---:|']
            for role,counts in goal_report(rows,roles)['all']['counts'].items():
                lines.append(f"|{role}|{counts['arrived']}|{counts['survived']}|{counts['contact']}|{counts['unknown']}|")
        lines+=['','配对新增/丢失、原alpha成本与自身zero变化分别保存；未知不按失败或成功填补。',
                '模型与参考模式按封存身份分别报告；goal_summary为主要到达指标，旧严格跟踪仅作诊断。' if 'goal_radius_m' in p else
                '结构/估计器比较同时改变优化条件，MC也改变优势目标；不能据单一曲线归因或称首次创新。',
                '全部实际XY、20ms奖励/累计奖励及逐分量/累计分量已导出，轨迹在真实终点截止。',
                '后续裁决看目标到达、整窗防摔、名义保留及原alpha代价；旧末段保持不作新任务门槛。' if 'goal_radius_m' in p else
                '后续裁决需同时看防摔、名义保留、受扰末段、路径/速度误差及任务代价，不能只看RMSE。']
        (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
        update(phase='completed',stage='numeric_and_figures_complete_review_pending',completed_cases=n)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise



def role_labels(rows):
    """Make execution mode visible when a learned Actor also uses support."""
    labels={}
    for row in rows:
        label=row['role']
        if row.get('feedback_gain_rad',0)>0:
            actor_label='parent task_only64' if row['role']=='parent_feedback_010' else 'free_GAE'
            label=f"{actor_label} + steering {row['feedback_gain_rad']:g}rad"
        if row.get('feedback_kind')==1:label='free_GAE + nominal-deviation feedback 0.1rad'
        if row.get('feedback_kind')==2:label='free_GAE + existing-gate feedback 0.1rad'
        if row['role']=='feedback_shared':label='free_GAE (original alpha) + common alpha0.5 profile feedback 0.1rad'
        if row.get('causal_support') and label!='parent_causal':
            label+=' + short support'
        if row['role'] in labels and labels[row['role']]!=label:
            raise ValueError('mixed support modes under one displayed role')
        labels[row['role']]=label
    return labels


def goal_report(rows,roles):
    baseline='parent_plain' if 'parent_plain' in roles else 'parent_causal' if 'parent_causal' in roles else roles[0]
    result={}
    for split,group in dict(all=rows,nominal=[r for r in rows if not r['force_N']],
                           disturbed=[r for r in rows if r['force_N']]).items():
        counts={}
        for role in roles:
            rr=[r for r in group if r['role']==role]
            counts[role]=dict(total=len(rr),arrived=sum(r['goal_arrived'] is True for r in rr),
                unknown=sum(r['goal_arrived'] is None for r in rr),survived=sum(r['physical_outcome']=='survived' for r in rr),
                contact=sum(r['physical_outcome']=='contact' for r in rr),mean_original_alpha_cost=float(np.mean([r['scored_cost'] for r in rr])) if rr else None)
        pairs={}
        groups={}
        for row in group:groups.setdefault((row['context'],row['alpha'],row['event_id']),{})[row['role']]=row
        for role in roles:
            if role==baseline:continue
            gain=loss=unknown=0
            for cell in groups.values():
                a,b=cell[baseline]['goal_arrived'],cell[role]['goal_arrived']
                if a is None or b is None:unknown+=1
                else:gain+=b and not a;loss+=a and not b
            pairs[role+'_vs_'+baseline]=dict(arrival_gains=gain,arrival_losses=loss,unknown=unknown)
        result[split]=dict(counts=counts,paired=pairs)
    return result


def render(out,rows,traces,paths,cfg,*,goal_radius=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .live_endpoint_analysis import block_reward_series
    from .path_command_reward import configured_cost_keys
    from .reference_task_feedback import PROBE_ROLES
    from .nominal_reference_feedback import NOMINAL_ROLES
    from .nominal_table_ablation import SHARED_ROLE,PARENT_ROLE
    available=ROLES+PROBE_ROLES+NOMINAL_ROLES+(SHARED_ROLE,PARENT_ROLE)
    used_roles=[role for role in available if any(r['role']==role for r in rows)]
    colors=dict(zip(used_roles,['#0072B2','#E69F00','#CC79A7','#009E73','#D55E00','#56B4E9','#000000','#999999']))
    styles=dict(zip(used_roles,[':', '-.', ':', '--', '-', '-.','--','-']))
    groups={};figures=[];cost_keys=list(configured_cost_keys(cfg));labels_by_role=role_labels(rows)
    for r in rows:groups.setdefault((r['context'],r['alpha'],r['event_id']),[]).append(r)
    alphas=sorted({r['alpha'] for r in rows})
    columns=max(sum(key[0]==context for key in groups)//len(alphas) for context in paths)
    overviews={context:plt.subplots(len(alphas),columns,figsize=(max(18,3*columns),3.2*len(alphas)),squeeze=False,layout='constrained') for context in paths}
    counters={context:0 for context in paths}
    cell_counters={(context,alpha):0 for context in paths for alpha in alphas}
    for number,((context,alpha,event),group) in enumerate(groups.items()):
        path=paths[context];reference=path['xy'][path['s']<=float(path['goal'])]
        fig,grid=plt.subplots(4,3,figsize=(15,14),layout='constrained');axes=list(grid.flat)
        overview,oa=overviews[context];small=oa[alphas.index(alpha),cell_counters[context,alpha]]
        cell_counters[context,alpha]+=1;counters[context]+=1
        for ax in [axes[0],small]:ax.plot(reference[:,0],reference[:,1],':',color='.55');ax.scatter(*reference[-1],marker='*',color='k')
        if goal_radius is not None:
            from matplotlib.patches import Circle
            goal=[np.interp(float(path['goal']),path['s'],path['xy'][:,i]) for i in [0,1]]
            for ax in [axes[0],small]:ax.add_patch(Circle(goal,goal_radius,fill=False,color='black'))
        ncols=4;nrows=(2*(len(cost_keys)+1)+ncols-1)//ncols
        cf,cgrid=plt.subplots(nrows,ncols,figsize=(18,3*nrows),layout='constrained');ca=list(cgrid.flat)
        clouds=[reference]
        for r in group:
            role=r['role'];d=traces[r['case_id']];a=d['active_tick'];t=d['physical_tick'][a]*.005;xy=d['actual_xy'][a]
            clouds.append(xy);style=dict(color=colors[role],ls=styles[role],label=labels_by_role[role])
            for ax in [axes[0],small]:
                ax.plot(xy[:,0],xy[:,1],**style);ax.scatter(*xy[-1],color=colors[role],marker='x' if r['category']=='contact' else 'o',s=24)
            for i,key in [(1,'path_cross_track'),(3,'path_heading_error'),(4,'phi'),(5,'actual_delta'),(10,'phi_dot')]:axes[i].plot(t,d[key][a],**style)
            axes[2].plot(t,d['actual_forward_speed'][a],**style)
            axes[2].plot(t,d['wheel_speed_proxy'][a],color=colors[role],ls=':',alpha=.5)
            axes[2].plot(t,d['reference'][a,0],color=colors[role],ls='--',alpha=.4)
            axes[8].plot(t,d['actor_action_valid'][a],**style)
            axes[9].plot(t,d['relay_profile'][a]/7.,**style)
            axes[11].plot(t,d['wheel_speed_proxy'][a]-d['actual_forward_speed'][a],**style)
            series=block_reward_series(d,config=cfg);u=series['time_s'];reward=series['scored_reward']
            axes[6].plot(u,reward,**style);axes[7].plot(u,np.cumsum(reward),**style)
            components=np.column_stack([series['signed_running_components'],series['terminal_replacement_reward']])
            for i,key in enumerate(cost_keys+['terminal_replacement']):
                ca[2*i].plot(u,components[:,i],**style);ca[2*i+1].plot(u,np.cumsum(components[:,i]),**style)
                ca[2*i].set_title(key+' signed reward /20ms');ca[2*i+1].set_title(key+' cumulative reward')
            np.savez_compressed(out/f'series_{number:03d}_{role}.npz',**series,cumulative_reward=np.cumsum(reward),
                cumulative_running_components=np.cumsum(series['signed_running_components'],axis=0),
                cumulative_terminal_replacement_reward=np.cumsum(series['terminal_replacement_reward']),
                cost_keys=np.asarray(cost_keys),actual_time_s=t,xy=xy,path_error=d['path_cross_track'][a],
                heading_error=d['path_heading_error'][a],reference=d['reference'][a],nominal_reference=d['nominal_reference'][a],
                actual_speed=d['actual_forward_speed'][a],wheel_speed_proxy=d['wheel_speed_proxy'][a],
                actual_delta=d['actual_delta'][a],actor_legal=d['actor_action_valid'][a],relay_profile=d['relay_profile'][a])
        cloud=np.vstack(clouds);lo=cloud.min(0)-.4;hi=cloud.max(0)+.4
        for ax in [axes[0],small]:ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='World X (m)',ylabel='World Y (m)');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
        small.set_title(f'alpha{alpha:g} {event}',fontsize=8);axes[0].legend(fontsize=7)
        titles=['Actual XY / original goal','Signed path error (m)','Speed: actual / wheel proxy(dotted) / ref(dashed)',
                'Signed heading error (rad)','Roll (rad)','Actual steering (rad)','Reward per20ms block',
                'Cumulative originalalpha reward','Legal Actor decisions','Requested support profile','Roll rate (rad/s)','Wheel proxy minus actual speed (m/s)']
        for i,(ax,title) in enumerate(zip(axes,titles)):
            ax.set_title(title,fontsize=9);ax.grid(alpha=.2)
            if i:ax.set(xlabel='Task time (s)',xlim=(0,10))
        axes[2].axhline(group[0]['requested_speed_m_s'],color='k',ls=':',lw=.8)
        used=ca[:2*(len(cost_keys)+1)]
        for ax in used:ax.set(xlabel='Task time (s)',xlim=(0,10));ax.grid(alpha=.2)
        for ax in ca[len(used):]:ax.set_visible(False)
        if group[0]['force_N']:
            start=group[0]['pulse_onset_task_tick']*.005
            for ax in axes[1:]+used:ax.axvspan(start,start+group[0]['pulse_duration_s'],color='red',alpha=.08)
        title=f'{context} | original alpha{alpha:g} | {event} | deterministic DEV'
        fig.suptitle(title);cf.suptitle(title+' | signed and cumulative original reward components')
        for figure,prefix in [(fig,'paired'),(cf,'components')]:
            base=f'{prefix}_{number:03d}_{context}_a{alpha:g}'
            for suffix in ['png','pdf']:figure.savefig(out/(base+'.'+suffix),dpi=110);figures.append(base+'.'+suffix)
            plt.close(figure)
    for context,(figure,axes) in overviews.items():
        handles,labels=axes[0,0].get_legend_handles_labels();figure.legend(handles,labels,loc='outside lower center',ncol=6,fontsize=8)
        figure.suptitle(context+f' | all{counters[context]} events /{sum(r["context"]==context for r in rows)} actual trajectories, original paths and true endpoints')
        name='all_xy_'+context+'.png';figure.savefig(out/name,dpi=120);plt.close(figure);figures.append(name)
    return figures+render_tradeoff(out,rows,context_count=len(paths),goal_radius=goal_radius)


def render_tradeoff(out,rows,*,context_count,goal_radius=None):
    """Render derived tradeoff points without regenerating raw diagnostics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .reference_task_feedback import PROBE_ROLES
    from .nominal_reference_feedback import NOMINAL_ROLES
    from .nominal_table_ablation import SHARED_ROLE,PARENT_ROLE
    available=ROLES+PROBE_ROLES+NOMINAL_ROLES+(SHARED_ROLE,PARENT_ROLE)
    used_roles=[role for role in available if any(r['role']==role for r in rows)]
    colors=dict(zip(used_roles,['#0072B2','#E69F00','#CC79A7','#009E73','#D55E00','#56B4E9','#000000','#999999']))
    labels_by_role=role_labels(rows)
    alphas=sorted({r['alpha'] for r in rows})
    figures=[]
    tradeoff,axes=plt.subplots(2,len(alphas),figsize=(3.8*len(alphas),8),squeeze=False,layout='constrained')
    markers=dict(zip(used_roles,['o','s','^','D','P','X','v','*']))
    for column,alpha in enumerate(alphas):
        for role in used_roles:
            nominal=[r for r in rows if r['role']==role and r['alpha']==alpha and not r['force_N']]
            disturbed=[r for r in rows if r['role']==role and r['alpha']==alpha and r['force_N']]
            expected_disturbed=len({(r['context'],r['event_id']) for r in rows if r['alpha']==alpha and r['force_N']})
            if len(nominal)!=context_count or len(disturbed)!=expected_disturbed:raise ValueError('matched route zeros and complete disturbed branches required')
            unknown=sum(r['category']=='unknown' for r in nominal+disturbed)
            if unknown:
                for ax in axes[:,column]:ax.text(.02,.98-used_roles.index(role)*.07,role+f': {unknown} UNKNOWN; omitted',
                    transform=ax.transAxes,va='top',fontsize=5,color=colors[role])
                continue
            x=float(np.mean([r['scored_cost'] for r in nominal]))
            physical=float(np.mean([r['category'] in ['safe_task','safe_task_failed'] for r in disturbed]))
            task=float(np.mean([r['goal_arrived'] is True if goal_radius is not None else r['category']=='safe_task' for r in disturbed]))
            for ax,y in zip(axes[:,column],[physical,task]):ax.scatter(x,y,color=colors[role],marker=markers[role],label=labels_by_role[role])
        for row,ax in enumerate(axes[:,column]):
            ax.set(title=f'Original alpha{alpha:g}',xlabel='Mean nominal originalalpha cost',
                   ylabel='Disturbed survival' if row==0 else ('Disturbed goal arrival' if goal_radius is not None else 'Disturbed joint task success'),ylim=(-.05,1.05));ax.grid(alpha=.2)
    axes[0,-1].legend(fontsize=6)
    tradeoff.suptitle(('Fixed-reference DEV: nominal cost / disturbed goal arrival; correlated branches, no continuous frontier or independent CI' if goal_radius is not None else
        'Empirical DEV cost / capability points; correlated branches, no continuous frontier or independent CI'))
    for suffix in ['png','pdf']:
        name='nominal_cost_vs_disturbed_outcomes.'+suffix;tradeoff.savefig(out/name,dpi=120);figures.append(name)
    plt.close(tradeoff)
    return figures
