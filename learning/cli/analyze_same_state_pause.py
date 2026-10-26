"""Audit fixed same-state branches; retain physical cost and eligibility separately."""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
from sttw_control.preference_best import atomic_json,digest
from sttw_control.same_state_reference_pause import ROLES,expected_mask


def summarize(rows):
    groups={}
    for r in rows:groups.setdefault((r['context'],r['alpha'],r['event_id']),{})[r['role']]=r
    if any(set(g)!=set(ROLES) for g in groups.values()):raise ValueError('complete three-branch comparisons required')
    for g in groups.values():
        if len({r['reached_fork'] for r in g.values()})!=1:raise ValueError('branch eligibility differs within same-state group')
    output={}
    for population in ['all_events','eligible_at_fork']:
        output[population]={}
        for split in ['nominal','disturbed']:
            selected=[g for g in groups.values() if bool(g[ROLES[0]]['force_N'])==(split=='disturbed') and (population=='all_events' or g[ROLES[0]]['reached_fork'])]
            counts={}
            for role in ROLES:
                rr=[g[role] for g in selected]
                counts[role]=dict(n=len(rr),arrived=sum(r['goal_arrived'] for r in rr),survived=sum(r['physical_outcome']=='survived' for r in rr),
                    arrived_and_survived=sum(r['goal_arrived'] and r['physical_outcome']=='survived' for r in rr),
                    legal_suppression_ticks=sum(r['legal_suppression_ticks'] for r in rr),
                    same_user_cost_by_alpha={str(a):(float(np.mean([r['scored_cost'] for r in rr if r['alpha']==a])) if any(r['alpha']==a for r in rr) else None) for a in [0.,1.]})
            pairs={}
            for role in ROLES[1:]:
                detail=[]
                for g in selected:
                    a,b=g[role],g[ROLES[0]]
                    detail.append(dict(case_id=a['case_id'],baseline=b['case_id'],alpha=a['alpha'],force_N=a['force_N'],context=a['context'],
                        arrival_change=int(a['goal_arrived'])-int(b['goal_arrived']),survival_change=int(a['physical_outcome']=='survived')-int(b['physical_outcome']=='survived'),
                        same_user_cost_change=a['scored_cost']-b['scored_cost'],minimum_goal_distance_change_m=a['minimum_upright_goal_distance_m']-b['minimum_upright_goal_distance_m']))
                pairs[role]=dict(n=len(detail),arrival_gain=sum(d['arrival_change']==1 for d in detail),arrival_loss=sum(d['arrival_change']==-1 for d in detail),
                    survival_gain=sum(d['survival_change']==1 for d in detail),survival_loss=sum(d['survival_change']==-1 for d in detail),details=detail)
            output[population][split]=dict(counts=counts,paired_vs_new_full_current=pairs,
                hindsight_arrival_union=sum(any(r['goal_arrived'] for r in g.values()) for g in selected),
                hindsight_survival_union=sum(any(r['physical_outcome']=='survived' for r in g.values()) for g in selected),
                warning='Separate hindsight branch-label unions; not a deployment selector, joint achieved performance or physical upper bound.')
    return output


def analyze(root):
    from sttw_control.anticipatory_run import physical_outcome
    from sttw_control.goal_preview_baseline import verify_goal_log
    from sttw_control.finite_value_evaluation import verify_gate_log
    from sttw_control.rejoin_comparison import geometric_task_trace
    from sttw_control.survival_endpoint_task_metrics import task_metrics,goal_arrival_metrics
    from sttw_control.anticipatory_analysis import cost_accounting
    from sttw_control.preference_trace_rescore import rescore_trace_alpha
    root=Path(root).resolve();physical=json.loads((root/'status.json').read_text())
    if physical['phase']!='completed' or physical['completed_cases']!=360:raise ValueError('complete new physical collection required')
    out=root/'analysis';out.mkdir(exist_ok=False)
    status=dict(phase='running',pid=os.getpid(),completed_cases=0,new_physics_steps=0,optimizer_updates=0)
    def update(**kw):status.update(kw,checked_unix=time.time());atomic_json(out/'status.json',status)
    update()
    try:
        plan=json.loads((root/'plan.json').read_text())
        if digest(root/'plan.json')!=json.loads((root/'prelaunch_receipt.json').read_text())['plan_sha256']:raise ValueError('plan changed')
        for file,sha in plan['sealed_files'].items():
            if digest(file)!=sha:raise ValueError('sealed input changed: '+file)
        cfg=json.loads((root/'common_environment_config.json').read_text());cases=json.loads((root/'inventory.json').read_text())['cases']
        records=json.loads((root/'physical/results.json').read_text());indices=json.loads((root/'physical/case_index.json').read_text())
        if len(cases)!=360 or len(records)!=360 or len(indices)!=360:raise ValueError('all360 branches required')
        byid={r['case_id']:r for r in records};index={r['case_id']:r for r in indices}
        if len(byid)!=360 or len(index)!=360 or set(byid)!=set(index) or set(byid)!={c['case_id'] for c in cases}:raise ValueError('complete unique branch identities required')
        paths={}
        for context,c in plan['contexts'].items():
            with np.load(c['path']) as z:paths[context]={k:z[k] for k in z.files}
        fork=json.loads((root/'physical/fork_receipt.json').read_text());cost=json.loads((root/'physical/cost_receipt.json').read_text())
        if not fork['all_carry_leaves_copied'] or fork['clock_reset']:raise ValueError('full common-state clone receipt required')
        if cost['new_physics_ticks']!=physical['active_lower_ticks'] or cost['prefix_actual_lower_ticks']+cost['suffix_actual_lower_ticks']!=physical['active_lower_ticks']:raise ValueError('shared-prefix physical accounting mismatch')
        if cost['prefix_allocated_slots']+cost['suffix_allocated_slots']!=physical['allocated_lower_slots']:raise ValueError('padding accounting mismatch')
        rows=[];audits=[];prefix_leaves={};suffix_total=0;logical_total=0
        for c in cases:
            r=byid[c['case_id']];item=index[c['case_id']]
            if any(r[k]!=v for k,v in c.items()):raise ValueError('execution metadata differs from sealed case')
            file=root/item['file']
            if digest(file)!=item['sha256']:raise ValueError('branch raw changed')
            with np.load(file) as z:raw={k:z[k] for k in z.files}
            a=raw['active_tick'].astype(bool);ticks=int(a.sum());logical_total+=ticks;suffix_total+=r['suffix_physical_ticks']
            np.testing.assert_array_equal(raw['physical_tick'][a],np.arange(ticks))
            prefix_n=r['prefix_physical_ticks'];logical_key=(r['context'],r['alpha'],r['event_id'])
            # Compare EVERY logged prefix field, not only pose/velocity.
            shared={k:v[:prefix_n] for k,v in raw.items() if not k.startswith('branch_')}
            if logical_key not in prefix_leaves:prefix_leaves[logical_key]=shared
            else:
                for key,value in shared.items():np.testing.assert_array_equal(value,prefix_leaves[logical_key][key])
            eligible=bool(fork['reached_fork'][c['prefix_index']])
            if eligible!=r['reached_fork']:raise ValueError('fork eligibility changed')
            if not eligible and r['pause_requested_active_lower_ticks']:raise ValueError('pre-fork terminal mislabeled as treated')
            mask=expected_mask(raw['physical_tick'],c['fork_task_tick'],c['pause_mask'])
            window=(raw['physical_tick']>=c['fork_task_tick'])&(raw['physical_tick']<c['fork_task_tick']+80)
            np.testing.assert_array_equal(raw['branch_pause_active'][a],window[a])
            np.testing.assert_array_equal(raw['branch_fork_task_tick'][a],np.full(ticks,c['fork_task_tick']))
            verify_gate_log(raw,c)
            goal_audit=verify_goal_log(raw,c,cfg,plan['feedback_spec'],paths[c['context']],expected_keep=mask)
            outcome=physical_outcome(a,raw['contact_failure'],raw['nonfinite']|raw['lower_fault']|raw['policy_fault'],2000)
            if outcome!=r['physical_outcome'] or outcome=='unknown':raise ValueError('unknown or inconsistent physical label')
            geometry=geometric_task_trace(raw,paths[c['context']],cfg,initial_xy=item['initial_xy'])
            metric=task_metrics(dict(raw,**geometry,domain_exit=np.zeros_like(a),task_reward_valid=a),speed=c['requested_speed_m_s'],goal_s=float(paths[c['context']]['goal']),config=cfg)
            path=paths[c['context']];point=[np.interp(float(path['goal']),path['s'],path['xy'][:,i]) for i in [0,1]]
            goal=goal_arrival_metrics(raw,goal_xy=point,radius_m=.5,dt=.005,horizon_s=10.)
            q=rescore_trace_alpha(raw,alpha=c['alpha'],config=cfg)
            np.testing.assert_allclose(q['scored_reward'],raw['scored_tick_reward'],rtol=2e-5,atol=2e-6)
            running=cost_accounting(raw,scale=cfg['path_reward']['scale'],dt=.005)
            legal=a&window&raw['actor_action_valid']
            suppressed=legal&np.any(~mask,axis=1)
            delta=raw['original_actor_latent']-raw['source_actor_latent']
            row=dict(r);row.update(metric);row.update(running);row.update(goal)
            row.update(raw_file=str(file),raw_sha256=item['sha256'],
                legal_suppression_ticks=int(suppressed.sum()),neural_latent_change_l1=float(abs(delta[legal]).sum()),
                reference_speed_variation_m_s=float(abs(np.diff(raw['reference'][a,0])).sum()),reference_steer_variation_rad=float(abs(np.diff(raw['reference'][a,1])).sum()))
            rows.append(row)
            audits.append(dict(case_id=c['case_id'],raw_sha256=item['sha256'],full_logged_prefix_equal=True,mask_and_window_verified=True,original_gate_verified=True,original_reward_reconstruction=True,physical_label_verified=True,goal_preview=goal_audit))
            if len(rows)%30==0:update(completed_cases=len(rows))
        if logical_total!=cost['logical_complete_trajectory_ticks'] or logical_total!=3*cost['prefix_actual_lower_ticks']+suffix_total or suffix_total!=cost['suffix_actual_lower_ticks']:raise ValueError('logical reuse vs actual prefix/suffix accounting mismatch')
        report=summarize(rows)
        atomic_json(out/'results.json',rows);atomic_json(out/'raw_audit.json',audits);atomic_json(out/'summary.json',report)
        lines=['# 同完整状态短暂停：固定三分支机会验证','',
            '原计划pulse结束时的同一完整carry分叉。特权实验时刻，不是部署方法；所有原120事件保留。暂停是神经请求置零，滤波/ESO/FIFO/短支持继续。完整观察到原10s，不以末段保持否决到达。','',
            '|范围|条件|固定分支|到达|整窗存活|','|---|---|---|---:|---:|']
        for population,splits in report.items():
            for split,d in splits.items():
                for role,n in d['counts'].items():lines.append(f"|{population}|{split}|{role}|{n['arrived']}/{n['n']}|{n['survived']}/{n['n']}|")
        lines+=['','各分支对本次新full_current的新增/丢失、原useralpha成本、实际合法神经请求变化见summary/results。未到fork不冒充接受干预；逻辑轨迹重复的prefix数据不重复计新物理。原始cost_receipt列实际/allocated/逻辑计数。','',
            '后验最佳分支的边际并集只说明有限集合中的机会，不当新控制策略或物理上界；本DEV已见旧结局，不是独立重复或最终TEST。一次固定集合结束，不扩窗/切点/动作库。','',
            '全部XY在五张all_xy_force图片，覆盖三分支/两alpha/12进入情境。原world路径、0.5m目标、真实终点、fork点；全active无平滑筛选，等米比例。工作图，期刊要求待定。']
        (out/'README.md').write_text('\n'.join(lines)+'\n')
        render(out,rows,paths)
        atomic_json(out/'receipt.json',dict(new_physics_steps=0,optimizer_updates=0,logical_complete_cases=360,shared_prefix_groups=120,raw_audits=360,cost=cost,visual_review_complete=False))
        update(phase='completed',stage='numeric_and_XY_complete_review_pending',completed_cases=360)
    except BaseException as exc:
        (out/'traceback.txt').write_text(traceback.format_exc());update(phase='error',error=repr(exc));raise


def render(out,rows,paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    colors=['#333333','#D55E00','#0072B2']
    for force in [0.,-16.,16.,-32.,32.]:
        fig,axes=plt.subplots(4,3,figsize=(16,15),layout='constrained')
        for ax,(context,path) in zip(axes.flat,paths.items()):
            s=path['s'];xy=path['xy'];g=float(path['goal']);point=[np.interp(g,s,xy[:,i]) for i in [0,1]]
            ax.plot(xy[s<=g,0],xy[s<=g,1],color='black',ls=':',lw=1);ax.add_patch(Circle(point,.5,fill=False,ec='black',lw=.7));ax.scatter(*point,c='black',marker='*',s=22)
            rr=[r for r in rows if r['context']==context and r['force_N']==force]
            if len(rr)!=6:raise ValueError('all six branches per force/context required')
            for role,color in zip(ROLES,colors):
                for alpha,style in [(0.,'-'),(1.,'--')]:
                    r=next(r for r in rr if r['role']==role and r['alpha']==alpha)
                    with np.load(r['raw_file']) as raw:
                        a=raw['active_tick'].astype(bool);actual=raw['actual_xy'][a]
                        if role==ROLES[0] and r['reached_fork']:ax.scatter(*actual[r['fork_task_tick']-1],c='gray',marker='D',s=14)
                    ax.plot(actual[:,0],actual[:,1],color=color,ls=style,lw=1)
                    ax.scatter(*actual[-1],color=color,s=14,marker='x' if r['physical_outcome']=='contact' else 'o' if alpha==0 else 's')
            prefix,geometry=context.split('__');ax.set_title(prefix+' / '+geometry,fontsize=8);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.set_xlabel('World X (m)',fontsize=8);ax.set_ylabel('World Y (m)',fontsize=8);ax.tick_params(labelsize=7)
        handles=[Line2D([],[],color=c,label=r) for c,r in zip(colors,ROLES)]
        handles += [Line2D([],[],color='black',ls='-',marker='o',label='alpha0'),Line2D([],[],color='black',ls='--',marker='s',label='alpha1'),Line2D([],[],color='gray',marker='D',ls='',label='Common fork state')]
        fig.legend(handles=handles,loc='outside lower center',ncol=3,fontsize=9)
        fig.suptitle(f'Same complete state / three fixed reference branches: force {force:g} N, pulse0.1s\nDeclared pulse-end fork; .4s neural request pause; full original10s horizon',fontsize=12)
        fig.savefig(out/f'all_xy_force_{force:+g}.png',dpi=140);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();analyze(args.run)
