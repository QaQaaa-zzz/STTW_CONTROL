"""Read-only complete paired analysis of the fixed108 permission-factor run."""
import argparse
import json
from pathlib import Path
import numpy as np
from sttw_control.preference_best import atomic_json,digest
from sttw_control.support_permission_experiment import inputs
from sttw_control.reference_preservation_confirmation import audit_pulse
from sttw_control.reference_bias_runtime import score_raw,encode_objective
from analyze_preservation_timecheck import add_radius_sensitivity

METHODS=['full','support_permission']


def paired_summary(groups):
    result={}
    for key,select in [('disturbed',lambda g:bool(g['full']['force_N'])),
                       ('nominal',lambda g:not g['full']['force_N'])]:
        chosen=[g for g in groups if select(g)]
        methods={}
        for m in METHODS:
            rr=[g[m] for g in chosen]
            fates={'arrived_survived':0,'not_arrived_survived':0,'arrived_contact':0,'not_arrived_contact':0}
            for r in rr:
                fate=('arrived' if r['goals']['1.0']['goal_arrived'] else 'not_arrived')+'_'+r['physical_outcome']
                fates[fate]+=1
            methods[m]=dict(n=len(rr),arrived={str(t):sum(r['goals'][str(t)]['goal_arrived'] for r in rr) for t in [.5,.75,1.,1.25]},
                survived=sum(r['physical_outcome']=='survived' for r in rr),fates=fates,
                scored_cost_mean_by_alpha={str(a):(float(np.mean([r['scored_cost'] for r in rr if r['alpha']==a]))
                    if any(r['alpha']==a for r in rr) else None) for a in [0.,.5,1.]})
        details=[]
        for g in chosen:
            b,c=[g[m] for m in METHODS]
            details.append(dict(context=b['context'],alpha=b['alpha'],event_id=b['event_id'],force_N=b['force_N'],
                arrival_change=int(c['goals']['1.0']['goal_arrived'])-int(b['goals']['1.0']['goal_arrived']),
                survival_change=int(c['physical_outcome']=='survived')-int(b['physical_outcome']=='survived'),
                minimum_goal_distance_change_m=c['goals']['1.0']['minimum_upright_goal_distance_m']-b['goals']['1.0']['minimum_upright_goal_distance_m'],
                full_minimum_goal_distance_m=b['goals']['1.0']['minimum_upright_goal_distance_m'],
                candidate_minimum_goal_distance_m=c['goals']['1.0']['minimum_upright_goal_distance_m'],
                scored_cost_change=c['scored_cost']-b['scored_cost'],
                common_survivor=all(r['physical_outcome']=='survived' for r in [b,c]),
                original_suppressed_support_ticks=b['suppressed_support_ticks'],
                candidate_override_ticks=c['support_permission_override_ticks']))
        result[key]=dict(methods=methods,
            arrival_gain=sum(d['arrival_change']==1 for d in details),arrival_loss=sum(d['arrival_change']==-1 for d in details),
            survival_gain=sum(d['survival_change']==1 for d in details),survival_loss=sum(d['survival_change']==-1 for d in details),
            detail=details,common_survivor_cost_change_by_alpha={str(a):dict(
                n=sum(d['common_survivor'] and d['alpha']==a for d in details),
                mean=float(np.mean([d['scored_cost_change'] for d in details if d['common_survivor'] and d['alpha']==a]))
                if any(d['common_survivor'] and d['alpha']==a for d in details) else None) for a in [0.,.5,1.]})
    return result


def analyze(root):
    root,plan,cases=inputs(root)
    status=json.loads((root/'status.json').read_text())
    if status.get('phase')!='completed' or status.get('completed_cases')!=108:
        raise ValueError('complete108 numerical results required; do not label incomplete as failed')
    candidates=json.loads((root/'candidate_results.json').read_text())
    baselines=json.loads((root/'baseline_rows.json').read_text())
    if len(candidates)!=108 or len(baselines)!=108:raise ValueError('complete paired inventory required')
    out=root/'analysis';out.mkdir(exist_ok=False)
    atomic_json(out/'status.json',dict(phase='running',new_physics_steps=0))
    config=json.loads((root/'common_environment_config.json').read_text());paths={}
    for ctx,c in plan['contexts'].items():
        if digest(c['path'])!=c['path_sha256']:raise ValueError('original path changed')
        with np.load(c['path']) as z:paths[ctx]={k:z[k] for k in z.files}
    results=[];groups=[]
    for i,(case,new,old) in enumerate(zip(cases,candidates,baselines)):
        if any(new.get(k)!=v for k,v in case.items()) or new['vector']!=[0.]*6:
            raise ValueError('executed candidate identity differs from fixed case')
        group={}
        for row,method in [(old,'full'),(new,'support_permission')]:
            if digest(row['raw_file'])!=row['raw_sha256']:raise ValueError('actual raw changed')
            with np.load(row['raw_file']) as z:raw={k:z[k] for k in z.files}
            path=paths[row['context']]
            # Runtime already audited the complete new action chain. Here
            # independently recalculate rewards/goals and actual pulse profile.
            q=score_raw(raw,row,config,path)
            add_radius_sensitivity(q,raw,path)
            if abs(q['scored_cost']-row['scored_cost'])>1e-8 or q['physical_outcome']!=row['physical_outcome']:
                raise ValueError('saved numeric row differs from raw reward/physical readback')
            active=np.asarray(raw['active_tick'],bool);legal=np.asarray(raw['actor_action_valid'],bool)
            q.update(role=method,inherited=method=='full',pulse_audit=audit_pulse(raw,row),
                suppressed_support_ticks=int((active&~legal&np.asarray(raw['relay_active'],bool)).sum()),
                support_permission_override_ticks=int(np.asarray(raw['support_permission_override'],bool).sum()) if method!='full' else 0)
            q=encode_objective(q);results.append(q);group[method]=q
        groups.append(group)
        atomic_json(out/'status.json',dict(phase='running',completed_pairs=i+1,new_physics_steps=0))
    if sum(r['physical_ticks'] for r in results if r['role']=='support_permission')!=status['active_lower_ticks']:
        raise ValueError('actual physical tick accounting differs')
    summary=paired_summary(groups)
    strata={}
    for ctx in paths:
        for a in [0.,.5,1.]:
            selected=[g for g in groups if g['full']['context']==ctx and g['full']['alpha']==a]
            strata[ctx+f'/a{a}']=paired_summary(selected)
    summary.update(by_context_and_original_alpha=strata,cost=status,
        inherited_actual_lower_ticks=plan['inherited_costs']['parent_total_actual_lower_ticks'],
        cumulative_actual_lower_ticks=plan['inherited_costs']['parent_total_actual_lower_ticks']+status['active_lower_ticks'],
        scope='Seen DEV permission-factor intervention.108 new candidate trajectories vs108 authenticated inherited baseline. Same ancestors; no independent training repeats or safety certificate.',
        source_results_sha256=digest(root/'candidate_results.json'),script_sha256=digest(__file__),visual_review_complete=False)
    atomic_json(out/'results.json',results);atomic_json(out/'summary.json',summary)
    render(out,results,paths)
    lines=['# 原速度支持与Actor权限分离：固定108配对结果','',
        '[全部无扰XY](xy_nominal_+0.png)','',
        '|方法|受扰例数|1m到达|全窗不触地|','|---|---:|---:|---:|']
    for m,q in summary['disturbed']['methods'].items():lines.append(f'|{m}|{q["n"]}|{q["arrived"]["1.0"]}|{q["survived"]}|')
    d=summary['disturbed'];lines+=['',f'受扰到达：{d["arrival_gain"]}增/{d["arrival_loss"]}失；存活：{d["survival_gain"]}增/{d["survival_loss"]}失。',
        '', '108候选全新前向（96受扰+12无扰）；108基线仅继承，不当新样本。原alpha/10s/世界目标1m/参考权限与原relay不变。',
        '支持继续可能只延缓触地或远离任务；连续目标距、全原alpha成本与共同存活条件成本及全部得失见summary.json。',
        '这是一项已见开发面板的权限因子诊断，不是独立测试、物理牛顿阈值、安全证明或新颖性证明。图生成不等于实际目视。']
    for phase in ['early','late']:
        for force in [-16.,16.,-32.,32.]:lines.append(f'\n[全部XY {phase}/{force:g}N](xy_{phase}_{force:+g}.png)')
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'status.json',dict(phase='completed',new_physics_steps=0,logical_rows=216,xy_figures=9,visual_review_complete=False))
    print(json.dumps({k:v for k,v in summary['disturbed'].items() if k!='detail'},ensure_ascii=False))


def render(out,rows,paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    colors=['#222222','#0072B2'];styles=['-','--']
    for phase,force in [('nominal',0.)]+[(p,f) for p in ['early','late'] for f in [-16.,16.,-32.,32.]]:
        fig,axes=plt.subplots(4,3,figsize=(16,15),layout='constrained')
        for i,(ctx,path) in enumerate(paths.items()):
            mask=path['s']<=float(path['goal']);point=[np.interp(float(path['goal']),path['s'],path['xy'][:,j]) for j in range(2)]
            for j,a in enumerate([0.,.5,1.]):
                ax=axes[i,j];ax.plot(path['xy'][mask,0],path['xy'][mask,1],ls=':',color='gray')
                ax.add_patch(Circle(point,1.,fill=False,ec='gray',lw=.6));ax.scatter(*point,c='black',marker='*',s=20)
                selected=[r for r in rows if r['context']==ctx and r['alpha']==a and r['force_N']==force and r['event_id'].split('_')[0]==phase]
                if len(selected)!=2:raise ValueError('both matched lines required')
                for method,color,style in zip(METHODS,colors,styles):
                    r=next(r for r in selected if r['role']==method)
                    with np.load(r['raw_file']) as raw:
                        active=raw['active_tick'].astype(bool);xy=raw['actual_xy'][active]
                    ax.plot(xy[:,0],xy[:,1],c=color,ls=style,lw=1)
                    ax.scatter(*xy[-1],c=color,marker='x' if r['physical_outcome']=='contact' else 'o',s=16)
                title=ctx+f' / alpha{a}'+(f' / t={r["pulse_onset_task_tick"]*.005:.2f}s' if force else '')
                ax.set(title=title,xlabel='World X (m)',ylabel='World Y (m)');ax.title.set_fontsize(7)
                ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
        fig.legend(handles=[Line2D([],[],c=c,ls=s,label=m) for m,c,s in zip(METHODS,colors,styles)],loc='outside lower center',ncol=2)
        fig.suptitle(f'Existing analytic speed-support permission / {phase} / {force:g}N,0.1s\nOriginal goal1m; x=contact, o=real horizon endpoint; matched task0, seen DEV',fontsize=11)
        fig.savefig(out/f'xy_{phase}_{force:+g}.png',dpi=130);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    try:analyze(args.root)
    except BaseException:
        from engineer_reference_bias import notify
        notify(args.root,'STTW支持权限结果分析报错','保留已有数值轨迹，不自动重跑物理；需要检查分析记录。')
        raise
